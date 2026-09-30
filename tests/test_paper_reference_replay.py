"""Small replay contract checks; no UMa generation or full 256Rx slot evaluation."""

import copy
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evaluation import export_paper_reference_replay as replay


def fixture():
    # Unequal axis lengths and tagged coordinates expose accidental permutations.
    b, t, f, m, k = 2, 3, 5, 4, 2
    h = torch.arange(b * t * f * m * k).reshape(b, t, f, m, k).to(torch.complex64)
    h = h + .125j * h
    y = torch.arange(b * t * f * m).reshape(b, t, f, m).to(torch.complex64)
    indices = torch.tensor([0, 2, 5, 9, 14])
    bits = torch.arange(b * k * 20).reshape(b, k, 20).remainder(2).float()
    batch = dict(y=y, h_hat=h, h_true=2 * h, err_var=h.real / 1000 + .01,
                 n0=torch.tensor(.4), Ruu=.4 * torch.eye(m, dtype=torch.complex64).expand(b, -1, -1),
                 data_indices=indices, coded_bits=bits, bits=bits[..., :7].clone(), metadata={"csi": "practical"})
    sample = dict(y=y.permute(0, 3, 1, 2).unsqueeze(1),
                  y_clean=(y - .125j).permute(0, 3, 1, 2).unsqueeze(1),
                  x=torch.arange(b * k * t * f).reshape(b, k, 1, t, f).to(torch.complex64))
    return sample, batch


class HardBitCodec:
    """Test double with a separate CRC decision; not an NR decoder qualification."""
    def decode(self, llr):
        bits = (llr.squeeze(2)[..., :7] > 0).float()
        return bits, (bits.sum(-1).long() % 2 == 0)


class ReplayContractTests(unittest.TestCase):
    def test_all_codeword_positions_and_axes_roundtrip(self):
        llr = torch.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4).float() - 57
        native = replay.llrs_to_native_order(llr)
        self.assertEqual(native.shape, (2, 3, 1, 20))
        for b in range(2):
            for u in range(3):
                for n in range(5):
                    for q in range(4):
                        self.assertEqual(native[b, u, 0, n * 4 + q], llr[b, n, u, q])
        self.assertTrue(torch.equal(replay.selected_native_llrs(native, torch.arange(5)), llr))

    def test_data_coordinates_truth_noise_and_a1_whitening(self):
        sample, batch = fixture()
        saved, axes = replay.capture_inputs(sample, batch)
        schema = replay.tensor_schema({"inputs": saved}, axes)
        self.assertEqual(schema["inputs.h_hat"]["shape"], [2, 1, 5, 4, 2])
        self.assertEqual(schema["inputs.y"]["dtype"], "torch.complex64")
        for n, flat in enumerate(batch["data_indices"].tolist()):
            t, f = divmod(flat, 5)
            self.assertTrue(torch.equal(saved["y"][:, 0, n], batch["y"][:, t, f]))
            self.assertTrue(torch.equal(saved["h_hat"][:, 0, n], batch["h_hat"][:, t, f]))
            self.assertTrue(torch.equal(saved["h_true"][:, 0, n], batch["h_true"][:, t, f]))
            self.assertTrue(torch.equal(saved["err_var"][:, 0, n], batch["err_var"][:, t, f]))
            self.assertTrue(torch.equal(saved["qam_symbols"][..., n], sample["x"][..., t, f]))
        self.assertTrue(torch.equal(saved["Ruu"], batch["Ruu"]))
        self.assertTrue(torch.equal(saved["coded_bits"], batch["coded_bits"]))
        self.assertTrue(torch.equal(saved["info_bits"], batch["bits"]))
        self.assertTrue(torch.equal(saved["realized_noise"], saved["y"] - saved["y_clean"]))
        scale = (.4 + saved["err_var"].sum(-1)).sqrt()
        self.assertTrue(torch.equal(saved["a1_inputs"]["y"], saved["y"] / scale))
        self.assertTrue(torch.equal(saved["a1_inputs"]["h"], saved["h_hat"] / scale[..., None]))
        self.assertEqual(set(saved["a1_inputs"]), {"y", "h", "ruu"})
        saved["y"].zero_()
        self.assertGreater(batch["y"].abs().sum().item(), 0)  # independent CPU storage

    def test_truth_and_labels_cannot_influence_a1_inputs(self):
        sample, batch = fixture()
        original, _ = replay.capture_inputs(sample, batch)
        batch["h_true"].fill_(complex(float("nan"), 0))
        batch["coded_bits"].fill_(float("nan"))
        batch["bits"].fill_(float("nan"))
        changed, _ = replay.capture_inputs(sample, batch)
        replay.assert_tensor_tree_equal(original["a1_inputs"], changed["a1_inputs"])
        self.assertEqual(set(changed["a1_inputs"]), {"y", "h", "ruu"})

    def test_serialized_llrs_decode_bits_crc_and_per_ue_errors(self):
        generator = torch.Generator().manual_seed(42)
        llr = torch.randn(2, 5, 3, 4, generator=generator)
        native = replay.llrs_to_native_order(llr)
        codec = HardBitCodec()
        decoded, crc = codec.decode(native)
        info = decoded.clone()
        info[0, 0, :3] = 1 - info[0, 0, :3]
        methods, axes = {}, {}
        for name in replay.METHODS:
            record, record_axes = replay.method_record(name, llr, native, decoded, crc, info,
                                                       native if name in replay.METHODS[:3] else None)
            methods[name] = record
            axes.update(record_axes)
            self.assertEqual(record["payload_bit_errors"].tolist(), [[3, 0, 0], [0, 0, 0]])
            self.assertTrue(torch.equal(record["block_error"], record["payload_bit_errors"] > 0))
            self.assertTrue(torch.equal(record["crc_ok"], crc))
            self.assertTrue(torch.all(record["total_payload_bits_per_ue"] == 7))
            self.assertTrue(torch.equal(record["decoder_input_llr"], native))
        payload = dict(methods=methods)
        replay.tensor_schema(payload, axes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.pt"
            record = replay.save_tensor_file(path, payload, Path(directory))
            self.assertEqual(record["sha256"], replay.file_sha256(path))
            loaded = replay.verify_serialized_decodes(path, codec, "cpu")
            replay.assert_tensor_tree_equal(payload, loaded)
            loaded["methods"]["ep5"]["crc_ok"] = ~loaded["methods"]["ep5"]["crc_ok"]
            torch.save(loaded, path)
            with self.assertRaisesRegex(ValueError, "decode mismatch"):
                replay.verify_serialized_decodes(path, codec, "cpu")

    def test_determinism_checks_dtype_shape_and_each_tensor(self):
        sample, batch = fixture()
        first, _ = replay.capture_inputs(sample, batch)
        second, _ = replay.capture_inputs(sample, batch)
        replay.assert_tensor_tree_equal(first, second)
        second["h_hat"][0, 0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "h_hat"):
            replay.assert_tensor_tree_equal(first, second)

    def test_runtime_metadata_reads_actual_fields_without_crc_guess(self):
        crc = SimpleNamespace(crc_degree="test_CRC", crc_length=23, k=71, n=94)
        enc = SimpleNamespace(tb_crc_encoder=crc, _n_rnti=[7, 9], _n_id=[101, 103])
        codec = SimpleNamespace(encoder=enc, decoder=SimpleNamespace(), metadata=lambda: {"num_bp_iter": 17})
        result = replay.codec_manifest(codec)
        self.assertEqual(result["runtime"]["tb_crc_encoder"]["attributes"]["crc_length"], 23)
        self.assertEqual(result["runtime"]["encoder"]["attributes"]["_n_rnti"], [7, 9])
        self.assertEqual(result["info_bits_transport_block_crc_membership"], replay.UNKNOWN_CRC)
        self.assertIn("k", result["runtime"]["encoder"]["unavailable"])

    def test_source_validation_checks_each_ue_and_integer_count(self):
        coded = dict(block_errors=1, bit_errors=3, per_ue_block_errors=[0, 1], bits=14, blocks=2)
        row = dict(seed=20260957, sampling_seed=23260964, data_ordinals=list(range(2304)),
                   receivers={name: dict(coded=copy.deepcopy(coded), soft=dict(bits=40, bit_errors=4))
                              for name in replay.METHODS})
        source = dict(status="complete", coded_bler_evaluated=True, selected_data_re=2304,
                      policies=["sionna_diagonal"], classical_and_gt_detr_ce_policy="sionna_diagonal",
                      points=[dict(ebno_db=-11., status="complete", channels=[row])])
        self.assertEqual(replay.select_source_cases(source, [(-11., 20260957)]), [row])
        actual = copy.deepcopy(row["receivers"])
        self.assertEqual(replay.source_mismatches(row, actual), [])
        actual["gt_ep"]["coded"]["per_ue_block_errors"] = [1, 0]
        self.assertIn("per_ue_block_errors", replay.source_mismatches(row, actual)[0])
        row["data_ordinals"][0], row["data_ordinals"][1] = 1, 0
        with self.assertRaisesRegex(ValueError, "ordering"):
            replay.select_source_cases(source, [(-11., 20260957)])

    def test_default_diagnostics_choose_one_rescue_and_one_harm(self):
        rows = [dict(receivers=dict(gt_ep=dict(coded=dict(block_errors=gt)),
                                    a1_ce_k256=dict(coded=dict(block_errors=a1))))
                for gt, a1 in ((2, 0), (4, 2), (1, 0), (0, 1), (0, 1), (0, 1), (6, 5), (0, 0))]
        self.assertEqual(replay.diagnostics_selection(replay.DEFAULT_CASES, rows, "representative"),
                         {replay.DEFAULT_CASES[0], replay.DEFAULT_CASES[3]})
        self.assertEqual(replay.diagnostics_selection(replay.DEFAULT_CASES, rows, "none"), set())

    def test_disk_projection_uses_serialized_sizes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = replay.save_tensor_file(root / "case.pt", {"y": torch.ones(1, 3, dtype=torch.complex64)}, root)
            diag = replay.save_tensor_file(root / "diag.pt", {"tokens": torch.ones(256, dtype=torch.uint8)}, root)
            with self.assertRaises(FileExistsError):
                replay.save_tensor_file(root / "case.pt", {}, root)
            usage = replay.disk_usage(root, {"cases": [dict(file=case, diagnostics=dict(file=diag))]})
            self.assertEqual(usage["estimated_eight_cases_plus_two_diagnostics_pt_bytes"],
                             8 * case["bytes"] + 2 * diag["bytes"])

    def test_export_orchestration_shares_one_sample_and_records_reload_and_hashes(self):
        sample, batch = fixture()
        sample.update(info=batch["bits"], coded=batch["coded_bits"], seed=20260957, ebno_db=-11.)
        evaluator_llr = 2 * replay.selected_bits(batch, torch.arange(5)) - 1
        native = replay.llrs_to_native_order(evaluator_llr)
        codec = HardBitCodec()
        decoded, crc = codec.decode(native)
        cfg = dict(mode="paper_reference", general=dict(device="cpu"),
                   channel_estimation=dict(mode="practical", covariance_path=""))
        codec.qm, codec.coded_bits, codec.info_bits = 4, 9216, 3104  # Preflight stub, no large tensors.
        codec.encoder, codec.decoder = SimpleNamespace(), SimpleNamespace()
        codec.metadata = lambda: {"synthetic_test_only": True}
        reference = dict(config=cfg, grid={"synthetic_test_only": True}, codec=codec.metadata())
        transmitted, same_samples, same_batches = [], [], []

        def transmit(size, ebno, seed):
            transmitted.append((size, ebno, seed))
            return sample

        link = SimpleNamespace(codec=codec, device="cpu", cfg=cfg, metadata=lambda: reference, transmit=transmit)

        def classical_evaluate(actual_sample, actual_batch):
            same_samples.append(actual_sample is sample)
            same_batches.append(actual_batch is batch)
            return {name: dict(llr=native, decoded=decoded, crc_ok=crc) for name in replay.METHODS[:3]}

        def neural_evaluate(models, actual_batch, ordinals, chunk_size):
            same_batches.append(actual_batch is batch)
            return {name: evaluator_llr for name in ("gt_ep", "detr_ep")}, {}

        def partner_evaluate(actual_batch, ordinals, policy, seed):
            same_batches.append(actual_batch is batch)
            values = replay.a1_inputs(actual_batch, ordinals, policy)
            self.assertEqual(set(values), {"y", "h", "ruu"})
            self.assertEqual(seed, 23260964)
            return {name: evaluator_llr for name in ("raw_k64", "raw_k256", "lmmse")}, {}

        partner = SimpleNamespace(metadata={}, check_constellation=lambda link: None, evaluate=partner_evaluate,
                                  metrics=SimpleNamespace(soft_metrics=lambda llr, bits:
                                      dict(bits=bits.numel(), bit_errors=int(((llr > 0) != bits.bool()).sum()))))
        counts = replay.block_statistics(sample, dict(decoded=decoded, crc_ok=crc))
        row = dict(seed=20260957, sampling_seed=23260964, data_ordinals=list(range(2304)),
                   receivers={name: dict(coded=counts, soft=dict(bits=batch["coded_bits"].numel(), bit_errors=0))
                              for name in replay.METHODS})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, gt, detr = (root / name for name in ("prior.pt", "gt.pt", "detr.pt"))
            for path in (prior, gt, detr):
                path.write_bytes(b"synthetic hash fixture, never loaded")
            reference["covariance"] = dict(sha256=replay.file_sha256(prior))
            source = dict(status="complete", coded_bler_evaluated=True, selected_data_re=2304,
                          policies=["sionna_diagonal"], classical_and_gt_detr_ce_policy="sionna_diagonal",
                          points=[dict(ebno_db=-11., status="complete", channels=[row])], reference=reference,
                          distribution_id="synthetic", a1=dict(manifest_sha256="synthetic", model={}),
                          neural_checkpoints={name: dict(sha256=replay.file_sha256(path))
                                              for name, path in (("gt_ep", gt), ("detr_ep", detr))},
                          provenance=dict(arguments=dict(re_chunk=13)))
            src = root / "source.json"
            src.write_text(json.dumps(source), encoding="utf-8")
            original = src.read_bytes()
            output = root / "replay"
            args = ["--source-comparison", str(src), "--output-dir", str(output), "--case=-11:20260957",
                    "--ce-covariance", str(prior), "--gt-checkpoint", str(gt), "--detr-checkpoint", str(detr),
                    "--device", "cpu", "--diagnostics", "none"]
            with (patch.object(replay, "SionnaReferenceLink", return_value=link),
                  patch.object(replay, "PartnerA1Adapter", return_value=partner),
                  patch.object(replay, "verify_bundle", return_value=source["a1"]),
                  patch.object(replay, "load_neural_checkpoint", return_value=(object(), {})),
                  patch.object(replay, "distribution_id", return_value="synthetic"),
                  patch.object(replay, "ClassicalDetectorAdapter", return_value=SimpleNamespace(evaluate=classical_evaluate)),
                  patch.object(replay, "make_detector_batch", return_value=batch),
                  patch.object(replay, "evaluate_neural", side_effect=neural_evaluate),
                  patch.object(replay, "provenance", return_value={}), patch("sys.stdout", new_callable=io.StringIO)):
                self.assertEqual(replay.main(args), 0)
                self.assertEqual(transmitted, [(1, -11., 20260957)] * 2)  # Second call is verification only.
                self.assertTrue(all(same_samples + same_batches))
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertTrue(manifest["cases"][0]["serialized_decode_verified"])
                self.assertEqual(manifest["cases"][0]["source_mismatches"], [])
                file = manifest["cases"][0]["file"]
                self.assertEqual(file["sha256"], replay.file_sha256(output / file["path"]))
                self.assertEqual(len(list((output / "cases").glob("*.pt"))), 1)
                self.assertEqual(src.read_bytes(), original)
                with self.assertRaises(FileExistsError):
                    replay.main(args)
                # A different source result must leave a failed, inspectable package.
                row["receivers"]["gt_ep"]["coded"] = dict(counts, bit_errors=999)
                src.write_text(json.dumps(source), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs from source"):
                    replay.main(args + ["--output-dir", str(root / "failed")])
                failed = json.loads((root / "failed" / "manifest.json").read_text())
                self.assertEqual(failed["status"], "failed")
                self.assertTrue(failed["cases"][0]["source_mismatches"])

    def test_frozen_comparator_is_byte_identical(self):
        self.assertEqual(replay.file_sha256(replay.COMPARATOR), replay.COMPARATOR_SHA256)


class NativeReplayTests(unittest.TestCase):
    def setUp(self):
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        if not version.startswith("2."):
            self.skipTest("Requires existing server Sionna 2.x; no local install/upgrade")

    def test_small_native_practical_replay_and_real_tb_decoder(self):
        from test_paper_reference import FlatChannel, paper_fixture
        from link_level.sionna_ce import COVARIANCE_KIND, distribution_id, regularize_covariance

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                cfg = paper_fixture()
                cfg["channel_estimation"]["mode"] = "practical"
                path = Path(directory) / "cov.pt"
                torch.save(dict(kind=COVARIANCE_KIND, distribution_id=distribution_id(cfg),
                                calibration_seeds=[4000000],
                                cov_mat_freq=regularize_covariance(4 * torch.ones(32, 32, dtype=torch.complex64)),
                                cov_mat_time=regularize_covariance(4 * torch.ones(4, 4, dtype=torch.complex64))), path)
                cfg["channel_estimation"]["covariance_path"] = str(path)
                link = replay.SionnaReferenceLink(cfg, channel_provider=FlatChannel())
                saved, native_llrs = [], []
                for _ in range(2):
                    sample = link.transmit(1, 12., 42)
                    batch = replay.make_detector_batch(link, sample, "practical")
                    inputs, axes = replay.capture_inputs(sample, batch)
                    output = link.receive(sample, "practical")
                    llr = output["llr"]
                    record, method_axes = replay.method_record("sionna_lmmse",
                        replay.selected_native_llrs(llr, torch.arange(64)), llr,
                        output["decoded"], output["crc_ok"], sample["info"], llr)
                    data = dict(inputs=inputs, methods={"sionna_lmmse": record})
                    replay.tensor_schema(data, dict(axes, **method_axes))
                    torch.save(data, Path(directory) / "replay.pt")
                    replay.verify_serialized_decodes(Path(directory) / "replay.pt", link.codec, "cpu")
                    saved.append(inputs)
                    native_llrs.append(llr)
                replay.assert_tensor_tree_equal(saved[0], saved[1])
                self.assertTrue(torch.equal(native_llrs[0], native_llrs[1]))
                actual = replay.codec_manifest(link.codec)
                self.assertEqual(actual["runtime"]["tb_crc_encoder"]["attributes"]["crc_length"],
                                 link.codec.encoder.tb_crc_encoder.crc_length)
        finally:
            torch.set_num_threads(previous_threads)


class ExternalPopulationTests(unittest.TestCase):
    def test_original_population_export_prefix_repeatability_and_bundle_unchanged(self):
        root = Path(__file__).resolve().parents[2] / "A1_PARTNER_MINIMAL"
        if not root.is_dir():
            self.skipTest("Read-only external A1 bundle not present")
        before = {p.relative_to(root).as_posix(): replay.file_sha256(p) for p in root.rglob("*") if p.is_file()}
        identity = replay.verify_bundle(root)
        old_threads, old_bytecode = torch.get_num_threads(), sys.dont_write_bytecode
        sys.path.insert(0, str(root / "src"))
        sys.dont_write_bytecode = True
        try:
            from r3p5_invariant.models.single_re import SingleREFlow
            from r3p5_invariant.models.a1.model_config import DiscretePosteriorFlowConfig
            from r3p5_invariant.models.a1.constellation import QAMTable
            from r3p5_invariant.models.a1_detector import A1Detector
            from r3p5_invariant.models.a1_execution import sample_population
            from r3p5_invariant.physics.qam import QAM

            torch.set_num_threads(2)
            weights = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
            qam = QAM(4)
            model = SingleREFlow(DiscretePosteriorFlowConfig(**weights["architecture"]),
                                 QAMTable(qam.points, qam.bits.float()), "combined")
            model.load_state_dict(weights["model_state"], strict=True)
            model.eval().requires_grad_(False)
            # Test-only interface avoids optional jsonschema/Sionna baseline imports.
            # Production exporter always calls the unmodified runtime.detector_for.
            def detector_for(model, y, h, seed):
                interface = SimpleNamespace(document=dict(grid=[1, 1], rx=256, stream_capacity=16,
                                                          active_streams=16, qm=4, dtype="complex64"))
                return A1Detector(model, interface, samples=256, sample_chunk=64, re_tile=64, seed=seed)

            partner = SimpleNamespace(model=model, device=torch.device("cpu"),
                                      runtime=SimpleNamespace(detector_for=detector_for, sample_population=sample_population))
            generator = torch.Generator().manual_seed(73)
            values = dict(y=torch.randn(1, 1, 1, 256, dtype=torch.complex64, generator=generator),
                          h=torch.randn(1, 1, 1, 256, 16, dtype=torch.complex64, generator=generator) / 16,
                          ruu=torch.eye(256, dtype=torch.complex64)[None])
            with torch.inference_mode():
                detector = detector_for(model, values["y"], values["h"], 53)
                population = sample_population(detector, detector.prepare(**values), prefixes=(64,))
                predictions = {"a1_ce_k256": detector.readout(population, (1, 1, 1), partner.device)[:, 0],
                               "a1_ce_k64": detector.readout(dict(rao_blackwell_probability=
                                   population["rao_blackwell_probability_prefixes"][64]), (1, 1, 1), partner.device)[:, 0]}
                exported, reason = replay.a1_population_diagnostics(partner, values, 53, predictions)
            self.assertIsNone(reason)
            self.assertEqual(exported["candidate_symbols"].shape, (1, 256, 16))
            self.assertTrue(torch.equal(exported["candidate_tokens"], population["candidate_tokens"]))
            self.assertTrue(torch.equal(exported["log_q"], population["path_log_q"]))
            self.assertTrue(torch.equal(exported["k64_prefix_rb_probabilities"],
                                        population["rao_blackwell_probability_prefixes"][64]))
            with self.assertRaisesRegex(ValueError, "only y/h/ruu"):
                replay.a1_population_diagnostics(partner, dict(values, h_true=values["h"]), 53, predictions)
            replay.verify_protected_files({str(replay.COMPARATOR): replay.COMPARATOR_SHA256}, root, identity)
        finally:
            sys.path.pop(0)
            sys.dont_write_bytecode = old_bytecode
            torch.set_num_threads(old_threads)
        after = {p.relative_to(root).as_posix(): replay.file_sha256(p) for p in root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
