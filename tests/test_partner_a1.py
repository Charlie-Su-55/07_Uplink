"""Small synthetic checks for paired A1 evaluation; never generate UMa here."""

import copy
import hashlib
import importlib.metadata
import io
import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evaluation import compare_paper_reference_a1 as evaluator
from link_level import partner_a1_adapter as adapter
from link_level.sionna_ce import distribution_id


def fixture():
    gen = torch.Generator().manual_seed(91)
    y = torch.randn(1, 2, 3, 4, generator=gen, dtype=torch.complex64)
    h = torch.randn(1, 2, 3, 4, 2, generator=gen, dtype=torch.complex64)
    return dict(y=y, h_hat=h, err_var=torch.rand(h.shape, generator=gen), n0=torch.tensor(.4),
                Ruu=.4 * torch.eye(4, dtype=torch.complex64)[None], data_indices=torch.tensor([0, 2, 5]),
                coded_bits=torch.randint(0, 2, (1, 2, 12), generator=gen).float(), metadata={"csi": "practical"})


def config():
    sections = ("ofdm", "bs_array", "ut_array", "topology", "channel", "stream_mapping", "link", "paper_contract")
    return dict({key: {} for key in sections}, general={"precision": "single"},
                nr={"mcs_table": 1, "mcs_index": 10}, mode="paper_reference")


class A1AdapterTests(unittest.TestCase):
    def test_ce_whitening_retains_uncertainty_and_never_passes_labels_or_truth(self):
        batch = fixture()
        before = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch["h_true"] = torch.full_like(batch["h_hat"], complex(float("nan"), 0))
        ordinals = torch.tensor([2, 0])
        output = adapter.a1_inputs(batch, ordinals, "sionna_diagonal")
        idx = batch["data_indices"][ordinals]
        y, h = batch["y"].reshape(1, 6, 4)[:, idx], batch["h_hat"].reshape(1, 6, 4, 2)[:, idx]
        variance = .4 + batch["err_var"].reshape(1, 6, 4, 2)[:, idx].sum(-1)
        self.assertEqual(set(output), {"y", "h", "ruu"})
        torch.testing.assert_close(output["y"][:, 0], y / variance.sqrt())
        torch.testing.assert_close(output["h"][:, 0], h / variance.sqrt()[..., None])
        torch.testing.assert_close(output["ruu"], torch.eye(4, dtype=torch.complex64)[None])
        for key, value in before.items():
            torch.testing.assert_close(batch[key], value, atol=0, rtol=0)
        raw = adapter.a1_inputs(batch, ordinals, "thermal_only")
        torch.testing.assert_close(raw["y"][:, 0], y)
        self.assertIs(raw["ruu"], batch["Ruu"])
        batch["metadata"]["csi"] = "perfect"
        with self.assertRaisesRegex(ValueError, "practical"):
            adapter.a1_inputs(batch, ordinals, "sionna_diagonal")

    def test_native_codeword_order_round_trip_and_subset(self):
        llr = torch.arange(1 * 2 * 1 * 12).reshape(1, 2, 1, 12).float()
        full = evaluator.selected_native_llrs(llr, torch.arange(3))
        torch.testing.assert_close(evaluator.llrs_to_native_order(full), llr)
        selected = evaluator.selected_native_llrs(llr, torch.tensor([2, 0]))
        torch.testing.assert_close(selected[0, 0, 1], llr[0, 1, 0, 8:12])
        torch.testing.assert_close(selected[0, 1, 0], llr[0, 0, 0, :4])

    def test_legacy_wrong_prior_and_train_validation_overlap_rejected(self):
        cfg = config()
        checkpoint = dict(kind="paper_reference_detector_v1", arch="gt_ep", bits_per_symbol=4,
                          mcs_table=1, mcs_index=10, csi="practical", custom_ce_policy="sionna_diagonal",
                          distribution_id=distribution_id(cfg), ce_covariance_sha256="prior-hash",
                          llr_convention="log(P1/P0)", system_config=cfg, train_seeds=[1, 2],
                          validation_seeds=[3], step=300)
        adapter.validate_neural_checkpoint(checkpoint, "gt_ep", cfg, "prior-hash", [8, 9])
        for key, value in (("kind", "legacy"), ("arch", "detr_ep"), ("ce_covariance_sha256", "other"),
                           ("custom_ce_policy", "none"), ("distribution_id", "other"), ("train_seeds", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate_neural_checkpoint(dict(checkpoint, **{key: value}), "gt_ep", cfg, "prior-hash", [8])
        for seeds in ([1], [3]):
            with self.assertRaisesRegex(ValueError, "overlap"):
                adapter.validate_neural_checkpoint(checkpoint, "gt_ep", cfg, "prior-hash", seeds)

    def test_bundle_tamper_and_escaping_manifest_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {"model.pt": b"test-only-placeholder", "src/r3p5_invariant/portable/runtime.py": b"# fixture\n"}
            sha = hashlib.sha256(files["model.pt"]).hexdigest()
            files["MODEL.json"] = json.dumps(dict(format="a1_partner_minimal_v1", variant="combined",
                                                 step=6000, model_file_sha256=sha)).encode()
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            manifest = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
            (root / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
            result = adapter.verify_bundle(root)
            self.assertFalse(result["mode_b_training_distribution_verified"])
            (root / "model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                adapter.verify_bundle(root)
            (root / "model.pt").write_bytes(files["model.pt"])
            (root / "BUNDLE_MANIFEST.json").write_text(json.dumps(dict(manifest, **{"../outside.py": "bad"})))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                adapter.verify_bundle(root)

    def test_neural_methods_share_exact_statistics_once_per_chunk(self):
        batch, seen = fixture(), []

        class TinyModel:
            def __call__(self, z, gram, **kwargs):
                seen.append((id(z), id(gram)))
                return {"llr": z.real[..., None].expand(-1, -1, 4)}

        output, timing = evaluator.evaluate_neural({"gt_ep": TinyModel(), "detr_ep": TinyModel()},
                                                   batch, torch.arange(3), 2)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(seen[2], seen[3])
        torch.testing.assert_close(output["gt_ep"], output["detr_ep"])
        self.assertEqual(output["gt_ep"].shape, (1, 3, 2, 4))
        self.assertGreaterEqual(timing["shared_frontend_seconds"], 0)


class PairedEvaluatorTests(unittest.TestCase):
    def test_one_transmit_shared_methods_and_no_subset_decoding(self):
        # Real evaluator orchestration and adapter ordering on tiny CPU tensors.
        class FakeLink:
            device = "cpu"
            data_indices = torch.tensor([0, 2, 5])
            covariance_metadata = dict(sha256="prior-hash", calibration_seeds=[4000000])

            def __init__(self, cfg):
                self.cfg, self.calls, self.decoded_calls = cfg, [], 0
                self.codec = SimpleNamespace(qm=4, coded_bits=9216, info_bits=3104, decode=self.decode)

            def decode(self, llr):
                self.decoded_calls += 1
                return torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool)

            def transmit(self, size, ebno, seed):
                self.calls.append((size, ebno, seed))
                return dict(info=torch.zeros(1, 2, 3), seed=seed)

            def check_codec_identity(self, sample):
                return {"status": "pass"}

            def metadata(self):
                return {"synthetic_test_only": True}

        batch = fixture()
        native = evaluator.llrs_to_native_order(2 * evaluator.selected_bits(batch, torch.arange(3)) - 1)
        identities = []

        class FakeClassical:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self, sample, received_batch):
                identities.append(id(sample))
                return {name: dict(llr=native, decoded=sample["info"], crc_ok=torch.ones(1, 2, dtype=torch.bool))
                        for name in ("sionna_lmmse", "custom_lmmse", "ep5")}

        class FakePartner:
            metadata = {"synthetic_test_only": True}
            metrics = SimpleNamespace(
                soft_metrics=lambda llr, bits: dict(bits=bits.numel(), bit_errors=int(((llr > 0) != bits.bool()).sum()),
                                                    ber=float(((llr > 0) != bits.bool()).float().mean()), bce_sum_nats=1.),
                cluster_comparison=lambda a, b: {"channels": len(a)})

            def __init__(self, *args):
                self.fail_at = None
                self.calls = 0

            def check_constellation(self, link):
                pass

            def evaluate(self, received_batch, ordinals, policy, seed):
                self.calls += 1
                if self.calls == self.fail_at:
                    raise RuntimeError("injected sampler failure")
                self.assert_same = received_batch is batch
                llr = evaluator.selected_native_llrs(native, ordinals)
                return dict(raw_k64=llr, raw_k256=llr, lmmse=llr), {"synthetic_test_only": True}

        with tempfile.TemporaryDirectory() as directory:
            for subset in (False, True):
                link, partner = FakeLink(config()), FakePartner()
                output = Path(directory) / ("subset.json" if subset else "coded.json")
                args = ["--ce-covariance", "prior.pt", "--channels", "2", "--ebno-dbs=-12,-9", "--seed", "80",
                        "--output", str(output)] + (["--re-per-channel", "2"] if subset else [])
                with (patch.object(evaluator, "training_config", return_value=config()),
                      patch.object(evaluator, "PartnerA1Adapter", return_value=partner),
                      patch.object(evaluator, "SionnaReferenceLink", return_value=link),
                      patch.object(evaluator, "ClassicalDetectorAdapter", FakeClassical),
                      patch.object(evaluator, "make_detector_batch", return_value=batch),
                      patch.object(evaluator, "csi_diagnostics", return_value={}),
                      patch.object(evaluator, "compare_lmmse_outputs", return_value={}),
                      patch.object(evaluator, "provenance", return_value={}),
                      patch("sys.stdout", new_callable=io.StringIO)):
                    evaluator.main(args)
                    self.assertEqual(link.calls, [(1, -12., 80), (1, -12., 81), (1, -9., 80), (1, -9., 81)])
                    self.assertEqual(partner.calls, 4)
                    self.assertTrue(partner.assert_same)
                    self.assertEqual(link.decoded_calls, 0 if subset else 12)
                    with self.assertRaises(FileExistsError):
                        evaluator.main(args)
                    report = json.loads(output.read_text())
                    self.assertEqual(report["status"], "complete")
                    self.assertEqual(report["coded_bler_evaluated"], not subset)
                    for point in report["points"]:
                        self.assertEqual(len(point["channels"]), 2)
                        for method in point["receivers"].values():
                            self.assertEqual("coded" in method, not subset)
                    partner.fail_at = partner.calls + 2
                    failed = Path(directory) / ("failed_subset.json" if subset else "failed_coded.json")
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        evaluator.main(args + ["--output", str(failed)])
                    partial = json.loads(failed.read_text())
                    self.assertEqual(partial["status"], "failed")
                    self.assertEqual(len(partial["points"][0]["channels"]), 1)
                    self.assertTrue(failed.with_suffix(".csv").exists())


class ExternalA1CpuTests(unittest.TestCase):
    def test_frozen_weights_original_sampler_and_k64_prefix_on_one_synthetic_re(self):
        root = Path(__file__).resolve().parents[2] / "A1_PARTNER_MINIMAL"
        if not root.is_dir():
            self.skipTest("External A1 bundle is not installed beside the repository")
        adapter.verify_bundle(root)
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
            from r3p5_invariant.portable.common import state_hash

            torch.set_num_threads(2)
            payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
            qam = QAM(4)
            model = SingleREFlow(DiscretePosteriorFlowConfig(**payload["architecture"]),
                                 QAMTable(qam.points, qam.bits.float()), "combined")
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval().requires_grad_(False)
            self.assertEqual(state_hash(model.state_dict()), payload["weights_sha256"])
            # Test-only interface fixture: no jsonschema/Sionna baseline dependency.
            # Actual runtime Contract.build and native LMMSE are exercised on server.
            interface = SimpleNamespace(document=dict(grid=[1, 1], rx=256, stream_capacity=16,
                                                     active_streams=16, qm=4, dtype="complex64"))
            detector = A1Detector(model, interface, samples=256, sample_chunk=64, re_tile=64, seed=73)
            gen = torch.Generator().manual_seed(44)
            y = torch.randn(1, 1, 1, 256, dtype=torch.complex64, generator=gen)
            h = torch.randn(1, 1, 1, 256, 16, dtype=torch.complex64, generator=gen) / 16
            ruu = torch.eye(256, dtype=torch.complex64)[None]
            with torch.inference_mode():
                prepared = detector.prepare(y, h, ruu)
                population = sample_population(detector, prepared, prefixes=(64,))
                large = detector.readout(population, (1, 1, 1), torch.device("cpu"))
                prefix = {"rao_blackwell_probability": population["rao_blackwell_probability_prefixes"][64]}
                small = detector.readout(prefix, (1, 1, 1), torch.device("cpu"))
                detector.samples = 64  # Independent prefix qualification, not production settings.
                direct = detector.readout(sample_population(detector, prepared), (1, 1, 1), torch.device("cpu"))
            self.assertEqual(large.shape, (1, 1, 1, 16, 4))
            self.assertTrue(torch.isfinite(large).all() and torch.isfinite(small).all())
            torch.testing.assert_close(small, direct, atol=0, rtol=0)
        finally:
            sys.path.pop(0)
            sys.dont_write_bytecode = old_bytecode
            torch.set_num_threads(old_threads)


    def test_native_runtime_three_inputs_and_lmmse_uncertainty_on_one_re(self):
        root = Path(__file__).resolve().parents[2] / "A1_PARTNER_MINIMAL"
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        if not root.is_dir() or version != "2.0.1":
            self.skipTest("Requires external A1 bundle and native Sionna 2.0.1")
        from detectors.classical.lmmse import LMMSESoftDetector
        old_threads = torch.get_num_threads()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = (torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        try:
            partner = adapter.PartnerA1Adapter(root, "cpu")
            gen = torch.Generator().manual_seed(19)
            batch = dict(y=torch.randn(1, 1, 1, 256, dtype=torch.complex64, generator=gen),
                         h_hat=torch.randn(1, 1, 1, 256, 16, dtype=torch.complex64, generator=gen) / 16,
                         err_var=torch.full((1, 1, 1, 256, 16), .02), n0=torch.tensor(.4),
                         Ruu=.4 * torch.eye(256, dtype=torch.complex64)[None], data_indices=torch.tensor([0]),
                         metadata={"csi": "practical"})
            ordinals = torch.tensor([0])
            output, timing = partner.evaluate(batch, ordinals, "sionna_diagonal", 53)
            transformed = adapter.a1_inputs(batch, ordinals, "sionna_diagonal")
            cfg = dict(general=dict(precision="single", device="cpu"), modulation={"bits_per_symbol": 4})
            custom = LMMSESoftDetector(cfg)(transformed["y"], transformed["h"], transformed["ruu"])
            torch.testing.assert_close(output["lmmse"], custom["llr"][:, 0], atol=2e-3, rtol=2e-3)
            self.assertEqual(output["raw_k64"].shape, (1, 1, 16, 4))
            self.assertTrue(torch.isfinite(output["raw_k256"]).all())
            self.assertEqual(timing["input_policy"], "sionna_diagonal")
        finally:
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old_cudnn


if __name__ == "__main__":
    unittest.main()
