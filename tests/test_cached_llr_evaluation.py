"""Cached conditional-scale evaluation contracts without UMa, legacy models, or A1."""

import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation import cached_llr_common as common
from evaluation import evaluate_cached_llr_scaling as evaluation
from models.conditional_llr_scaling import CANDIDATES, make_model
from training.train_cached_llr_scaling import write_manifest


class ToyCodec:
    def decode(self, native):
        llr = native.squeeze(2)
        decoded = (llr[..., :7] > 0).float()
        decoded[..., 0] = torch.where(llr.abs().mean(-1) < .75, 1 - decoded[..., 0], decoded[..., 0])
        return decoded, decoded.sum(-1).long().remainder(2) == 0


def configuration(mode="smoke"):
    config = common.load_config(mode=mode)
    config["runtime"]["device"] = "cpu"
    config["bootstrap"]["replicates"] = 30
    config["evaluation_chunk"] = 2
    return config


def fixture():
    generator = torch.Generator().manual_seed(7)
    coded = torch.randint(2, (1, 3, 20), generator=generator).float()
    channel = dict(split="calibration", seed=81000024, ebno_db=-10.5,
                   data_indices=torch.tensor([1, 4, 8, 10, 12]), coded_bits=coded, info_bits=coded[..., :7].clone())
    labels = evaluation.coded_labels(channel)
    llr = (2 * labels - 1) * .4
    native = evaluation.llrs_to_native_order(llr)
    decoded, crc = ToyCodec().decode(native)
    per_ue, _ = evaluation.payload_statistics(channel["info_bits"], decoded, crc)
    channel["methods"] = dict(ep5=dict(llr=llr, decoder_input_llr=native, **per_ue))
    channel["features"] = dict(ep5_mean=torch.randn(1, 5, 3, dtype=torch.complex64, generator=generator),
                               ep5_variance=torch.full((1, 5, 3), .3),
                               gram=torch.eye(3, dtype=torch.complex64).expand(1, 5, -1, -1).clone(),
                               residual=torch.ones(1, 5), eta=torch.full((1, 5, 3), .1))
    return channel


def normalization():
    return dict(mean=[0.] * 6, std=[1.] * 6, feature_names=common.FEATURE_NAMES, fit_split="train",
                train_seeds=[81000000], fit_sources=[dict(file="train.pt", sha256="train-hash")])


def receivers(config):
    result = [dict(method="ep5_raw", candidate="ep5", training_seed=None, checkpoint_type="raw", step=None, parameter_count=0, alpha=[1.]),
              dict(method="global_train24", candidate="global_train24", training_seed=None, checkpoint_type="fixed", step=None, parameter_count=1, alpha=[2.]),
              dict(method="per_bit_train24", candidate="per_bit_train24", training_seed=None, checkpoint_type="fixed", step=None, parameter_count=4, alpha=[2.] * 4)]
    for candidate in CANDIDATES:
        for seed in config["training"]["seeds"]:
            for kind in ("best", "last"):
                model = make_model(candidate, config).eval()
                result.append(dict(method=f"{candidate}_seed{seed}_{kind}", candidate=candidate,
                                   training_seed=seed, checkpoint_type=kind, step=2,
                                   parameter_count=sum(parameter.numel() for parameter in model.parameters()), model=model))
    return result


def context(config, channel):
    train_record = dict(split="calibration", seed=81000000, ebno_db=-10.5, status="complete", file="train.pt", sha256="train-hash")
    val_record = dict(split="calibration", seed=81000024, ebno_db=-10.5, status="complete", file="val.pt", sha256="val-hash")
    return dict(config=config, run_identity_id="new-run", run_identity={"fixture": True}, cache_dir=Path("unused-cache"),
                cache_manifest=dict(identity_id="historical-cache", channels=[train_record, val_record]),
                source_cache=dict(identity_id="historical-cache"), environment={}, code_identity={},
                split_plan=dict(train=[81000000], validation=[81000024], development=[82000000]))


def write_training(root, config, prepared):
    root.mkdir(parents=True)
    manifest = dict(schema_version=1, status="complete", run_identity_id=prepared["run_identity_id"], config=config,
                    run_identity=prepared["run_identity"], source_cache=prepared["source_cache"], split_plan=prepared["split_plan"],
                    source_cache_identity_id="historical-cache", normalization=normalization(), records=[],
                    constants=dict(global_scale=dict(alpha=[2.]), per_bit_scale=dict(alpha=[2.] * 4),
                                   fit_split="train", train_seeds=[81000000], ebno_dbs=[-10.5], settings=config["constant_fit"],
                                   fit_sources=[dict(seed=81000000, ebno_db=-10.5, file="train.pt", sha256="train-hash")]))
    for candidate in CANDIDATES:
        for seed in config["training"]["seeds"]:
            model = make_model(candidate, config)
            record = dict(candidate=candidate, training_seed=seed,
                          parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                          sampling_plan_sha256="fixture-plan", history=[dict(step=2, validation=dict(bce_nats=.2))])
            for kind in ("best", "last"):
                relative = f"{candidate}/seed_{seed}/{kind}.pth"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(schema_version=1, kind="cached_conditional_llr_scaling_v1", candidate=candidate, training_seed=seed, step=2,
                                sampling_plan_sha256="fixture-plan", parameter_count=record["parameter_count"],
                                run_identity_id="new-run", source_cache_identity_id="historical-cache",
                                normalization=normalization(), config=config, model_state=model.state_dict()), path)
                record[kind] = dict(file=relative, step=2, sha256=evaluation.file_hash(path), validation_bce_nats=.2)
            manifest["records"].append(record)
    write_manifest(root / "manifest.json", manifest)
    return manifest


class CachedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_full_codeword_raw_identity_all_fresh_models_and_rescue_counts(self):
        config, channel = configuration(), fixture()
        with patch.object(common, "feature_tensors", wraps=common.feature_tensors) as feature:
            outcomes, rows, elapsed = evaluation.evaluate_channel(channel, receivers(config), ToyCodec(), config,
                normalization(), feature, common.normalize_nodes)
        self.assertEqual(feature.call_count, 1)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(len(rows), 11)
        raw = outcomes["ep5_raw"]["decoded_payload_bits"]
        for candidate in CANDIDATES:
            for kind in ("best", "last"):
                self.assertTrue(torch.equal(outcomes[f"{candidate}_seed42_{kind}"]["decoded_payload_bits"], raw))
        self.assertEqual(outcomes["global_train24"]["rescue_vs_raw"].sum().item(), 3)
        self.assertTrue(all(row["positive_scale_hard_decisions_unchanged"] for row in rows))
        self.assertTrue(all(row["coded_bits"] == 60 and row["blocks"] == 3 for row in rows))

    def test_original_decoded_crc_and_order_mismatches_stop(self):
        config, channel = configuration(), fixture()
        channel["methods"]["ep5"]["crc_ok"] = ~channel["methods"]["ep5"]["crc_ok"]
        with self.assertRaisesRegex(ValueError, "re-decode"):
            evaluation.evaluate_channel(channel, receivers(config), ToyCodec(), config,
                                        normalization(), common.feature_tensors, common.normalize_nodes)
        channel = fixture()
        channel["methods"]["ep5"]["decoder_input_llr"] = -channel["methods"]["ep5"]["decoder_input_llr"]
        with self.assertRaisesRegex(ValueError, "ordering"):
            evaluation.evaluate_channel(channel, receivers(config), ToyCodec(), config,
                                        normalization(), common.feature_tensors, common.normalize_nodes)

    def test_three_training_seeds_never_triple_independent_channels(self):
        config = configuration("development")
        _, rows, _ = evaluation.evaluate_channel(fixture(), receivers(config), ToyCodec(), config,
                                                 normalization(), common.feature_tensors, common.normalize_nodes)
        for row in rows:
            if row["candidate"] == "graph":
                row["bce_sum_nats"] *= .5
        all_rows = []
        for seed in (82000000, 82000001):
            for ebno in (-12., -10.5, -9.):
                all_rows.extend(dict(row, seed=seed, ebno_db=ebno) for row in rows)
        summary = evaluation.summarize(all_rows, config)
        for item in summary["training_seed_summary"]:
            self.assertEqual(item["independent_channels"], 2)
            self.assertEqual(item["training_seeds"], [42, 43, 44])
            if item["ebno_db"] is None:
                self.assertEqual(item["channel_ebno_observations_per_training_seed"], 6)
        graph = [item for item in summary["paired_comparisons"] if item["category"] == "graph_vs_mlp"]
        self.assertEqual(len(graph), 24)
        self.assertTrue(all(item["bler"]["independent_channels"] == 2 for item in graph))
        for decision in summary["decision_summary"]["comparisons"]:
            if decision["comparison"] in ("graph_vs_mlp", "mlp_vs_affine", "mlp_vs_no_residual_eta"):
                self.assertEqual(decision["bler_tied_seeds"], 3)
            self.assertEqual(len(decision["per_training_seed"]), 3)
            for item in decision["per_training_seed"]:
                self.assertEqual(len(item["bler_interval"]), 2)
                self.assertEqual(len(item["bce_interval"]), 2)
                self.assertEqual(item["confidence"], config["bootstrap"]["confidence"])
                self.assertEqual(item["bce_reduced_without_bler_reduction"],
                                 item["delta_bce_nats"] < 0 and item["delta_bler"] >= 0)
                if decision["comparison"] == "graph_vs_mlp":
                    self.assertTrue(item["bce_reduced_without_bler_reduction"])
        self.assertEqual(len(summary["decision_summary"]["comparisons"]), 22)

    def test_archived_32cal_is_labeled_separate_and_pairs_exact_development_coordinates(self):
        config, channel = configuration(), fixture()
        _, rows, _ = evaluation.evaluate_channel(channel, receivers(config), ToyCodec(), config,
                                                 normalization(), common.feature_tensors, common.normalize_nodes)
        prepared = context(config, channel)
        config["mode"] = "development"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["source_archive"] = str(root)
            old_rows = [dict(rows[0], method=method, variant=variant)
                        for method in ("gt_best", "gt_last", "detr_best", "detr_last", "ep5")
                        for variant in ("raw", "global_scale", "per_bit_scale")]
            (root / "summary.json").write_text(json.dumps(dict(identity_id="historical-cache",
                metrics=[dict(method="gt_best", variant="per_bit_scale", bler=.25)])), encoding="utf-8")
            (root / "temperatures.json").write_text(json.dumps(dict(identity_id="historical-cache", alpha="not-used")), encoding="utf-8")
            (root / "channels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in old_rows), encoding="utf-8")
            report = evaluation.historical_comparisons(prepared, config, rows)
            self.assertIn("archived_32cal", report["label"])
            self.assertEqual(report["metrics"][0]["bler"], .25)
            self.assertTrue(any(item["method"] == "graph_seed42_best" and "gt_best" in item["baseline"]
                                for item in report["paired_comparisons"]))
            old_rows[0]["seed"] += 1
            (root / "channels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in old_rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not paired"):
                evaluation.historical_comparisons(prepared, config, rows)

    def test_saturation_bounds_and_bit_axis(self):
        config = configuration()
        alpha = torch.tensor([.05, 8., .5, 1.]).expand(7, 3, -1)
        summary = evaluation.alpha_statistics(alpha, config)
        self.assertEqual(summary["lower_count"], 21)
        self.assertEqual(summary["upper_count"], 21)
        torch.testing.assert_close(torch.tensor(summary["per_bit_mean"]), alpha[0, 0])
        with self.assertRaises(ValueError):
            evaluation.alpha_statistics(torch.zeros_like(alpha), config)
        with self.assertRaises(ValueError):
            evaluation.alpha_statistics(torch.full_like(alpha, .0499999), config)

    def test_training_complete_identity_checkpoint_metadata_and_path_guards(self):
        config, channel = configuration(), fixture()
        prepared = context(config, channel)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            manifest = write_training(root, config, prepared)
            _, methods, artifacts = evaluation.training_artifacts(root, prepared, config)
            self.assertEqual(len(methods), 11)
            self.assertEqual(len(artifacts), 9)
            checkpoint_record = manifest["records"][0]["best"]
            checkpoint_path = root / checkpoint_record["file"]
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            checkpoint["source_cache_identity_id"] = "wrong-historical-cache"
            torch.save(checkpoint, checkpoint_path)
            checkpoint_record["sha256"] = evaluation.file_hash(checkpoint_path)
            write_manifest(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "provenance|metadata"):
                evaluation.training_artifacts(root, prepared, config)
            manifest["records"][0]["best"]["file"] = "../escaped.pth"
            write_manifest(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "artifact|path/hash"):
                evaluation.training_artifacts(root, prepared, config, load_models=False)
            manifest["records"].pop()
            write_manifest(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "missing/duplicate|missing, duplicate"):
                evaluation.training_artifacts(root, prepared, config, load_models=False)

    def test_smoke_cli_validation_only_complete_hash_reuse_no_extra_inference(self):
        config, channel = configuration(), fixture()
        prepared = context(config, channel)
        with tempfile.TemporaryDirectory() as temporary:
            paths = {name: Path(temporary) / name for name in ("training", "results", "logs")}
            write_training(paths["training"], config, prepared)
            args = ["--run-id", "unit", "--mode", "smoke"]
            with patch.object(common, "load_config", return_value=config), \
                    patch.object(common, "prepare", return_value=prepared) as preflight, \
                    patch.object(common, "run_paths", return_value=paths), \
                    patch("evaluation.paper_soft_common.load_channel", return_value=channel) as loader, \
                    patch.object(evaluation, "codec_from_cache", return_value=ToyCodec()) as codec:
                self.assertEqual(evaluation.main(args), 0)
                summary_path = paths["results"] / "summary.json"
                summary = json.loads(summary_path.read_text())
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["split"], "validation")
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(loader.call_args.args[1]["seed"], 81000024)
                self.assertEqual(summary["archived_32cal"]["status"], "not_evaluated")
                self.assertEqual(summary["initial_alpha_one_identity"]["candidates"], {candidate: True for candidate in CANDIDATES})
                codec.reset_mock()
                self.assertEqual(evaluation.main(args + ["--check-complete"]), 0)
                self.assertFalse(preflight.call_args.kwargs["require_gpu"])
                codec.assert_not_called()
                self.assertEqual(evaluation.main(args + ["--reuse-complete"]), 0)
                incomplete = json.loads(json.dumps(summary))
                incomplete["artifacts"].pop(0)
                incomplete["artifact_digest"] = evaluation.digest({key: value for key, value in incomplete.items() if key != "artifact_digest"})
                summary_path.write_text(json.dumps(incomplete), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "artifact coverage"):
                    evaluation.main(args + ["--check-complete"])
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    evaluation.main(args)
                artifact = paths["results"] / summary["artifacts"][0]["file"]
                with artifact.open("ab") as handle:
                    handle.write(b"changed")
                with self.assertRaisesRegex(ValueError, "artifact"):
                    evaluation.main(args + ["--check-complete"])


def native_sionna_available():
    try:
        return importlib.metadata.version("sionna") == "2.0.1"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(native_sionna_available(), "Native Sionna 2.0.1 absent; synthetic toy codec tests do not qualify NR decoding")
class NativeCachedEvaluationTests(unittest.TestCase):
    def test_fresh_scales_preserve_encoded_nr_payload_crc_and_full_codeword_order(self):
        from link_level.nr_codec import NRTransportBlockCodec

        config = configuration()
        codec = NRTransportBlockCodec(64, 2, device="cpu")
        generator = torch.Generator().manual_seed(554)
        info = torch.randint(2, (1, 2, codec.info_bits), generator=generator).float()
        coded, _ = codec.encode(info)
        channel = dict(split="calibration", seed=81000024, ebno_db=-10.5,
                       data_indices=torch.arange(64), coded_bits=coded, info_bits=info)
        labels = evaluation.coded_labels(channel)
        llr = (2 * labels - 1) * 20
        native = evaluation.llrs_to_native_order(llr)
        self.assertTrue(torch.equal(native.squeeze(2) > 0, coded.bool()))
        decoded, crc = codec.decode(native)
        self.assertTrue(torch.equal(decoded, info))
        self.assertTrue(crc.all())
        outcomes, _ = evaluation.payload_statistics(info, decoded, crc)
        channel["methods"] = dict(ep5=dict(llr=llr, decoder_input_llr=native, **outcomes))
        channel["features"] = dict(ep5_mean=torch.zeros(1, 64, 2, dtype=torch.complex64),
                                   ep5_variance=torch.ones(1, 64, 2), residual=torch.ones(1, 64),
                                   eta=torch.full((1, 64, 2), .1),
                                   gram=torch.eye(2, dtype=torch.complex64).expand(1, 64, -1, -1).clone())
        qualified = evaluation.verify_initial_identity(channel, codec, config, normalization(),
                                                       common.feature_tensors, common.normalize_nodes)
        self.assertEqual(qualified["candidates"], {candidate: True for candidate in CANDIDATES})
        scaled = evaluation.scale_llr(llr, [.5, 1., 2., 4.])
        self.assertTrue(torch.equal(scaled > 0, llr > 0))
        decoded_scaled, crc_scaled = codec.decode(evaluation.llrs_to_native_order(scaled))
        self.assertTrue(torch.equal(decoded_scaled, info))
        self.assertTrue(crc_scaled.all())


if __name__ == "__main__":
    unittest.main()
