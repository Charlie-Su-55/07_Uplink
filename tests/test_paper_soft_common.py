"""Small CPU provenance/cache contracts; no server artifacts or external A1 access."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation import paper_soft_common as common
from link_level.detector_adapter import llrs_to_native_order


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "experiment_snapshots" / "mode_b_v1"


def tiny_channel(identity_id, split="calibration", seed=100):
    info = torch.zeros((1, 2, 5), dtype=torch.float32)
    llr = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) - 11.5
    native = llrs_to_native_order(llr)
    decoded = info.clone()
    decoded[0, 0, 0] = 1
    errors = (decoded != info).sum(-1)
    methods = {name: dict(llr=llr.clone(), decoder_input_llr=native.clone(),
                         decoded_payload_bits=decoded.clone(), crc_ok=torch.tensor([[False, True]]),
                         payload_bit_errors=errors.clone(), block_error=errors > 0,
                         total_payload_bits_per_ue=torch.full_like(errors, 5)) for name in common.METHODS}
    payload = dict(schema_version=common.SCHEMA_VERSION, status="complete", identity_id=identity_id,
                   split=split, ebno_db=-10.5, seed=seed, n0=torch.tensor(0.5, dtype=torch.float32),
                   data_indices=torch.tensor([0, 2, 4]), native_data_ordinals=torch.arange(3),
                   coded_bits=(native.squeeze(2) > 0).float(), info_bits=info, methods=methods,
                   features=dict(z=torch.ones((1, 3, 2), dtype=torch.complex64),
                                 gram=torch.eye(2, dtype=torch.complex64).expand(1, 3, 2, 2).clone(),
                                 ep5_mean=torch.zeros((1, 3, 2), dtype=torch.complex64),
                                 ep5_variance=torch.ones((1, 3, 2)), received_energy=torch.ones((1, 3)),
                                 residual=torch.ones((1, 3)), eta=torch.full((1, 3, 2), 0.1)))
    shapes = common.tensor_schema(payload)
    payload["metadata"] = dict(tensor_shapes=shapes,
                               tensor_axes={name: [f"axis{index}" for index in range(len(item["shape"]))]
                                            for name, item in shapes.items()})
    return payload


def tiny_identity():
    config = common.load_experiment_config(ROOT / common.DEFAULT_CONFIG, mode="smoke")
    plan = dict(calibration=[100], development=[200], excluded_seeds=[1, 2, 3, 4],
                exclusion_sources=[dict(seeds=dict(train_seeds=[1], validation_seeds=[2],
                                                  calibration_seeds=[3], evaluation_seeds=[4]))],
                runtime_covariance_seeds=[5], runtime_checkpoints=[dict(train_seeds=[6], validation_seeds=[7])])
    return dict(experiment_config=config, seed_plan=plan,
                reference=dict(grid=dict(num_ues=2, num_data_symbols=3, num_ofdm_symbols=5,
                                         fft_size=1, pilot_symbols=[1, 3]),
                               codec=dict(bits_per_symbol=4, coded_bits_per_ue=12, info_bits_per_ue=5)))


class PaperSoftProvenanceTests(unittest.TestCase):
    def test_config_smoke_defaults_and_invalid_settings(self):
        config = common.load_experiment_config(ROOT / common.DEFAULT_CONFIG)
        smoke = common.load_experiment_config(ROOT / common.DEFAULT_CONFIG, mode="smoke")
        self.assertEqual(config["ebno_dbs"], [-12.0, -10.5, -9.0])
        self.assertEqual([config["splits"][name]["channels"] for name in ("calibration", "development")], [32, 64])
        self.assertEqual(smoke["ebno_dbs"], [-10.5])
        self.assertEqual([specification["channels"] for specification in smoke["splits"].values()], [1, 1])
        self.assertEqual(config["re_chunk"], 128)
        self.assertEqual(config["runtime"]["expected_sionna"], "2.0.1")
        self.assertEqual(config["runtime"]["expected_torch"], "2.11.0")
        self.assertNotIn("lr2e5", json.dumps(config["checkpoints"]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            changes = (("batch_size", 2), ("re_chunk", 0), ("ebno_dbs", [-10.5, -10.5]),
                       ("ebno_dbs", [float("nan")]), ("schema_version", 999))
            for key, value in changes:
                with self.subTest(key=key, value=value):
                    invalid = copy.deepcopy(config)
                    invalid[key] = value
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        common.load_experiment_config(path)
        for run_id in ("", ".", "..", "../outside", "nested/run", "nested\\run", "-flag"):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                common.run_paths(config, run_id)
        self.assertEqual(common.run_paths(config, "dev_20261008.v1")["cache"],
                         Path("data/cache/paper_soft_output_dev/dev_20261008.v1"))

    def test_archive_exclusions_contain_complete_history_and_all_categories(self):
        evidence = common.archive_evidence(SNAPSHOT)
        excluded = set(evidence["excluded_seeds"])
        for path in (SNAPSHOT / "training_history").glob("*.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
            self.assertGreaterEqual(len(metadata["train_seeds"]), 300, path.name)
            self.assertTrue(set(metadata["train_seeds"]).issubset(excluded))
            self.assertTrue(set(metadata["validation_seeds"]).issubset(excluded))
        for category in ("train_seeds", "validation_seeds", "calibration_seeds", "evaluation_seeds"):
            seeds = {seed for source in evidence["sources"] for seed in source["seeds"][category]}
            self.assertTrue(seeds, category)
            self.assertTrue(seeds.issubset(excluded))
        config = common.load_experiment_config(ROOT / common.DEFAULT_CONFIG)
        plan = common.seed_plan(config, evidence)
        self.assertEqual(len(plan["calibration"]), 32)
        self.assertEqual(len(plan["development"]), 64)
        self.assertFalse(set(plan["calibration"]) & set(plan["development"]))
        self.assertFalse(excluded & set(plan["calibration"] + plan["development"]))
        self.assertEqual(len(common.expected_cases(dict(experiment_config=config, seed_plan=plan))), 288)

    def test_missing_indexed_archive_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = dict(sources=[dict(archived="eval/missing.json", archived_sha256="missing")])
            (root / "snapshot_index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Missing indexed archived evidence"):
                common.archive_evidence(root)

    def test_seed_selection_skips_every_source_and_other_split(self):
        config = common.load_experiment_config(ROOT / common.DEFAULT_CONFIG, mode="smoke")
        for specification in config["splits"].values():
            specification.update(start_seed=100, channels=2)
        evidence = dict(excluded_seeds=[100, 102], sources=[dict(path="archive.json")])
        runtime_checkpoints = [dict(path="synthetic.pth", train_seeds=[104], validation_seeds=[105])]
        plan = common.seed_plan(config, evidence, covariance_seeds=[101], checkpoint_metadata=runtime_checkpoints)
        self.assertEqual(plan["calibration"], [103, 106])
        self.assertEqual(plan["development"], [107, 108])
        self.assertEqual(plan["excluded_seeds"], [100, 101, 102, 104, 105])
        self.assertEqual(plan["runtime_checkpoints"], runtime_checkpoints)
        config["splits"]["calibration"].update(start_seed=2 ** 31 - 1)
        with self.assertRaisesRegex(ValueError, "Exhausted"):
            common.seed_plan(config, evidence)

    def test_checkpoint_metadata_rejects_legacy_wrong_prior_and_seed_overlap(self):
        checkpoint = copy.deepcopy(json.loads((SNAPSHOT / "training_history" /
                                                "paper_reference_gt_ep_smoke300.json").read_text(encoding="utf-8"))["metadata"])
        checkpoint["step"] = 7
        cfg = checkpoint["system_config"]
        prior_hash = checkpoint["ce_covariance_sha256"]
        common.validate_neural_checkpoint(checkpoint, "gt_ep", cfg, prior_hash, [81000000])
        mutations = {"kind": "legacy", "arch": "detr_ep", "bits_per_symbol": 6, "mcs_index": 11,
                     "csi": "perfect", "custom_ce_policy": "ignore", "ce_covariance_sha256": "wrong",
                     "llr_convention": "log(P0/P1)", "distribution_id": "wrong", "step": -1,
                     "train_seeds": [], "validation_seeds": []}
        for key, value in mutations.items():
            invalid = dict(checkpoint, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.validate_neural_checkpoint(invalid, "gt_ep", cfg, prior_hash, [81000000])
        for key in ("train_seeds", "validation_seeds"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "overlap"):
                common.validate_neural_checkpoint(checkpoint, "gt_ep", cfg, prior_hash, [checkpoint[key][-1]])

    def test_frozen_sources_and_historical_archives_unchanged(self):
        protected = ["link_level", "training", "models", "detectors", "configs/system",
                     "docs/experiment_snapshots/mode_b_v1", "evaluation/compare_paper_reference_a1.py",
                     "evaluation/evaluate_paper_reference.py", "evaluation/evaluate_sionna_bler.py",
                     "evaluation/export_paper_reference_replay.py"]
        changed = subprocess.check_output(["git", "diff", "--name-only", "eaf735bf30974cf05371a9ca49386012a863cb3f",
                                           "--", *protected], cwd=ROOT, text=True)
        self.assertEqual(changed.strip(), "")
        tag = subprocess.check_output(["git", "rev-parse", "paper-reference-classical-v1^{commit}"], cwd=ROOT, text=True)
        self.assertEqual(tag.strip(), "f7741e53a9cad31cf62eda44b4607dfd1e3c9147")
        sources = common.source_identity()["python_sources_lf_sha256"]
        self.assertIn("evaluation/paper_soft_common.py", sources)
        self.assertIn("link_level/nr_codec.py", sources)


class PaperSoftCacheIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        identity = tiny_identity()
        identity_id = common.digest(identity)
        records = []
        for split, ebno, seed in common.expected_cases(identity):
            payload = tiny_channel(identity_id, split, seed)
            filename = common.channel_name(split, ebno, seed)
            storage = common.atomic_torch_save(self.root / filename, payload)
            records.append(dict(file=filename, status="complete", split=split, ebno_db=ebno, seed=seed, **storage))
        self.manifest = dict(schema_version=common.SCHEMA_VERSION, status="complete", identity=identity,
                             identity_id=identity_id, seed_plan=copy.deepcopy(identity["seed_plan"]), channels=records)
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_full_cpu_cache_roundtrip_retains_all_methods_order_and_counts(self):
        manifest = common.load_complete_cache(self.root)
        for record in manifest["channels"]:
            payload = common.load_channel(self.root, record, manifest["identity_id"])
            self.assertEqual(set(payload["methods"]), set(common.METHODS))
            for method in payload["methods"].values():
                self.assertTrue(torch.equal(llrs_to_native_order(method["llr"]), method["decoder_input_llr"]))
                self.assertTrue(torch.equal(method["payload_bit_errors"], torch.tensor([[1, 0]])))
                self.assertEqual(method["llr"].device.type, "cpu")
                self.assertEqual(method["llr"].dtype, torch.float32)

    def test_incomplete_manifest_duplicate_missing_and_identity_rejected(self):
        valid = copy.deepcopy(self.manifest)
        mutations = (("status", "running"), ("schema_version", 999), ("identity_id", "wrong"),
                     ("channels", valid["channels"][:1]),
                     ("channels", valid["channels"] + valid["channels"][:1]))
        for key, value in mutations:
            with self.subTest(key=key):
                self.manifest = dict(valid, **{key: value})
                self.write_manifest()
                with self.assertRaises(ValueError):
                    common.load_complete_cache(self.root)

    def test_record_hash_size_path_status_and_coordinates_rejected(self):
        record = self.manifest["channels"][0]
        mutations = dict(sha256="0" * 64, bytes=record["bytes"] + 1, file="../outside.pt",
                         status="running", seed=record["seed"] + 1)
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.load_channel(self.root, dict(record, **{key: value}), self.manifest["identity_id"])
        path = self.root / record["file"]
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "hash"):
            common.load_channel(self.root, record, self.manifest["identity_id"])

    def test_channel_shapes_dtypes_bits_sign_order_and_axes_rejected(self):
        payload = tiny_channel("identity")
        changes = [lambda item: item.update(native_data_ordinals=torch.tensor([0, 2, 1])),
                   lambda item: item.update(data_indices=torch.tensor([0, 0, 4])),
                   lambda item: item.update(n0=torch.tensor([0.5])),
                   lambda item: item["methods"]["ep5"].update(decoder_input_llr=-item["methods"]["ep5"]["decoder_input_llr"]),
                   lambda item: item["methods"]["ep5"].update(decoder_input_llr=item["methods"]["ep5"]["decoder_input_llr"].double()),
                   lambda item: item["methods"]["ep5"].update(decoded_payload_bits=torch.full((1, 2, 5), 0.5)),
                   lambda item: item["methods"]["ep5"].update(block_error=torch.tensor([[1, 0]])),
                   lambda item: item["features"].update(gram=torch.zeros((1, 3, 2, 2))),
                   lambda item: item["metadata"]["tensor_axes"].pop("n0"),
                   lambda item: item["metadata"]["tensor_shapes"].pop("n0")]
        for index, change in enumerate(changes):
            invalid = copy.deepcopy(payload)
            change(invalid)
            with self.subTest(index=index), self.assertRaises(ValueError):
                common.validate_channel(invalid, "identity")

    def test_seed_plan_mismatch_and_historical_leakage_rejected(self):
        valid = copy.deepcopy(self.manifest)
        self.manifest["seed_plan"]["calibration"] = [999]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "seed plan"):
            common.load_complete_cache(self.root)
        for key in ("excluded_seeds", "runtime_covariance_seeds", "runtime_checkpoints", "exclusion_sources"):
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(valid)
                plan = self.manifest["identity"]["seed_plan"]
                if key in ("excluded_seeds", "runtime_covariance_seeds"):
                    plan[key].append(100)
                elif key == "runtime_checkpoints":
                    plan[key].append(dict(train_seeds=[100], validation_seeds=[999]))
                else:
                    plan[key].append(dict(seeds=dict(evaluation_seeds=[100])))
                self.manifest["seed_plan"] = copy.deepcopy(plan)
                self.manifest["identity_id"] = common.digest(self.manifest["identity"])
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "historical"):
                    common.load_complete_cache(self.root)

    def test_cross_split_leakage_and_duplicate_seed_identity_rejected(self):
        self.manifest["identity"]["seed_plan"]["development"] = [100]
        self.manifest["seed_plan"] = copy.deepcopy(self.manifest["identity"]["seed_plan"])
        self.manifest["identity_id"] = common.digest(self.manifest["identity"])
        self.manifest["channels"][1]["seed"] = 100
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Calibration/development seed leakage"):
            common.load_complete_cache(self.root)

    def test_grid_and_codec_identity_must_match_cached_tensors(self):
        for target, key, value in (("grid", "num_ues", 3), ("codec", "info_bits_per_ue", 6),
                                   ("grid", "pilot_symbols", [0, 1])):
            with self.subTest(target=target, key=key), patch.object(common, "load_channel", return_value=tiny_channel("ignored")):
                invalid = copy.deepcopy(self.manifest)
                invalid["identity"]["reference"][target][key] = value
                invalid["identity_id"] = common.digest(invalid["identity"])
                self.manifest, previous = invalid, self.manifest
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "grid|codec"):
                    common.load_complete_cache(self.root)
                self.manifest = previous

    def test_atomic_verification_failure_leaves_no_complete_or_temporary_file(self):
        target = self.root / "new" / "channel.pt"
        def reject(path):
            self.assertTrue(path.exists())
            self.assertFalse(target.exists())
            raise ValueError("verification failed")
        with self.assertRaisesRegex(ValueError, "verification failed"):
            common.atomic_torch_save(target, tiny_channel("identity"), verify=reject)
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.iterdir()), [])
        common.atomic_torch_save(target, tiny_channel("identity"))
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            common.atomic_torch_save(target, {})
        self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
