"""Read-only archive/provenance tests; temporary fixtures are never formal data."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation import llr_mechanism_common as common
from evaluation.paper_soft_common import atomic_torch_save
from models.conditional_llr_scaling import make_model
from tests.test_paper_soft_common import tiny_channel


ROOT = Path(__file__).resolve().parents[1]
START = "d02097f6431318942fbbe1dfc5bec77d24750308"


class MechanismCommonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = common.load_config(ROOT / common.DEFAULT_CONFIG)
        cls.archive = common.frozen_evidence(cls.config)
        cls.source = common.read_archive(cls.config)
        cls.plan = common.confirmation_plan(cls.config, cls.source["manifest"], cls.archive)

    def test_actual_seed_plan_excludes_all_history_and_retains_full_clusters(self):
        self.assertEqual(self.plan["train"], list(range(81000000, 81000024)))
        self.assertEqual(self.plan["validation"], list(range(81000024, 81000032)))
        self.assertEqual(self.plan["confirmation"], list(range(90000000, 90000512)))
        excluded = set(self.plan["confirmation_excluded_seeds"])
        for key in ("calibration", "development", "excluded_seeds", "runtime_covariance_seeds"):
            self.assertTrue(set(self.source["manifest"]["seed_plan"][key]).issubset(excluded))
        self.assertFalse(excluded & set(self.plan["confirmation"]))
        context = dict(config=self.config, split_plan=self.plan, cache_manifest=self.source["manifest"])
        records = common.records_for(context, "confirmation")
        self.assertEqual(len(records), 3584)
        for seed in self.plan["confirmation"]:
            self.assertEqual([record["ebno_db"] for record in records if record["seed"] == seed], [-12, -11.5, -11, -10.5, -10, -9.5, -9])
        altered = copy.deepcopy(self.config)
        altered["confirmation"]["start_seed"] = 81000000
        selected = common.confirmation_plan(altered, self.source["manifest"], self.archive)["confirmation"]
        self.assertFalse(set(selected) & excluded)

    def test_smoke_and_training_cannot_use_old_development_or_confirmation(self):
        cfg = common.load_config(ROOT / common.DEFAULT_CONFIG, "smoke")
        context = dict(config=cfg, split_plan=self.plan, cache_manifest=self.source["manifest"])
        self.assertEqual({record["seed"] for record in common.records_for(context, "train")}, {81000000})
        self.assertEqual(len(common.records_for(context, "train")), 3)
        self.assertEqual([(r["seed"], r["ebno_db"]) for r in common.records_for(context, "validation")], [(81000024, -10.5)])
        for group in ("development", "confirmation"):
            with self.assertRaises(ValueError): common.records_for(context, group)
            with patch.object(common, "load_channel") as load, self.assertRaises(ValueError):
                common.load_training_items(context, group)
            load.assert_not_called()

    def test_fixed_protocol_configuration_and_protected_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for section, key, value in (("training", "steps", 1000), ("model", "alpha_max", 9),
                                        ("training", "learning_rate", .01), ("noise_conditioning", "outside_range", "extrapolate"),
                                        ("confirmation", "channels", 100), ("confirmation", "fixed_budget", False)):
                bad = copy.deepcopy(self.config)
                bad[section][key] = value
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(section=section, key=key), self.assertRaises(ValueError): common.load_config(path)
        bad = copy.deepcopy(self.config)
        bad["outputs"]["training_root"] = bad["frozen_training_root"]
        with self.assertRaises(ValueError): common.run_paths(bad, "new_run")
        for name in ("../x", "a/b", "", ".."):
            with self.assertRaises(ValueError): common.run_paths(self.config, name)

    def test_receiver_configuration_is_frozen_to_the_actual_cached_frontend(self):
        source_manifest = self.source["manifest"]
        common.validate_receiver_config(self.config, source_manifest)
        changes = {"re_chunk": 64, "residual_atol": 1e-4, "residual_rtol": 1e-4, "ce_epsilon": 1.0}
        for key, value in changes.items():
            altered = copy.deepcopy(self.config)
            altered["receiver"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cached frontend"):
                common.validate_receiver_config(altered, source_manifest)
        with tempfile.TemporaryDirectory() as temporary:
            altered = copy.deepcopy(self.config)
            altered["source_cache"] = temporary
            altered["receiver"]["ce_epsilon"] = 1.0
            path = Path(temporary) / "manifest.json"
            path.write_bytes((Path(altered["source_archive"]) / "cache_manifest.json").read_bytes().replace(b"\r\n", b"\n"))
            with patch.object(common, "load_complete_cache", return_value=source_manifest), \
                 patch.object(common, "frozen_references") as references, \
                 patch.object(common, "numeric_runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "cached frontend"):
                    common.prepare(altered, require_gpu=False)
                references.assert_not_called()
                runtime.assert_not_called()
            config_path = Path(temporary) / "config.json"
            for bad_receiver in ({}, dict(self.config["receiver"], re_chunk=0),
                                 dict(self.config["receiver"], residual_atol=-1)):
                invalid = copy.deepcopy(self.config)
                invalid["receiver"] = bad_receiver
                config_path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Receiver"):
                    common.load_config(config_path)


    def test_frozen_reference_metadata_is_not_rewritten_for_new_code(self):
        manifest, record = self.archive["manifest"], self.archive["records"][0]
        checkpoint = dict(schema_version=1, kind="cached_conditional_llr_scaling_v1", candidate="mlp",
                          training_seed=record["training_seed"], step=record["best"]["step"],
                          run_identity_id=manifest["run_identity_id"], source_cache_identity_id=manifest["source_cache_identity_id"],
                          normalization=manifest["normalization"], config=manifest["config"],
                          sampling_plan_sha256=record["sampling_plan_sha256"], parameter_count=record["parameter_count"],
                          model_state=make_model("mlp", manifest["config"]).state_dict())
        self.assertEqual(sum(p.numel() for p in common.validate_frozen_checkpoint(checkpoint, record, self.archive).parameters()), 1412)
        for key, value in (("step", 3000), ("run_identity_id", "new-commit-identity"), ("source_cache_identity_id", "other-cache"),
                           ("training_seed", 7), ("normalization", {}), ("kind", "legacy")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.validate_frozen_checkpoint(dict(checkpoint, **{key: value}), record, self.archive)
        descriptors = common.frozen_references(self.config, self.archive, load_weights=False)
        self.assertEqual([d["step"] for d in descriptors], [1000] * 3)
        self.assertEqual([d["sha256"] for d in descriptors], [r["best"]["sha256"] for r in self.archive["records"]])
        bad = copy.deepcopy(self.archive)
        bad["records"][0]["best"]["file"] = "../foreign.pth"
        with self.assertRaises(ValueError): common.frozen_references(self.config, bad, load_weights=False)

    def test_documentation_commit_does_not_change_algorithm_identity_but_code_changes_fail(self):
        with patch.object(common.subprocess, "check_output", side_effect=["run-commit\n", "branch\n", ""]):
            one = common.algorithm_identity(self.archive)
        with patch.object(common.subprocess, "check_output", side_effect=["archive-only-commit\n", "branch\n", ""]):
            two = common.algorithm_identity(self.archive)
        self.assertNotEqual(one["commit"], two["commit"])
        self.assertEqual(one["protected_algorithm_hashes"], two["protected_algorithm_hashes"])
        original = common.canonical_file_hash
        def changed(path):
            return "altered" if Path(path).as_posix().endswith("link_level/nr_codec.py") else original(path)
        with patch.object(common, "canonical_file_hash", side_effect=changed), self.assertRaisesRegex(ValueError, "Protected"):
            common.algorithm_identity(self.archive)

    def test_prepare_keeps_original_data_identity_and_stable_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = copy.deepcopy(self.config)
            cfg["source_cache"] = temporary
            path = Path(temporary) / "manifest.json"
            path.write_bytes((Path(cfg["source_archive"]) / "cache_manifest.json").read_bytes().replace(b"\r\n", b"\n"))
            before = path.read_bytes()
            prior = self.source["manifest"]["identity"]["reference"]["covariance"]
            original_hash = common.file_sha256
            def file_hash(name):
                return prior["sha256"] if str(name) == prior["path"] else original_hash(name)
            references = common.frozen_references(cfg, self.archive, load_weights=False)
            codes = [dict(commit=commit, protected_algorithm_hashes={"algorithm.py": "same"}) for commit in ("run", "docs-only")]
            with patch.object(common, "load_complete_cache", return_value=self.source["manifest"]) as loader, \
                 patch.object(common, "frozen_references", return_value=references), \
                 patch.object(common, "numeric_runtime", return_value={"test_only": True}), \
                 patch.object(common, "algorithm_identity", side_effect=codes), patch.object(common, "file_sha256", side_effect=file_hash):
                one, two = common.prepare(cfg, require_gpu=False), common.prepare(cfg, require_gpu=False)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(one["run_identity_id"], two["run_identity_id"])
            self.assertNotEqual(one["code_identity"]["commit"], two["code_identity"]["commit"])
            self.assertEqual(one["source_cache"]["identity_id"], self.source["manifest"]["identity_id"])
            self.assertEqual(before, path.read_bytes())
            path.write_text('{}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source cache"):
                common.prepare(cfg, require_gpu=False)

    def test_missing_cache_stops_without_prior_checkpoint_or_environment_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(self.config, source_cache=str(Path(temporary) / "missing"))
            with patch.object(common, "frozen_references") as refs, patch.object(common, "numeric_runtime") as runtime:
                with self.assertRaisesRegex(FileNotFoundError, "STOP"):
                    common.prepare(cfg)
                refs.assert_not_called()
                runtime.assert_not_called()

    def test_training_items_use_cached_actual_n0_and_strict_hash_loader(self):
        cfg = common.load_config(ROOT / common.DEFAULT_CONFIG, "smoke")
        with tempfile.TemporaryDirectory() as temporary:
            payload = tiny_channel("unit-test", seed=81000000)
            path = Path(temporary) / "tiny.pt"
            saved = atomic_torch_save(path, payload)
            record = dict(split="calibration", seed=81000000, ebno_db=-10.5, status="complete", file=path.name, **saved)
            ctx = dict(config=cfg, split_plan=self.plan, cache_dir=Path(temporary), cache_manifest=dict(channels=[record]), source_cache=dict(identity_id="unit-test"))
            items = common.load_training_items(ctx, "train")
            self.assertEqual(items[0]["n0"], .5)
            self.assertEqual(items[0]["nodes"].shape, (3, 2, 6))
            record["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "hash"): common.load_training_items(ctx, "train")

    def test_all_original_tracked_files_and_frozen_tag_remain_unchanged(self):
        previous = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", START], cwd=ROOT, text=True).splitlines())
        changed = set(subprocess.check_output(["git", "diff", "--name-only", START], cwd=ROOT, text=True).splitlines())
        self.assertFalse(previous & changed)
        self.assertEqual(subprocess.check_output(["git", "rev-parse", "paper-reference-classical-v1^{}"], cwd=ROOT, text=True).strip(),
                         "f7741e53a9cad31cf62eda44b4607dfd1e3c9147")


if __name__ == "__main__":
    unittest.main()
