"""Tiny CPU fixtures exercise contracts; they are not formal cached experiments."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation import cached_llr_common as common
from evaluation.paper_soft_common import atomic_torch_save, tensor_schema
from tests.test_paper_soft_common import tiny_channel


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "40ea0098e378254246b501375033e55f91dbbbed"


def config():
    return common.load_config(ROOT / common.DEFAULT_CONFIG)


def item(seed=81000000, split="train", offset=0):
    return dict(split=split, seed=seed, ebno_db=-10.5, file=f"fixture_{seed}.pt", sha256="unit-test-only",
                nodes=torch.arange(36, dtype=torch.float32).reshape(3, 2, 6) + offset)


class CachedLLRCommonTests(unittest.TestCase):
    def test_actual_archive_hashes_and_strict_24_8_64_groups(self):
        cfg = config()
        evidence = common.read_archive(cfg)
        plan = common.split_plan(evidence["manifest"], cfg)
        self.assertEqual(plan["train"], list(range(81000000, 81000024)))
        self.assertEqual(plan["validation"], list(range(81000024, 81000032)))
        self.assertEqual(plan["development"], list(range(82000000, 82000064)))
        self.assertEqual(len(plan["files"]), 288)
        membership = {}
        for record in plan["files"]:
            membership.setdefault(record["seed"], set()).add(record["split"])
            self.assertEqual(len(record["sha256"]), 64)
        self.assertTrue(all(len(values) == 1 for values in membership.values()))
        self.assertEqual(evidence["manifest"]["identity_id"], "c1a4f87a0ead529adf0435ba7e8ab72b4d3bbf06445441a65466acfa1ddd33bf")
        for alteration in ("duplicate", "missing", "wrong_split", "overlap", "failed", "excluded"):
            bad = copy.deepcopy(evidence["manifest"])
            if alteration == "duplicate": bad["channels"].append(bad["channels"][0])
            if alteration == "missing": bad["channels"].pop()
            if alteration == "wrong_split": bad["channels"][0]["split"] = "development"
            if alteration == "overlap": bad["seed_plan"]["development"][0] = bad["seed_plan"]["calibration"][0]
            if alteration == "failed": bad["channels"][0]["status"] = "failed"
            if alteration == "excluded": bad["seed_plan"]["excluded_seeds"].append(81000000)
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                common.split_plan(bad, cfg)

    def test_config_and_output_paths_reject_protocol_changes_and_source_overlap(self):
        cfg = config()
        smoke = common.load_config(ROOT / common.DEFAULT_CONFIG, mode="smoke")
        self.assertEqual(smoke["training"]["seeds"], [42])
        self.assertEqual(smoke["training"]["steps"], 2)
        for name in ("../outside", "a/b", "a\\b", "", ".."):
            with self.assertRaises(ValueError): common.run_paths(cfg, name)
        bad = copy.deepcopy(cfg)
        bad["outputs"]["training_root"] = cfg["source_cache"]
        with self.assertRaisesRegex(ValueError, "source cache"):
            common.run_paths(bad, "test")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for section, key, value in (("protocol", "train_channels", 23), ("model", "alpha_max", 10),
                                        ("training", "seeds", [42]), ("features", "log_epsilon", 0),
                                        ("constant_fit", "alpha_min", .01), ("smoke", "training_steps", 0),
                                        ("smoke", "training_steps", 1.5), ("smoke", "training_seed", True)):
                bad = copy.deepcopy(cfg)
                bad[section][key] = value
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    common.load_config(path)

    def test_missing_cache_stops_without_generation_or_environment_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config()
            cfg["source_cache"] = str(Path(temporary) / "absent")
            with patch.object(common, "load_complete_cache") as load, patch.object(common, "numeric_runtime") as runtime:
                with self.assertRaisesRegex(FileNotFoundError, "STOP.*No channel generation"):
                    common.prepare(cfg)
                load.assert_not_called()
                runtime.assert_not_called()

    def test_source_identity_is_strict_and_separate_from_training_commit(self):
        cfg = config()
        evidence = common.read_archive(cfg)
        # Only an archived JSON manifest is copied; no fake formal channel result is created.
        with tempfile.TemporaryDirectory() as temporary:
            cfg["source_cache"] = temporary
            path = Path(temporary) / "manifest.json"
            path.write_bytes((Path(cfg["source_archive"]) / "cache_manifest.json").read_bytes().replace(b"\r\n", b"\n"))
            source_before = path.read_bytes()
            with patch.object(common, "load_complete_cache", return_value=evidence["manifest"]) as strict_loader, \
                 patch.object(common, "numeric_runtime", return_value={"test_only": True}), \
                 patch.object(common, "training_code_identity", return_value={"commit": "new-training-code"}):
                context = common.prepare(cfg, require_gpu=False)
                strict_loader.assert_called_once_with(Path(temporary))
                self.assertEqual(context["source_cache"]["identity_id"], evidence["manifest"]["identity_id"])
                self.assertEqual(context["run_identity"]["training_code_identity"]["commit"], "new-training-code")
                self.assertNotEqual(context["run_identity_id"], context["source_cache"]["identity_id"])
            self.assertEqual(path.read_bytes(), source_before)
            changed = copy.deepcopy(evidence["manifest"])
            changed["identity"]["source"]["commit"] = "new-training-code"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with patch.object(common, "load_complete_cache") as strict_loader, self.assertRaisesRegex(ValueError, "original bytes"):
                common.prepare(cfg, require_gpu=False)
            strict_loader.assert_not_called()

    def test_records_keep_all_ebnos_and_smoke_never_reads_development(self):
        cfg = config()
        manifest = common.read_archive(cfg)["manifest"]
        context = dict(config=cfg, cache_manifest=manifest, split_plan=common.split_plan(manifest, cfg))
        for split, length in (("train", 72), ("validation", 24), ("development", 192)):
            records = common.records_for(context, split)
            self.assertEqual(len(records), length)
            self.assertEqual({r["seed"] for r in records}, set(context["split_plan"][split]))
        context["config"] = common.load_config(ROOT / common.DEFAULT_CONFIG, mode="smoke")
        self.assertEqual([(r["seed"], r["ebno_db"]) for r in common.records_for(context, "validation")], [(81000024, -10.5)])
        with self.assertRaisesRegex(ValueError, "development"):
            common.records_for(context, "development")
        with patch.object(common, "load_channel") as loader, self.assertRaisesRegex(ValueError, "cannot access development"):
            common.load_prepared(context, "development")
        loader.assert_not_called()

    def test_training_preloader_reuses_strict_hash_schema_and_original_mapping(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temporary:
            payload = tiny_channel("unit-test-identity", seed=81000000)
            path = Path(temporary) / "tiny.pt"
            saved = atomic_torch_save(path, payload)
            record = dict(split="calibration", seed=81000000, ebno_db=-10.5, file=path.name, status="complete", **saved)
            context = dict(config=cfg, cache_dir=Path(temporary), cache_manifest=dict(channels=[record]),
                           source_cache=dict(identity_id="unit-test-identity"), split_plan=dict(train=[81000000], validation=[81000024], development=[82000000]))
            values = common.load_prepared(context, "train")
            self.assertEqual(len(values), 1)
            self.assertTrue(torch.equal(values[0]["labels"], common.coded_labels(payload)[0]))
            self.assertTrue(torch.equal(values[0]["llr"], payload["methods"]["ep5"]["llr"][0]))
            record["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "hash"):
                common.load_prepared(context, "train")
            record["sha256"] = saved["sha256"]
            payload["schema_version"] = 99
            torch.save(payload, path)
            record.update(sha256=common.file_sha256(path), bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "schema"):
                common.load_prepared(context, "train")

    def test_feature_allowlist_and_train_only_normalization(self):
        cfg = config()
        payload = tiny_channel("unit-test")
        nodes, edges = common.feature_tensors(payload, cfg)
        self.assertEqual(nodes.shape, (3, 2, 6))
        self.assertEqual(edges.shape, (3, 2, 2, 3))
        hostile = copy.deepcopy(payload)
        for key in ("coded_bits", "info_bits", "methods", "seed", "split"):
            hostile[key] = "unavailable oracle field"
        hostile["h_true"] = torch.full((3,), float("nan"))
        actual = common.feature_tensors(hostile, cfg)
        self.assertTrue(torch.equal(nodes, actual[0]))
        self.assertTrue(torch.equal(edges, actual[1]))
        train = [item(), item(81000001, offset=3)]
        normalizer = common.fit_normalizer(train, cfg)
        expected = torch.cat([x["nodes"].reshape(-1, 6).double() for x in train])
        self.assertTrue(torch.allclose(torch.tensor(normalizer["mean"], dtype=torch.float64), expected.mean(0)))
        self.assertTrue(torch.allclose(torch.tensor(normalizer["std"], dtype=torch.float64), expected.std(0, unbiased=False)))
        for split in ("validation", "development"):
            with self.assertRaisesRegex(ValueError, "only consume the train"):
                common.fit_normalizer([item(split=split, offset=1e9)], cfg)
        scaled = common.normalize_nodes(train[0]["nodes"], normalizer, cfg)
        self.assertTrue(torch.isfinite(scaled).all())
        self.assertLessEqual(float(scaled.abs().max()), 8)

    def test_receiver_features_are_ue_equivariant_and_logs_have_fixed_epsilon(self):
        cfg = config()
        payload = tiny_channel("unit-test")
        features = payload["features"]
        features["ep5_mean"] += torch.tensor([1j, 2+.5j])
        features["eta"] += torch.tensor([.1, .2])
        features["ep5_variance"].zero_()
        features["residual"].zero_()
        nodes, edges = common.feature_tensors(payload, cfg)
        permuted = copy.deepcopy(payload)
        for key in ("ep5_mean", "ep5_variance", "eta"):
            permuted["features"][key] = features[key][..., [1, 0]]
        permuted["features"]["gram"] = features["gram"][..., [1, 0], :][..., [1, 0]]
        new_nodes, new_edges = common.feature_tensors(permuted, cfg)
        self.assertTrue(torch.equal(new_nodes, nodes[:, [1, 0]]))
        self.assertTrue(torch.equal(new_edges, edges[:, [1, 0]][:, :, [1, 0]]))
        self.assertTrue(torch.isfinite(nodes).all())
        self.assertTrue(torch.equal(nodes[..., 2], torch.full_like(nodes[..., 2], -18)))

    def test_original_tracked_files_and_frozen_tag_unchanged(self):
        previous = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE], cwd=ROOT, text=True).splitlines())
        changed = set(subprocess.check_output(["git", "diff", "--name-only", BASELINE], cwd=ROOT, text=True).splitlines())
        self.assertFalse(previous & changed, f"Protected existing files changed: {previous & changed}")
        tag = subprocess.check_output(["git", "rev-parse", "paper-reference-classical-v1^{}"], cwd=ROOT, text=True).strip()
        self.assertEqual(tag, "f7741e53a9cad31cf62eda44b4607dfd1e3c9147")


if __name__ == "__main__":
    unittest.main()
