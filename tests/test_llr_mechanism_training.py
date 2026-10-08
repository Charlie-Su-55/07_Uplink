"""Temporary synthetic-only fitting, mechanism training, and artifact reuse tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import torch

from evaluation.cached_llr_common import fit_normalizer
from evaluation.llr_mechanism_common import load_config
from models.llr_mechanism_controls import CANDIDATES
from training.train_cached_llr_scaling import write_manifest
from training.train_llr_mechanism_controls import fit_noise_per_bit, main, train_candidate, validate_training


def synthetic_items(split, seeds):
    generator = torch.Generator().manual_seed(827)
    items = []
    for seed in seeds:
        for ebno, noise in ((-12.0, 4.0), (-10.5, 2.0), (-9.0, 1.0)):
            labels = torch.randint(0, 2, (4, 3, 4), generator=generator).float()
            items.append(dict(split=split, seed=seed, ebno_db=ebno, n0=noise, file=f"{split}_{seed}_{ebno}.pt",
                              sha256=f"toy-{split}-{seed}-{ebno}", nodes=torch.randn(4, 3, 6, generator=generator),
                              edges=torch.randn(4, 3, 3, 3, generator=generator), labels=labels,
                              llr=(2 * labels - 1) + torch.randn(4, 3, 4, generator=generator)))
    return items


class MechanismTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = load_config(mode="smoke")
        self.config["runtime"]["device"] = "cpu"
        self.config["training"]["re_per_step"] = 4
        self.config["training"]["validation_chunk"] = 4
        self.train = synthetic_items("train", (81000000,))
        self.validation = synthetic_items("validation", (81000024,))[1:2]
        self.context = dict(config=self.config, run_identity_id="new-fixed-algorithm", run_identity=dict(algorithm="fixed-hash"),
                            source_cache=dict(identity_id="old-cache"), cache_manifest=dict(identity_id="old-cache"),
                            split_plan=dict(train=[81000000], validation=[81000024], olddevelopment=[82000000], confirmation=[90000000]),
                            code_identity=dict(commit="training-commit", protected_algorithm_hashes={}), environment=dict(device="cpu"))
        self.context["cache_manifest"]["channels"] = [dict(split="calibration", status="complete",
            **{key: item[key] for key in ("seed", "ebno_db", "file", "sha256")}) for item in self.train + self.validation]

    def test_noise_fit_uses_all_train_points_actual_n0_and_no_other_labels(self):
        fitted = fit_noise_per_bit(self.train, self.config)
        self.assertEqual([knot["n0"] for knot in fitted["knots"]], [1.0, 2.0, 4.0])
        self.assertEqual(fitted["train_seeds"], [81000000])
        self.assertEqual({item["ebno_db"] for item in fitted["fit_sources"]}, {-12, -10.5, -9})
        for split in ("validation", "olddevelopment", "confirmation"):
            with self.assertRaises(ValueError):
                fit_noise_per_bit(synthetic_items(split, (90000000,)), self.config)
        with self.assertRaises(ValueError):
            fit_noise_per_bit(self.train[:1], self.config)
        inconsistent = copy.deepcopy(self.train)
        extra = copy.deepcopy(inconsistent[0])
        extra.update(seed=81000001, n0=8)
        inconsistent.append(extra)
        with self.assertRaises(ValueError):
            fit_noise_per_bit(inconsistent, self.config)

    def test_all_controls_share_sample_plan_and_complete_budget(self):
        normalization = fit_normalizer(self.train, self.config)
        with tempfile.TemporaryDirectory() as temporary:
            records = [train_candidate(candidate, 42, self.train, self.validation, self.config, self.context,
                                       normalization, Path(temporary) / candidate) for candidate in CANDIDATES]
            self.assertEqual(len({record["sampling_plan_sha256"] for record in records}), 1)
            self.assertTrue(all(record["last"]["step"] == 2 for record in records))
            self.assertEqual({record["parameter_count"] for record in records[:-1]}, {1412})
            self.assertEqual(records[-1]["parameter_count"], 28)
            for record in records:
                saved = torch.load(record["last"]["file"], weights_only=True, map_location="cpu")
                self.assertEqual(saved["kind"], "llr_mechanism_controls_v1")
                self.assertTrue(all(value.device.type == "cpu" for value in saved["model_state"].values()))

    def test_smoke_main_and_strict_reuse_without_loading_or_training_again(self):
        def preload(context, split):
            self.assertIs(context, self.context)
            self.assertIn(split, ("train", "validation"))
            return copy.deepcopy(self.train if split == "train" else self.validation)

        with tempfile.TemporaryDirectory() as temporary:
            paths = {name: Path(temporary) / name for name in ("training", "results", "logs")}
            with patch("evaluation.llr_mechanism_common.load_config", return_value=self.config), \
                 patch("evaluation.llr_mechanism_common.prepare", return_value=self.context), \
                 patch("evaluation.llr_mechanism_common.run_paths", return_value=paths), \
                 patch("evaluation.llr_mechanism_common.load_training_items", side_effect=preload) as loader:
                self.assertEqual(main(["--mode", "smoke", "--run-id", "toy"]), 0)
                self.assertEqual(loader.call_args_list, [call(self.context, "train"), call(self.context, "validation")])
                path = paths["training"] / "manifest.json"
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(len(manifest["records"]), 5)
                self.assertEqual(len(manifest["constants"]["noise_per_bit"]["knots"]), 3)
                self.assertEqual(len(manifest["normalization"]["fit_sources"]), 3)
                with patch("training.train_llr_mechanism_controls.train_candidate", side_effect=AssertionError("No retraining on reuse")):
                    self.assertEqual(main(["--mode", "smoke", "--run-id", "toy", "--reuse-complete"]), 0)
                self.assertEqual(loader.call_count, 2)
                next_context = copy.deepcopy(self.context)
                next_context["code_identity"]["commit"] = "documentation-only-commit"
                self.assertEqual(validate_training(paths["training"], next_context, self.config)["status"], "complete")
                bad = copy.deepcopy(manifest)
                bad["constants"]["noise_per_bit"]["knots"][0]["alpha"][0] = 1.234
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    validate_training(paths["training"], self.context, self.config)
                bad = copy.deepcopy(manifest)
                bad["records"][0]["last"]["step"] = 1
                write_manifest(path, bad)
                with self.assertRaises(ValueError):
                    validate_training(paths["training"], self.context, self.config)
                bad = copy.deepcopy(manifest)
                bad["records"] = bad["records"][:-1]
                write_manifest(path, bad)
                with self.assertRaises(ValueError):
                    validate_training(paths["training"], self.context, self.config)


if __name__ == "__main__":
    unittest.main()
