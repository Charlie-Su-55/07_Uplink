"""Synthetic temporary-only training checks with no source cache or GPU access."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import torch

from models.conditional_llr_scaling import ConditionalScale
from training.train_cached_llr_scaling import (
    fit_constants, make_sampling_plan, reuse_manifest, train_candidate, validate, validate_fit_sources, write_manifest,
)


def synthetic_items(split, seeds=(81000000, 81000001)):
    generator = torch.Generator().manual_seed(927)
    items = []
    for seed in seeds:
        for ebno in (-12.0, -10.5, -9.0):
            labels = torch.randint(0, 2, (8, 3, 4), generator=generator).float()
            llr = (2 * labels - 1) + torch.randn(8, 3, 4, generator=generator)
            items.append(dict(split=split, seed=seed, ebno_db=ebno, file=f"{split}_{seed}_{ebno}.pt",
                              sha256=f"synthetic-{split}-{seed}-{ebno}", nodes=torch.randn(8, 3, 6, generator=generator),
                              edges=torch.randn(8, 3, 3, 3, generator=generator), llr=llr, labels=labels))
    return items


def training_config():
    return dict(mode="smoke", model=dict(hidden=32, alpha_min=0.05, alpha_max=8.0, layers=2),
                candidates=["affine"], runtime=dict(device="cpu"), smoke=dict(training_steps=2, training_seed=42),
                training=dict(steps=2, re_per_step=4, seeds=[42], learning_rate=0.001,
                              weight_decay=0.0001, grad_clip=1.0, validate_every=1, validation_chunk=4),
                constant_fit=dict(method="bounded_derivative_bisection", precision="float64", alpha_min=0.05,
                                  alpha_max=8.0, alpha_tolerance=1e-5, gradient_tolerance=1e-7, max_iterations=40))


class CachedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.train = synthetic_items("train")
        self.validation = synthetic_items("validation", (81000024,))
        self.context = dict(run_identity_id="new-run-identity", cache_manifest=dict(identity_id="old-cache-identity"),
                            run_identity=dict(code="new-code"), source_cache=dict(identity_id="old-cache-identity"),
                            split_plan=dict(train=[81000000, 81000001], validation=[81000024], development=[82000000]))
        self.normalization = dict(mean=[0.0] * 6, std=[1.0] * 6, train_seeds=[81000000, 81000001], fit_split="train",
                                  fit_sources=[{key: item[key] for key in ("file", "sha256")} for item in self.train])

    def test_sampling_plan_deterministic_and_private_from_model_rng(self):
        first, first_hash = make_sampling_plan(self.train, 42, 12, 4)
        torch.manual_seed(999)
        ConditionalScale("graph")
        second, second_hash = make_sampling_plan(self.train, 42, 12, 4)
        self.assertEqual(first_hash, second_hash)
        for one, two in zip(first, second):
            self.assertEqual(one["item_index"], two["item_index"])
            self.assertTrue(torch.equal(one["re_indices"], two["re_indices"]))
            self.assertEqual(len(set(one["re_indices"].tolist())), 4)
        self.assertNotEqual(first_hash, make_sampling_plan(self.train, 43, 12, 4)[1])

    def test_validation_or_development_labels_cannot_fit_or_sample(self):
        settings = dict(method="bounded_derivative_bisection", precision="float64", alpha_min=0.05,
                        alpha_max=8.0, alpha_tolerance=1e-5, gradient_tolerance=1e-7, max_iterations=40)
        fitted = fit_constants(self.train, settings)
        self.assertEqual(fitted["train_seeds"], [81000000, 81000001])
        self.assertEqual(len(fitted["per_bit_scale"]["alpha"]), 4)
        self.assertEqual(fitted["ebno_dbs"], [-12, -10.5, -9])
        for split in ("validation", "development"):
            forbidden = synthetic_items(split)
            with self.assertRaises(ValueError):
                fit_constants(forbidden, settings)
            with self.assertRaises(ValueError):
                make_sampling_plan(forbidden, 42, 2, 4)
        with self.assertRaises(ValueError):
            validate(ConditionalScale("affine"), synthetic_items("development"), "cpu", 4)

    def test_training_records_best_and_last_cpu_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = train_candidate("affine", 42, self.train, self.validation, training_config(),
                                     self.context, self.normalization, Path(temporary) / "affine" / "seed_42")
            self.assertEqual(record["last"]["step"], 2)
            self.assertEqual(record["best"]["validation_bce_nats"], min(row["validation"]["bce_nats"] for row in record["history"]))
            self.assertEqual([row["step"] for row in record["history"]], [0, 1, 2])
            payload = torch.load(record["last"]["file"], map_location="cpu", weights_only=True)
            self.assertTrue(all(value.device.type == "cpu" for value in payload["model_state"].values()))
            self.assertEqual(payload["run_identity_id"], "new-run-identity")
            self.assertEqual(payload["source_cache_identity_id"], "old-cache-identity")
            self.assertEqual(payload["sampling_plan_sha256"], record["sampling_plan_sha256"])
            self.assertGreater(record["parameter_count"], 0)

    def test_same_seed_training_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = [train_candidate("mlp", 42, self.train, self.validation, training_config(),
                                       self.context, self.normalization, Path(temporary) / str(index)) for index in range(2)]
            states = [torch.load(record["last"]["file"], weights_only=True)["model_state"] for record in records]
            self.assertEqual(records[0]["sampling_plan_sha256"], records[1]["sampling_plan_sha256"])
            self.assertTrue(all(torch.equal(value, states[1][name]) for name, value in states[0].items()))

    def test_complete_reuse_requires_hashes_and_identities(self):
        config = training_config()
        with tempfile.TemporaryDirectory() as temporary, patch("evaluation.cached_llr_common.records_for", return_value=self.train):
            directory = Path(temporary)
            record = train_candidate("affine", 42, self.train, self.validation, config,
                                     self.context, self.normalization, directory / "affine" / "seed_42")
            manifest = dict(schema_version=1, status="complete", run_identity_id=self.context["run_identity_id"],
                            run_identity=self.context["run_identity"], source_cache=self.context["source_cache"],
                            source_cache_identity_id=self.context["cache_manifest"]["identity_id"],
                            split_plan=self.context["split_plan"], config=config,
                            constants=fit_constants(self.train, config["constant_fit"]),
                            normalization=self.normalization, records=[record])
            path = directory / "manifest.json"
            write_manifest(path, manifest)
            self.assertEqual(reuse_manifest(directory, self.context, config)["status"], "complete")
            wrong_context = dict(self.context, run_identity_id="different-new-code")
            with self.assertRaises(ValueError):
                reuse_manifest(directory, wrong_context, config)
            modified = copy.deepcopy(manifest)
            modified["records"][0]["last"]["sha256"] = "wrong"
            write_manifest(path, modified)
            with self.assertRaises(ValueError):
                reuse_manifest(directory, self.context, config)
            tampered = copy.deepcopy(manifest)
            tampered["constants"]["global_scale"]["alpha"] = [1.2]
            path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                reuse_manifest(directory, self.context, config)
            manifest["status"] = "failed"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                reuse_manifest(directory, self.context, config)

    def test_fits_reject_foreign_sources_and_validation_checks_identity(self):
        constants = fit_constants(self.train, training_config()["constant_fit"])
        validate_fit_sources(self.train, self.normalization, constants)
        foreign = copy.deepcopy(self.normalization)
        foreign["fit_sources"][0]["sha256"] = "development-source"
        with self.assertRaises(ValueError):
            validate_fit_sources(self.train, foreign, constants)
        for candidate in ("affine", "mlp", "graph", "mlp_no_residual_eta"):
            model = ConditionalScale(candidate)
            metrics = validate(model, self.validation, "cpu", 4, require_identity=True)
            self.assertEqual(metrics["alpha_mean"], 1)
            self.assertEqual(metrics["alpha_std"], 0)
            self.assertEqual(metrics["bce_nats"], metrics["raw_bce_nats"])
            self.assertEqual(metrics["near_lower_boundary_fraction"], 0)
            self.assertGreaterEqual(metrics["coded_ber"], 0)
            with torch.no_grad():
                model.head.bias.fill_(0.1)
            with self.assertRaises(ValueError):
                validate(model, self.validation, "cpu", 4, require_identity=True)


    def test_smoke_main_all_candidates_then_complete_reuse(self):
        from evaluation.cached_llr_common import load_config
        from training.train_cached_llr_scaling import main

        config = load_config(mode="smoke")
        config["runtime"]["device"] = "cpu"
        config["training"]["re_per_step"] = 4
        config["training"]["validation_chunk"] = 4
        train_items = [copy.deepcopy(self.train[1])]
        validation_items = [copy.deepcopy(self.validation[1])]
        for item in train_items + validation_items:
            for key in ("nodes", "edges", "llr", "labels"):
                item[key] = item[key][:4].clone()
        context = copy.deepcopy(self.context)
        context["config"] = config
        context["cache_manifest"]["channels"] = [dict(split="calibration", status="complete",
            **{key: item[key] for key in ("seed", "ebno_db", "file", "sha256")})
            for item in train_items + validation_items]

        def prepared(actual_context, split):
            self.assertIs(actual_context, context)
            self.assertIn(split, ("train", "validation"))
            return copy.deepcopy(train_items if split == "train" else validation_items)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / name for name in ("training", "results", "logs")}
            with patch("evaluation.cached_llr_common.load_config", return_value=config), \
                 patch("evaluation.cached_llr_common.prepare", return_value=context), \
                 patch("evaluation.cached_llr_common.run_paths", return_value=paths), \
                 patch("evaluation.cached_llr_common.load_prepared", side_effect=prepared) as loader:
                self.assertEqual(main(["--mode", "smoke", "--run-id", "synthetic_smoke"]), 0)
                self.assertEqual(loader.call_args_list, [call(context, "train"), call(context, "validation")])
                manifest = json.loads((paths["training"] / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual({record["candidate"] for record in manifest["records"]}, set(config["candidates"]))
                self.assertEqual(len(manifest["records"]), 4)
                self.assertEqual(manifest["normalization"]["train_seeds"], [train_items[0]["seed"]])
                self.assertEqual(manifest["constants"]["train_seeds"], [train_items[0]["seed"]])
                self.assertEqual(manifest["normalization"]["fit_sources"],
                                 [{key: train_items[0][key] for key in ("file", "sha256")}])
                for record in manifest["records"]:
                    self.assertEqual(record["last"]["step"], 2)
                    for selection in ("best", "last"):
                        self.assertTrue((paths["training"] / record[selection]["file"]).is_file())
                with patch("training.train_cached_llr_scaling.train_candidate",
                           side_effect=AssertionError("Complete reuse must never train")):
                    self.assertEqual(main(["--mode", "smoke", "--run-id", "synthetic_smoke", "--reuse-complete"]), 0)
                self.assertEqual(loader.call_count, 2)


    def test_train_validation_overlap_rejected(self):
        overlapping = synthetic_items("validation")
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            train_candidate("affine", 42, self.train, overlapping, training_config(),
                            self.context, self.normalization, temporary)


if __name__ == "__main__":
    unittest.main()
