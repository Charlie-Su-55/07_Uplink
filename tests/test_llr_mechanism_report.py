"""Synthetic paired statistics/plots only; no formal confirmation results are generated."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation import report_llr_mechanism_controls as report


ROOT = Path(__file__).resolve().parents[1]


def fixture_config():
    config = json.loads((ROOT / "configs/evaluation/llr_mechanism_controls.yaml").read_text(encoding="utf-8"))
    config["mode"] = "confirmation"
    config["confirmation"].update(channels=3, ebno_dbs=[-12, -10.5, -9])
    config["bootstrap"]["replicates"] = 32
    return config


def fixture_rows(config):
    descriptors = [(method, "ep5" if method == "ep5_raw" else method, None, "raw" if index < 2 else "fixed",
                    (0, 0, 1, 4, 12)[index]) for index, method in enumerate(report.BASELINES)]
    for candidate in config["candidates"]:
        for seed in config["training"]["seeds"]:
            descriptors.append((f"{candidate}_seed{seed}_best", candidate, seed, "best", 28 if candidate == "affine_all_features" else 1412))
    for seed in (42, 43, 44):
        descriptors.append((f"frozen_reference_seed{seed}", "frozen_reference", seed, "frozen_best", 1412))
    rows = []
    for method, candidate, training_seed, kind, parameters in descriptors:
        reduction = 3 if candidate == "full_mlp" else 2 if candidate in config["candidates"] else 1 if candidate == "frozen_reference" else 0
        for seed in range(90000000, 90000003):
            for index, ebno in enumerate(config["confirmation"]["ebno_dbs"]):
                raw_errors = (8, 5, 3)[index]
                errors = raw_errors - reduction
                blocks = [position < errors for position in range(20)]
                alpha = None if method == "native_lmmse" else dict(count=160, sum=160., min=1., max=1., mean=1.,
                                                                  per_bit_mean=[1.] * 4, lower_count=0, upper_count=0,
                                                                  histogram_edges=config["scale_distribution"]["bin_edges"],
                                                                  histogram_counts=[0, 0, 0, 0, 0, 160, 0, 0, 0])
                rows.append(dict(method=method, candidate=candidate, training_seed=training_seed, checkpoint_type=kind,
                                 parameter_count=parameters, step=3000 if kind == "best" else 1000 if kind == "frozen_best" else None,
                                 seed=seed, ebno_db=ebno, n0=float(index + 1), coded_bits=160, coded_bit_errors=20,
                                 bce_sum_nats=80. - reduction, blocks=20, block_errors=errors, payload_bits=100,
                                 payload_bit_errors=errors, crc_failures=errors, undetected_block_errors=0,
                                 crc_failures_without_payload_error=0, rescue_vs_raw=reduction, harm_vs_raw=0,
                                 alpha=alpha, scale_forward_seconds=0. if parameters == 0 else .002,
                                 decode_seconds=.01, per_ue=dict(block_error=[blocks], crc_ok=[[not error for error in blocks]])))
    return rows


class MechanismReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_log_bler_interpolation_uses_only_adjacent_measured_bracket(self):
        settings = fixture_config()["targets"]
        crossing = report.target_crossing([(-12, .2), (-11, .05), (-10, .02)], .1, settings)
        self.assertEqual(crossing["status"], "resolved")
        self.assertAlmostEqual(crossing["required_ebno_db"], -11.5)
        self.assertEqual(len(crossing["brackets"]), 1)
        self.assertFalse(crossing["extrapolated"])
        exact = report.target_crossing([(-12, .2), (-11, .1), (-10, .01)], .1, settings)
        self.assertEqual(exact["required_ebno_db"], -11)
        self.assertEqual(exact["reason"], "exact_measured_target")

    def test_targets_never_extrapolate_or_choose_favorable_crossing(self):
        settings = fixture_config()["targets"]
        cases = [([(-12, .3), (-11, .2)], "target_not_bracketed_by_measured_points"),
                 ([(-12, .2), (-11, .05), (-10, .15), (-9, .03)], "nonmonotonic_measured_curve"),
                 ([(-12, .2), (-11, .1), (-10, .1), (-9, .02)], "multiple_exact_points_or_plateau"),
                 ([(-12, .2), (-11, 0.)], "zero_endpoint_cannot_interpolate_log_bler"),
                 ([(-12, .09), (-11, .1), (-10, .05)], "nonmonotonic_measured_curve")]
        for points, reason in cases:
            with self.subTest(reason=reason):
                result = report.target_crossing(points, .1, settings)
                self.assertEqual(result["status"], "unresolved")
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["required_ebno_db"])
        result = report.target_crossing(cases[1][0], .1, settings)
        self.assertEqual(len(result["brackets"]), 3)
        for points in ([(-12, .1), (-12, .05)], [(-12, -1)], [(-12, float("nan"))]):
            with self.assertRaises(ValueError):
                report.target_crossing(points, .1, settings)

    def test_paired_sign_rescue_harm_and_seed_counts_are_not_inflated(self):
        config = fixture_config()
        summary = report.summarize(fixture_rows(config), config)
        pair = next(item for item in summary["paired_comparisons"]
                    if item["method"] == "full_mlp_seed42_best" and item["baseline"] == "global_train24" and item["ebno_db"] is None)
        self.assertAlmostEqual(pair["bler"]["delta_bler"], -3 / 20)
        self.assertLess(pair["bce"]["delta_bce_nats"], 0)
        self.assertEqual((pair["rescue"], pair["harm"]), (27, 0))
        self.assertEqual(pair["bler"]["independent_channels"], 3)
        self.assertEqual(pair["bler"]["channel_ebno_observations"], 9)
        mean = next(item for item in summary["training_seed_summary"] if item["candidate"] == "full_mlp" and item["ebno_db"] is None)
        self.assertEqual(mean["training_seeds"], [42, 43, 44])
        self.assertEqual(mean["independent_channels"], 3)
        self.assertEqual(mean["channel_ebno_observations_per_training_seed"], 9)
        self.assertAlmostEqual(mean["bler"]["mean"], mean["bler"]["min"])
        self.assertEqual(mean["crc_failures"]["mean"], mean["block_errors"]["mean"])
        self.assertEqual(mean["rescue_vs_raw"]["mean"], 27)
        self.assertEqual(mean["harm_vs_raw"]["max"], 0)
        self.assertEqual(mean["alpha_mean"]["mean"], 1.)
        pooled = next(item for item in summary["metrics"] if item["method"] == "full_mlp_seed42_best" and item["ebno_db"] is None)
        self.assertEqual(sum(pooled["alpha_histogram_counts"]), 9 * 160)
        self.assertEqual(pooled["alpha_histogram_fractions"][5], 1.)
        without_eta = next(item for item in summary["metrics"] if item["candidate"] == "mlp_without_eta")
        self.assertEqual((without_eta["parameter_count"], without_eta["nominal_input_features"], without_eta["effective_input_features"]), (1412, 6, 5))
        self.assertEqual(len(summary["metrics"]), 23 * 4)
        self.assertEqual(len(summary["paired_comparisons"]), 3 * 8 * 4)

    def test_missing_case_method_last_checkpoint_and_noise_mismatch_rejected(self):
        config = fixture_config()
        original = fixture_rows(config)
        missing_method = [row for row in original if row["method"] != "mlp_without_eta_seed44_best"]
        for label, rows in (("missing_case", original[1:]), ("duplicate", original + [original[0]]),
                            ("missing_method", missing_method)):
            with self.subTest(label=label), self.assertRaises(ValueError):
                report.summarize(rows, config)
        for key, value in (("checkpoint_type", "last"), ("n0", 99.), ("parameter_count", 17), ("scale_forward_seconds", -1)):
            rows = copy.deepcopy(original)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                report.summarize(rows, config)
        incomplete_budget = dict(config, confirmation=dict(config["confirmation"], channels=4))
        with self.assertRaisesRegex(ValueError, "fixed complete"):
            report.summarize(original, incomplete_budget)

    def test_targets_report_every_training_seed_and_mean_without_best_seed_selection(self):
        config = fixture_config()
        summary = report.summarize(fixture_rows(config), config)
        curves = {item["curve"] for item in summary["targets"]}
        self.assertTrue({f"full_mlp_seed{seed}_best" for seed in (42, 43, 44)}.issubset(curves))
        self.assertIn("full_mlp_training_seed_mean", curves)
        self.assertEqual(len(summary["targets"]), (23 + 6) * 2)
        unresolved = [item for item in summary["target_gains"] if item["status"] == "unresolved"]
        self.assertTrue(unresolved)
        self.assertTrue(all(item["gain_db"] is None for item in unresolved))
        self.assertTrue(all(item["gain_db"] >= 0 for item in summary["target_gains"] if item["status"] == "resolved"))

    def test_smoke_is_old_validation_only_and_all_three_frozen_references_remain(self):
        config = fixture_config()
        config["mode"] = "smoke"
        config["training"]["seeds"] = [42]
        rows = [row for row in fixture_rows(config) if row["seed"] == 90000000 and row["ebno_db"] == -10.5]
        for row in rows:
            row["seed"] = 81000024
        summary = report.summarize(rows, config)
        self.assertEqual(len({item["method"] for item in summary["metrics"]}), 13)
        self.assertTrue(all(item["independent_channels"] == 1 for item in summary["metrics"]))
        self.assertTrue(all(item["status"] == "unresolved" or item["reason"] == "exact_measured_target" for item in summary["targets"]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Existing matplotlib is required for vector-artifact tests; no installation performed")
    def test_vector_and_csv_artifacts_have_hashes_and_refuse_overwrite(self):
        config = fixture_config()
        rows = fixture_rows(config)
        summary = report.summarize(rows, config)
        summary["shared_pipeline_timing"] = dict(case_count=9, totals_seconds=dict(transmit_seconds=.9, ce_seconds=.18, ep_seconds=.09))
        with tempfile.TemporaryDirectory(prefix="synthetic mechanism plots ") as temporary:
            artifacts = report.write_outputs(temporary, summary, rows, config)
            paths = {item["file"] for item in artifacts}
            expected = {f"plots/{name}.{extension}" for name in ("bler_curves", "feature_ablation", "parameters_added_time")
                        for extension in ("pdf", "svg")}
            self.assertTrue(expected.issubset(paths))
            self.assertIn("parameters_added_time.csv", paths)
            for artifact in artifacts:
                path = Path(temporary) / artifact["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
                self.assertEqual(path.stat().st_size, artifact["bytes"])
                self.assertGreater(artifact["bytes"], 0)
                if path.suffix == ".pdf":
                    self.assertTrue(path.read_bytes().startswith(b"%PDF"))
                if path.suffix == ".svg":
                    self.assertIn("<svg", path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                report.write_outputs(temporary, summary, rows, config)


if __name__ == "__main__":
    unittest.main()
