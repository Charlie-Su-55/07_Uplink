"""Small cached-LLR calibration checks; native NR qualification is explicitly optional."""

import copy
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation import evaluate_paper_soft_output as evaluation


SETTINGS = dict(method="bounded_derivative_bisection", precision="float64", alpha_min=.01,
                alpha_max=8., alpha_tolerance=1e-8, gradient_tolerance=1e-10, max_iterations=60)
BOOTSTRAP = dict(replicates=200, confidence=.95, seed=12345)


class ToyCodec:
    def decode(self, native):
        codeword = native.squeeze(2)
        decoded = (codeword[..., :5] > 0).float()
        decoded[..., 0] = torch.where(codeword.abs().mean(-1) < .75, 1 - decoded[..., 0], decoded[..., 0])
        return decoded, decoded.sum(-1).long().remainder(2) == 0


def fixture(split="development", seed=82000000):
    coded = torch.arange(2 * 12).reshape(1, 2, 12).remainder(2).float()
    info = coded[..., :5].clone()
    channel = dict(split=split, seed=seed, ebno_db=-10.5, data_indices=torch.tensor([1, 3, 8]),
                   native_data_ordinals=torch.arange(3), coded_bits=coded, info_bits=info, methods={})
    labels = evaluation.coded_labels(channel)
    for method, magnitude in (("gt_best", .4), ("gt_last", 1.2)):
        llr = (2 * labels - 1) * magnitude
        native = evaluation.llrs_to_native_order(llr)
        decoded, crc = ToyCodec().decode(native)
        outcomes, _ = evaluation.payload_statistics(info, decoded, crc)
        channel["methods"][method] = dict(llr=llr, decoder_input_llr=native, **outcomes)
    return channel


def temperatures(channel):
    return dict(methods={method: dict(global_scale=dict(alpha=[2.]), per_bit_scale=dict(alpha=[2., 2., 2., 2.]))
                         for method in channel["methods"]})


def record(channel, name):
    return dict(split=channel["split"], seed=channel["seed"], ebno_db=channel["ebno_db"],
                status="complete", file=name, sha256="fixture-" + name)


class CalibrationTests(unittest.TestCase):
    def test_convex_fit_known_optimum_and_mapper_positions(self):
        signed = torch.tensor([[1.] * 4] * 3 + [[-1.] * 4], dtype=torch.float64)
        factors = torch.tensor([1., 2., .5, 1.5], dtype=torch.float64)
        result = evaluation.fit_positive_scale(signed * factors, SETTINGS, per_bit=True)
        expected = torch.log(torch.tensor(3., dtype=torch.float64)) / factors
        torch.testing.assert_close(torch.tensor(result["alpha"], dtype=torch.float64), expected, atol=1e-7, rtol=0)
        self.assertFalse(any(result["boundary_hit"]))
        global_result = evaluation.fit_positive_scale(signed, SETTINGS)
        self.assertAlmostEqual(global_result["alpha"][0], float(expected[0]), places=7)

    def test_fixed_boundaries_and_invalid_configuration(self):
        positive = evaluation.fit_positive_scale(torch.ones(10, 4), SETTINGS)
        negative = evaluation.fit_positive_scale(-torch.ones(10, 4), SETTINGS)
        self.assertEqual(positive["alpha"], [SETTINGS["alpha_max"]])
        self.assertEqual(positive["stop_reason"], ["upper_boundary"])
        self.assertEqual(negative["alpha"], [SETTINGS["alpha_min"]])
        flat = evaluation.fit_positive_scale(torch.zeros(10, 4), SETTINGS)
        self.assertEqual(flat["alpha"], [SETTINGS["alpha_min"]])
        with self.assertRaises(ValueError):
            evaluation.fit_positive_scale(torch.ones(10, 4), dict(SETTINGS, alpha_min=0))
        with self.assertRaises(ValueError):
            evaluation.fit_positive_scale(torch.ones(10, 4), dict(SETTINGS, precision="float32"))

    def test_development_labels_never_loaded_and_all_calibration_points_pooled(self):
        first = fixture("calibration", 81000000)
        second = fixture("calibration", 81000000)
        second["ebno_db"] = -12.
        for method in second["methods"]:
            second["methods"][method]["llr"] *= -1
        development = fixture()
        channels = {"one": first, "two": second, "forbidden": development}
        manifest = dict(channels=[record(channel, name) for name, channel in channels.items()])
        loaded = []

        def loader(item):
            self.assertEqual(item["split"], "calibration")
            loaded.append(item["file"])
            return channels[item["file"]]

        fitted = evaluation.fit_cache_temperatures(manifest, loader, SETTINGS)
        self.assertEqual(loaded, ["one", "two"])
        self.assertEqual(fitted["pooled_ebno_dbs"], [-12., -10.5])
        self.assertEqual(fitted["independent_channel_seeds"], [81000000])
        self.assertEqual(fitted["channel_ebno_observations"], 2)
        self.assertEqual(fitted["methods"]["gt_best"]["global_scale"]["alpha"], [.01])
        with self.assertRaisesRegex(ValueError, "Development labels"):
            evaluation.fit_cache_temperatures(manifest, lambda _: development, SETTINGS)
        with self.assertRaisesRegex(ValueError, "Complete calibration"):
            evaluation.fit_cache_temperatures(dict(channels=[record(development, "dev")]), loader, SETTINGS)

    def test_full_codeword_sign_order_alpha_identity_and_positive_scales(self):
        channel = fixture()
        labels = evaluation.coded_labels(channel)
        native = evaluation.llrs_to_native_order(labels)
        self.assertTrue(torch.equal(native.squeeze(2), channel["coded_bits"]))
        llr = torch.arange(24).float().reshape(1, 3, 2, 4) - 10
        self.assertTrue(torch.equal(evaluation.scale_llr(llr, [1.]), llr))
        scaled = evaluation.scale_llr(llr, [.05, .3, 2., 8.])
        self.assertTrue(torch.equal(scaled > 0, llr > 0))
        self.assertTrue(torch.equal(scaled[..., 2], llr[..., 2] * 2))
        with self.assertRaises(ValueError):
            evaluation.scale_llr(llr, [-1.])
        with self.assertRaises(ValueError):
            evaluation.scale_llr(llr, [1., 2.])
        perfect = evaluation.soft_statistics((2 * labels - 1) * 20, labels)
        reversed_sign = evaluation.soft_statistics((1 - 2 * labels) * 20, labels)
        self.assertEqual(perfect["coded_ber"], 0)
        self.assertEqual(reversed_sign["coded_ber"], 1)

    def test_serialized_raw_decode_and_per_ue_rescue_harm(self):
        channel = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "cache.pt"
            evaluation.atomic_torch_save(target, channel)
            loaded = torch.load(target, weights_only=True)
        outcomes, rows = evaluation.evaluate_channel(loaded, temperatures(channel), ToyCodec())
        self.assertTrue(all(row["alpha_one_raw_identity_passed"] for row in rows))
        self.assertEqual(outcomes["gt_best"]["global_scale"]["rescue_vs_raw"].sum().item(), 2)
        self.assertFalse(outcomes["gt_last"]["global_scale"]["harm_vs_raw"].any())
        self.assertTrue(torch.equal(outcomes["gt_best"]["raw"]["decoded_payload_bits"],
                                    channel["methods"]["gt_best"]["decoded_payload_bits"]))
        loaded["methods"]["gt_best"]["crc_ok"] = ~loaded["methods"]["gt_best"]["crc_ok"]
        with self.assertRaisesRegex(ValueError, "cached raw re-decode"):
            evaluation.evaluate_channel(loaded, temperatures(channel), ToyCodec())

    def test_decoder_order_corruption_rejected(self):
        channel = fixture()
        channel["methods"]["gt_best"]["decoder_input_llr"] = channel["methods"]["gt_best"]["decoder_input_llr"].flip(-1)
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            evaluation.evaluate_channel(channel, temperatures(channel), ToyCodec())

    def test_bootstrap_keeps_reused_seeds_clustered_and_is_deterministic(self):
        raw = [dict(seed=seed, ebno_db=ebno, blocks=16, block_errors=3)
               for seed in (12, 13) for ebno in (-12., -10.5, -9.)]
        calibrated = [dict(row, block_errors=1 if row["seed"] == 12 else 5) for row in raw]
        first = evaluation.paired_bootstrap(calibrated, raw, BOOTSTRAP)
        self.assertEqual(first, evaluation.paired_bootstrap(calibrated, raw, BOOTSTRAP))
        self.assertEqual(first["independent_channels"], 2)
        self.assertEqual(first["channel_ebno_observations"], 6)
        self.assertEqual(first["delta_bler"], 0.)
        self.assertEqual(first["interval"], [-.125, .125])
        with self.assertRaises(ValueError):
            evaluation.paired_bootstrap(calibrated[:-1], raw, BOOTSTRAP)

    def test_best_last_summary_uses_paired_same_realizations(self):
        _, rows = evaluation.evaluate_channel(fixture(), temperatures(fixture()), ToyCodec())
        summary = evaluation.summarize(rows, BOOTSTRAP)
        self.assertEqual(len(summary["best_vs_last"]), 6)
        raw = next(item for item in summary["best_vs_last"] if item["variant"] == "raw" and item["ebno_db"] is None)
        self.assertEqual(raw["delta_bler"], -1.)
        self.assertFalse(raw["interval_informative"])

    def test_atomic_failed_write_never_publishes_completed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "decoded.pt"
            with patch.object(evaluation.torch, "save", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    evaluation.atomic_torch_save(target, {})
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_cached_fit_evaluate_reuse_and_identity_guard(self):
        from evaluation import paper_soft_common as common

        calibration, development = fixture("calibration", 81000000), fixture()
        channels = {"cal.pt": calibration, "dev.pt": development}
        manifest = dict(identity_id="test-run", identity={"fixture": True}, channels=[record(channel, name) for name, channel in channels.items()])
        configuration = dict(calibration=copy.deepcopy(SETTINGS), bootstrap=BOOTSTRAP, runtime=dict(device="cpu", torch_threads=2))
        manifest["identity"].update(experiment_config=copy.deepcopy(configuration), environment={})
        with tempfile.TemporaryDirectory() as temporary:
            paths = {name: Path(temporary) / name for name in ("cache", "calibration", "results", "logs")}
            paths["cache"].mkdir()
            (paths["cache"] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            args = ["--config", "fixture.yaml", "--run-id", "unit", "--stage", "all"]
            with patch.object(common, "load_experiment_config", return_value=configuration), \
                    patch.object(common, "run_paths", return_value=paths), \
                    patch.object(common, "load_complete_cache", return_value=manifest), \
                    patch.object(common, "load_channel", side_effect=lambda _, item, identity: channels[item["file"]]), \
                    patch.object(evaluation, "codec_from_cache", return_value=ToyCodec()), \
                    patch.object(common, "numeric_runtime", return_value={}), \
                    patch.object(common, "source_identity", return_value={}):
                self.assertEqual(evaluation.main(args), 0)
                summary = json.loads((paths["results"] / "summary.json").read_text())
                self.assertEqual(summary["status"], "complete")
                self.assertTrue(summary["all_raw_redecode_checks_passed"])
                for item in summary["artifacts"]:
                    self.assertEqual(evaluation.file_hash(paths["results"] / item["file"]), item["sha256"])
                self.assertEqual(evaluation.main(args + ["--reuse-results"]), 0)
                summary_path = paths["results"] / "summary.json"
                original_artifact_path = summary["artifacts"][0]["file"]
                summary["artifacts"][0]["file"] = "../outside.pt"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "escapes the results directory"):
                    evaluation.main(args + ["--reuse-results"])
                summary["artifacts"][0]["file"] = original_artifact_path
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    evaluation.main(args)
                manifest["channels"][0]["sha256"] = "changed-calibration-cache"
                with self.assertRaises(FileExistsError):
                    evaluation.main(args + ["--reuse-results"])
                manifest["channels"][0]["sha256"] = "fixture-cal.pt"
                manifest_path = paths["cache"] / "manifest.json"
                manifest_path.write_text(json.dumps(dict(manifest, changed_record_hash=True)), encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    evaluation.main(args + ["--reuse-results"])
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                temperature_path = paths["calibration"] / "temperatures.json"
                saved_temperatures = json.loads(temperature_path.read_text())
                saved_temperatures["methods"]["gt_best"]["global_scale"]["alpha"] = [1.234]
                temperature_path.write_text(json.dumps(saved_temperatures), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "artifact checksum"):
                    evaluation.main(args + ["--reuse-results"])
                configuration["calibration"]["alpha_max"] = 7.
                with self.assertRaisesRegex(ValueError, "Cache experiment configuration"):
                    evaluation.main(args + ["--reuse-results"])


def has_native_sionna():
    try:
        return importlib.metadata.version("sionna").split(".")[0] == "2"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(has_native_sionna(), "Native Sionna 2.x is absent; CPU toy codec is not NR acceptance")
class NativeCachedCodecTests(unittest.TestCase):
    def test_original_native_codec_cache_reload_and_alpha_one(self):
        from link_level.nr_codec import NRTransportBlockCodec

        codec = NRTransportBlockCodec(64, 2, device="cpu")
        generator = torch.Generator().manual_seed(901)
        info = torch.randint(2, (1, 2, codec.info_bits), generator=generator).float()
        coded, _ = codec.encode(info)
        llr = (coded.reshape(1, 2, 64, 4).permute(0, 2, 1, 3) * 2 - 1) * 20
        native = evaluation.llrs_to_native_order(llr)
        decoded, crc = codec.decode(native)
        self.assertTrue(torch.equal(decoded, info))
        self.assertTrue(crc.all())
        outcomes, _ = evaluation.payload_statistics(info, decoded, crc)
        channel = dict(split="development", seed=901, ebno_db=-10.5, coded_bits=coded, info_bits=info,
                       data_indices=torch.arange(64), methods={"ep5": dict(llr=llr, decoder_input_llr=native, **outcomes)})
        calibrated = dict(methods={"ep5": dict(global_scale=dict(alpha=[1.]), per_bit_scale=dict(alpha=[1.] * 4))})
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "native.pt"
            evaluation.atomic_torch_save(target, channel)
            exported, _ = evaluation.evaluate_channel(torch.load(target, weights_only=True), calibrated, codec)
        for variant in evaluation.VARIANTS:
            self.assertTrue(torch.equal(exported["ep5"][variant]["decoded_payload_bits"], info))
            self.assertTrue(exported["ep5"][variant]["crc_ok"].all())


if __name__ == "__main__":
    unittest.main()
