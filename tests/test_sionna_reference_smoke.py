"""No UMa generation: accounting and synthetic <=4Rx reference smoke checks."""

import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from evaluation.evaluate_sionna_bler import (
    add_statistics, assess_acceptance, block_statistics, output_path, save_json,
)
from link_level.sionna_reference import (
    SionnaReferenceLink, compare_classical_lmmse, power_diagnostics, validate_reference_config,
)


def tiny_config():
    return {
        "general": dict(seed=42, device="cpu", precision="single", num_ues=2, streams_per_ue=1),
        "ofdm": dict(num_subcarriers=48, fft_size=48, subcarrier_spacing_hz=30000.,
                     num_ofdm_symbols=4, cyclic_prefix_length=4, omitted_symbols=[],
                     dmrs_symbols=[], data_symbols=[0, 1, 2, 3]),
        "bs_array": dict(num_rows_per_panel=1, num_cols_per_panel=2, polarization="dual"),
        "ut_array": dict(num_rows_per_panel=2, num_cols_per_panel=1, polarization="dual"),
        "topology": {"scenario": "uma"}, "channel": {"normalize_channel": True},
        "stream_mapping": {"mode": "equal_gain"}, "power_control": {"mode": "none"},
        "interference": {"enabled": False}, "channel_estimation": {"mode": "perfect"},
        "link": dict(noise_mode="sionna_ebno", energy_convention="unit_energy_per_ue",
                     coderate_convention="payload_bits_over_coded_bits"),
        "nr": dict(mcs_table=1, mcs_index=10, num_bp_iter=20),
    }


def has_sionna2():
    try:
        return importlib.metadata.version("sionna").split(".")[0] == "2"
    except importlib.metadata.PackageNotFoundError:
        return False


class ReferenceGuardsTests(unittest.TestCase):
    def test_grid_and_legacy_guards(self):
        self.assertEqual(validate_reference_config(tiny_config())["num_data_symbols"], 192)
        for section, field, value in (
            ("channel", "normalize_channel", False), ("power_control", "mode", "fractional"),
            ("interference", "enabled", True), ("channel_estimation", "mode", "lmmse"),
            ("general", "streams_per_ue", 16), ("link", "rx_snr_db", 10),
            ("ofdm", "dmrs_symbols", [1]), ("ofdm", "data_symbols", [2, 3]),
            ("ofdm", "num_guard_carriers", [1, 1]), ("ofdm", "dc_null", True)):
            cfg = tiny_config()
            cfg[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                validate_reference_config(cfg)

    def test_large_cpu_simulation_rejected_without_sionna_import(self):
        cfg = tiny_config()
        cfg["bs_array"].update(num_rows_per_panel=8, num_cols_per_panel=16)
        with self.assertRaisesRegex(ValueError, "GPU server"):
            SionnaReferenceLink(cfg)

    def test_atomic_output_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_json(path, {"status": "running", "points": [1]})
            with self.assertRaises(FileExistsError):
                output_path(SimpleNamespace(output=str(path), overwrite=False))
            save_json(path, {"status": "complete", "points": [1, 2]})
            self.assertEqual(json.loads(path.read_text())["points"], [1, 2])
            with self.assertRaises(ValueError):
                save_json(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text())["status"], "complete")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_acceptance_insufficient_support_floor_and_paired_increase(self):
        def point(ebno, errors):
            return dict(status="complete", ebno_db=ebno, channels=len(errors),
                        bler=sum(errors) / (2 * len(errors)), per_channel_block_errors=errors)

        low, high = point(-10, [2] * 1000), point(10, [0] * 1000)
        passed = assess_acceptance([low, high], 2)
        self.assertEqual(passed["status"], "pass")
        self.assertEqual(passed["measured_crossing_brackets"], [[-10, 10]])
        smoke = assess_acceptance([point(-10, [2]), point(10, [0])], 2)
        self.assertEqual(smoke["status"], "unresolved")
        floor = assess_acceptance([low, point(10, [1] * 1000)], 2)
        self.assertEqual(floor["status"], "unresolved")
        rising = assess_acceptance([low, point(0, [0] * 1000), point(5, [1] * 1000), high], 2)
        self.assertEqual(rising["significant_paired_increases"], [[0, 5]])
        self.assertEqual(rising["status"], "unresolved")
        # With 100 zero-error channels, the upper bound still exceeds 0.01.
        self.assertEqual(assess_acceptance([point(-10, [2] * 100), point(10, [0] * 100)], 2)["status"], "unresolved")


@unittest.skipIf(torch is None, "Requires small CPU PyTorch tensors")
class TensorContractTests(unittest.TestCase):
    def test_cli_pairing_partial_batches_and_failure_checkpoint(self):
        from evaluation import evaluate_sionna_bler as evaluator

        class FakeLink:
            num_ues, num_rx, precision = 2, 4, "single"
            codec = SimpleNamespace(mcs=SimpleNamespace(table=1, index=10))

            def __init__(self):
                self.calls = []
                self.fail_at = None

            def metadata(self):
                return {"test_only": True}

            def noise_variance(self, ebno):
                return torch.tensor(10 ** (-ebno / 10))

            def transmit(self, size, ebno, seed):
                if len(self.calls) == self.fail_at:
                    raise RuntimeError("injected channel failure")
                self.calls.append((size, ebno, seed))
                return dict(info=torch.zeros(size, 2, 5), ebno=ebno)

            def check_codec_identity(self, sample):
                return {"status": "pass"}

            def receive(self, sample):
                info = sample["info"]
                return dict(decoded=info if sample["ebno"] > 0 else 1 - info,
                            crc_ok=torch.full(info.shape[:2], sample["ebno"] > 0))

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(tiny_config()), encoding="utf-8")
            result = Path(directory) / "result.json"
            args = ["--config", str(config), "--output", str(result), "--channels", "3",
                    "--batch-size", "2", "--seed", "91", "--ebno-dbs=-5,5", "--require-acceptance"]
            link = FakeLink()
            with (patch.object(evaluator, "SionnaReferenceLink", return_value=link),
                  patch.object(evaluator, "provenance", return_value={"test_only": True}),
                  patch.object(evaluator, "power_diagnostics", return_value={"reconstruction_relative_rms": 0.}),
                  patch("sys.stdout", new_callable=io.StringIO)):
                self.assertEqual(evaluator.main(args), 2)  # smoke is unresolved
                report = json.loads(result.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "complete")
                self.assertEqual(link.calls, [(2, -5., 91), (1, -5., 93), (2, 5., 91), (1, 5., 93)])
                self.assertEqual([p["blocks"] for p in report["points"]], [6, 6])
                self.assertEqual([p["bler"] for p in report["points"]], [1., 0.])
                self.assertFalse(report["acceptance"]["first_acceptance_case"])
                link.calls = []
                link.fail_at = 3
                with self.assertRaisesRegex(RuntimeError, "injected channel failure"):
                    evaluator.main(args + ["--overwrite"])
                partial = json.loads(result.read_text(encoding="utf-8"))
                self.assertEqual(partial["status"], "failed")
                self.assertEqual(partial["points"][0]["status"], "complete")
                self.assertEqual(partial["points"][1]["channels"], 2)
                self.assertEqual(partial["points"][1]["status"], "running")

    def test_projection_matches_physical_antenna_transmission(self):
        from data.link.stream_mapping import RankOneStreamMapper
        torch.manual_seed(27)
        raw = torch.randn(2, 1, 4, 3, 4, 2, 5, dtype=torch.complex64)
        x = torch.randn(2, 3, 1, 2, 5, dtype=torch.complex64)
        mapper = RankOneStreamMapper(4, device="cpu")
        h_eff = mapper(raw)
        x_ant = mapper.map_symbols(x[:, :, 0].permute(0, 2, 3, 1))
        clean = torch.einsum("brmkatf,bkatf->brmtf", raw, x_ant)
        sample = dict(x=x, x_ant=x_ant, h_raw=raw, h_eff=h_eff,
                      y_clean=clean, y=clean + 0.1 * torch.randn_like(clean), no=torch.tensor(.01),
                      seed=27, ebno_db=0.)
        diagnostic = power_diagnostics(sample)
        self.assertLess(diagnostic["reconstruction_relative_rms"], 2e-7)
        torch.testing.assert_close(x.abs().square().sum(2), x_ant.abs().square().sum(2))
        self.assertAlmostEqual(mapper.precoder.abs().square().sum().item(), 1)

    def test_bler_counts_are_payload_based_and_batch_safe(self):
        info = torch.zeros(2, 3, 5)
        decoded = info.clone()
        decoded[0, 1, 2] = decoded[1, 2, 4] = 1
        crc = torch.tensor([[True, True, True], [False, True, False]])
        counts = block_statistics({"info": info}, {"decoded": decoded, "crc_ok": crc})
        self.assertEqual(counts["block_errors"], 2)
        self.assertEqual(counts["per_ue_block_errors"], [0, 1, 1])
        self.assertEqual(counts["per_channel_block_errors"], [1, 1])
        self.assertEqual(counts["undetected_block_errors"], 1)
        self.assertEqual(counts["crc_failures_without_payload_error"], 1)
        point = {}
        add_statistics(point, counts)
        add_statistics(point, counts)
        self.assertEqual(point["channels"], 4)
        self.assertAlmostEqual(point["bler"], 1 / 3)


class SyntheticChannel:
    """Unit raw-link power with distinct UE/antenna/time/frequency phases."""
    scale = 1.0

    def sample(self, batch_size):
        m = torch.arange(4).reshape(1, 1, 4, 1, 1, 1, 1)
        k = torch.arange(2).reshape(1, 1, 1, 2, 1, 1, 1)
        a = torch.arange(4).reshape(1, 1, 1, 1, 4, 1, 1)
        t = torch.arange(4).reshape(1, 1, 1, 1, 1, 4, 1)
        f = torch.arange(48).reshape(1, 1, 1, 1, 1, 1, 48)
        b = torch.arange(batch_size).reshape(batch_size, 1, 1, 1, 1, 1, 1)
        phase = 2 * torch.pi * m * k / 4 + .03 * (k + 1) * f + .2 * t + .1 * b + .04 * a
        return self.scale * torch.exp(1j * phase), {}


@unittest.skipUnless(torch is not None and has_sionna2(), "Requires native Sionna 2.x reference")
class NativeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_coded_end_to_end_batch_two_and_no_noise_recalibration(self):
        provider = SyntheticChannel()
        link = SionnaReferenceLink(tiny_config(), channel_provider=provider)
        sample = link.transmit(2, 12., 43)
        self.assertEqual(link.check_codec_identity(sample)["status"], "pass")
        result = link.receive(sample)
        self.assertTrue(torch.equal(result["decoded"], sample["info"]))
        self.assertTrue(result["crc_ok"].all().item())
        self.assertEqual(tuple(sample["h"].shape), (2, 1, 4, 2, 1, 4, 48))
        self.assertEqual(tuple(result["llr"].shape), (2, 2, 1, 768))
        diagnostic = power_diagnostics(sample)
        self.assertLess(diagnostic["reconstruction_relative_rms"], 1e-6)
        for ratio in diagnostic["measured_noise_over_n0"]:
            self.assertLess(abs(ratio - 1), .15)
        original_noise = sample["y"] - sample["y_clean"]
        provider.scale = 3.
        scaled = link.transmit(2, 12., 43)
        self.assertEqual(sample["no"].item(), scaled["no"].item())
        torch.testing.assert_close(scaled["y_clean"], 3 * sample["y_clean"])
        torch.testing.assert_close(scaled["y"] - scaled["y_clean"], original_noise, atol=3e-6, rtol=1e-4)
        torch.testing.assert_close(scaled["coded"], sample["coded"])

    def test_lmmse_core_agrees_with_native_and_batch_noise_is_paired(self):
        link = SionnaReferenceLink(tiny_config(), channel_provider=SyntheticChannel())
        sample = link.transmit(2, -5., 42)
        received = link.receive(sample)
        comparison = compare_classical_lmmse(link, sample, received)
        for key, error in comparison.items():
            if key.endswith("relative_rms"):
                self.assertLess(error, 1e-4, key)
        higher = link.transmit(2, 0., 42)
        torch.testing.assert_close(higher["h_raw"], sample["h_raw"])
        torch.testing.assert_close(higher["x"], sample["x"])
        torch.testing.assert_close(
            (higher["y"] - higher["y_clean"]) / higher["no"].sqrt(),
            (sample["y"] - sample["y_clean"]) / sample["no"].sqrt(), atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
