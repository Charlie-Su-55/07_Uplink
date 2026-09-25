"""Mode B contracts and small synthetic CPU checks; never generate UMa here."""

import copy
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

from link_level.detector_adapter import (
    ClassicalDetectorAdapter, compare_lmmse_outputs, llrs_to_native_order,
    make_detector_batch, native_channel_to_grid, validate_detector_request, whiten_diagonal_uncertainty,
)
from link_level.sionna_ce import (
    COVARIANCE_KIND, channel_second_moments, csi_diagnostics, distribution_id,
    regularize_covariance, validate_covariance_artifact,
)
from link_level.sionna_reference import SionnaReferenceLink, validate_reference_config


def has_sionna2():
    try:
        return importlib.metadata.version("sionna").split(".")[0] == "2"
    except importlib.metadata.PackageNotFoundError:
        return False


def paper_fixture():
    return dict(
        mode="paper_reference",
        general=dict(seed=42, device="cpu", precision="single", num_ues=2,
                     streams_per_ue=1, carrier_frequency_hz=6.7e9),
        ofdm=dict(num_subcarriers=32, fft_size=32, subcarrier_spacing_hz=30000.,
                  num_ofdm_symbols=4, cyclic_prefix_length=4, omitted_symbols=[],
                  dmrs_symbols=[1, 3], data_symbols=[0, 2], pilot_pattern="kronecker", pilot_seed=314159),
        bs_array=dict(num_rows_per_panel=1, num_cols_per_panel=2, polarization="dual"),
        ut_array=dict(num_rows_per_panel=2, num_cols_per_panel=1, polarization="dual"),
        topology={"scenario": "uma"}, channel={"normalize_channel": True},
        stream_mapping={"mode": "equal_gain"}, power_control={"mode": "none"}, interference={"enabled": False},
        channel_estimation=dict(mode="perfect", interpolation_order="f-t", rx_chunk_size=2, covariance_path="unused.pt"),
        link=dict(noise_mode="sionna_ebno", energy_convention="unit_energy_per_ue",
                  coderate_convention="payload_bits_over_coded_bits"),
        nr=dict(mcs_table=1, mcs_index=10, num_bp_iter=20),
        paper_contract=dict(num_data_symbols=64, coded_bits_per_ue=256, info_bits_per_ue=80))


class PaperContractTests(unittest.TestCase):
    def test_csi_switch_cannot_change_grid_code_or_distribution(self):
        perfect = paper_fixture()
        practical = copy.deepcopy(perfect)
        practical["channel_estimation"]["mode"] = "practical"
        self.assertEqual(validate_reference_config(perfect), validate_reference_config(practical))
        practical["general"].update(seed=73, device="cuda:0")
        self.assertEqual(distribution_id(perfect), distribution_id(practical))
        practical["ofdm"]["pilot_seed"] += 1
        self.assertNotEqual(distribution_id(perfect), distribution_id(practical))

    def test_frozen_contract_and_pilot_guards(self):
        for section, field, value in (
            ("nr", "mcs_index", 11), ("ofdm", "dmrs_symbols", []),
            ("ofdm", "pilot_pattern", "empty"), ("ofdm", "data_symbols", [0, 1, 2, 3]),
            ("paper_contract", "coded_bits_per_ue", 999), ("ofdm", "omitted_symbols", [0])):
            cfg = paper_fixture()
            cfg[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reference_config(cfg)

    def test_neural_and_silent_ce_drop_are_rejected(self):
        for detector in ("gt", "gt_ep", "detr", "flow"):
            with self.assertRaises(ValueError):
                validate_detector_request(("sionna_lmmse", detector), "perfect", "require_explicit")
        with self.assertRaisesRegex(ValueError, "err_var cannot be dropped"):
            validate_detector_request(("custom_lmmse", "ep5"), "practical", "require_explicit")
        validate_detector_request(("custom_lmmse", "ep5"), "practical", "sionna_diagonal")


@unittest.skipIf(torch is None, "Requires small synthetic CPU PyTorch tensors")
class PaperTensorTests(unittest.TestCase):
    def test_covariance_complex_orientation_and_absolute_power(self):
        time = torch.arange(3).reshape(1, 3, 1, 1, 1)
        freq = torch.arange(4).reshape(1, 1, 4, 1, 1)
        h = (2 * torch.exp(1j * (.3 * time + .7 * freq))).expand(2, 3, 4, 2, 2)
        rf, rt = channel_second_moments(h)
        torch.testing.assert_close(rf[0, 1], torch.tensor(4.) * torch.exp(torch.tensor(-.7j)))
        torch.testing.assert_close(rt[0, 1], torch.tensor(4.) * torch.exp(torch.tensor(-.3j)))
        for matrix in (rf, rt):
            regularized = regularize_covariance(matrix)
            self.assertAlmostEqual(regularized.diagonal().real.mean().item(), 4, places=5)
            self.assertGreater(torch.linalg.eigvalsh(regularized).min().item(), 0)

    def test_covariance_cache_rejects_legacy_or_wrong_distribution(self):
        cfg = paper_fixture()
        artifact = dict(kind=COVARIANCE_KIND, distribution_id=distribution_id(cfg),
                        calibration_seeds=[4000000], cov_mat_freq=torch.eye(32, dtype=torch.complex64),
                        cov_mat_time=torch.eye(4, dtype=torch.complex64))
        validate_covariance_artifact(artifact, cfg)
        for key, value in (("kind", "legacy"), ("distribution_id", "wrong"), ("calibration_seeds", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_covariance_artifact(dict(artifact, **{key: value}), cfg)

    def test_explicit_ce_whitening_matches_effective_covariance(self):
        torch.manual_seed(32)
        y = torch.randn(2, 1, 3, 4, dtype=torch.complex64)
        h = torch.randn(2, 1, 3, 4, 2, dtype=torch.complex64)
        error = torch.rand(h.shape) * .2
        no = torch.tensor(.3)
        yw, hw = whiten_diagonal_uncertainty(y, h, error, no)
        inv = torch.diag_embed((no + error.sum(-1)).reciprocal()).to(h.dtype)
        torch.testing.assert_close(hw.mH @ hw, h.mH @ inv @ h)
        torch.testing.assert_close(hw.mH @ yw.unsqueeze(-1), h.mH @ inv @ y.unsqueeze(-1))
        perfect_y, perfect_h = whiten_diagonal_uncertainty(y, h, torch.zeros_like(error), no)
        torch.testing.assert_close(perfect_y, y / no.sqrt())
        torch.testing.assert_close(perfect_h, h / no.sqrt())

    def test_codeword_and_native_channel_order(self):
        tags = torch.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4)
        native = llrs_to_native_order(tags)
        for b in range(2):
            for k in range(3):
                torch.testing.assert_close(native[b, k, 0], tags[b, :, k].flatten())
        h = torch.arange(2 * 4 * 3 * 2 * 5).reshape(2, 1, 4, 3, 1, 2, 5)
        grid = native_channel_to_grid(h)
        self.assertEqual(grid[1, 1, 4, 2, 1].item(), h[1, 0, 2, 1, 0, 1, 4].item())

    def test_practical_estimates_cached_and_err_var_reaches_equalizer(self):
        link = SionnaReferenceLink.__new__(SionnaReferenceLink)
        link.csi, link.device, link.dtype, link.num_rx = "practical", "cpu", torch.float32, 4
        link.cfg = paper_fixture()
        link.covariance_metadata = {"calibration_seeds": [4000000]}
        calls = []

        def estimator(y, no):
            calls.append(y.shape[2])
            shape = (2, 1, y.shape[2], 2, 1, 4, 32)
            return torch.full(shape, 7 + 1j), torch.full(shape, .25)

        captured = {}

        def equalizer(y, h_hat, err_var, no):
            captured.update(h_hat=h_hat, err_var=err_var, no=no)
            return torch.ones(2, 2, 1, 64, dtype=torch.complex64), torch.ones(2, 2, 1, 64)

        link.channel_estimator, link.equalizer = estimator, equalizer
        link.demapper = lambda x, no: torch.zeros(2, 2, 1, 256)
        link.codec = SimpleNamespace(decode=lambda llr: (torch.zeros(2, 2, 80), torch.ones(2, 2, dtype=torch.bool)))
        sample = dict(y=torch.ones(2, 1, 4, 4, 32, dtype=torch.complex64),
                      h=torch.full((2, 1, 4, 2, 1, 4, 32), 99 + 0j), no=torch.tensor(.1), seed=42)
        result = link.receive(sample)
        self.assertEqual(calls, [2, 2])
        self.assertTrue(torch.all(captured["h_hat"] == 7 + 1j))
        self.assertTrue(torch.all(captured["err_var"] == .25))
        self.assertIs(captured["err_var"], result["err_var"])
        link.receive(sample)
        self.assertEqual(calls, [2, 2])  # no re-estimation for the next detector
        with self.assertRaisesRegex(ValueError, "overlaps"):
            link.estimate_channel(dict(sample, seed=4000000, csi_estimates={}), "practical")

    def test_one_transmit_serves_both_csi_modes_and_all_detectors(self):
        from evaluation import evaluate_paper_reference as evaluator

        class FakeLink:
            num_ues, precision = 2, "single"
            covariance_metadata = {"calibration_seeds": [4000000]}

            def __init__(self):
                self.calls = []
                self.identities = []
                self.fail_at = None

            def metadata(self):
                return {"test_only": True}

            def noise_variance(self, ebno):
                return torch.tensor(1.)

            def transmit(self, size, ebno, seed):
                if len(self.calls) == self.fail_at:
                    raise RuntimeError("injected failure")
                self.calls.append((size, ebno, seed))
                return dict(info=torch.zeros(size, 2, 3), seed=seed)

            def check_codec_identity(self, sample):
                return {"status": "pass"}

        link = FakeLink()

        class FakeAdapter:
            def __init__(self, *args):
                pass

            def evaluate(self, sample, batch):
                link.identities.append(id(sample))
                return {name: dict(decoded=sample["info"], crc_ok=torch.ones(sample["info"].shape[:2], dtype=torch.bool))
                        for name in ("sionna_lmmse", "custom_lmmse", "ep5")}

        with tempfile.TemporaryDirectory() as directory:
            config, output = Path(directory) / "config.json", Path(directory) / "out.json"
            config.write_text(json.dumps(paper_fixture()), encoding="utf-8")
            args = ["--config", str(config), "--output", str(output), "--csi", "practical",
                    "--channels", "3", "--batch-size", "2", "--ebno-dbs=-20,-18",
                    "--detectors", "sionna_lmmse,custom_lmmse,ep5", "--custom-ce-policy", "sionna_diagonal"]
            with (patch.object(evaluator, "SionnaReferenceLink", return_value=link),
                  patch.object(evaluator, "ClassicalDetectorAdapter", FakeAdapter),
                  patch.object(evaluator, "make_detector_batch", side_effect=lambda link, sample, csi: {}),
                  patch.object(evaluator, "csi_diagnostics", return_value={"mean_err_var": .1}),
                  patch.object(evaluator, "compare_lmmse_outputs", return_value={}),
                  patch.object(evaluator, "power_diagnostics", return_value={"reconstruction_relative_rms": 0.}),
                  patch.object(evaluator, "provenance", return_value={}), patch("sys.stdout", new_callable=io.StringIO)):
                evaluator.main(args)
                self.assertEqual(link.calls, [(2, -20., 42), (1, -20., 44), (2, -18., 42), (1, -18., 44)])
                for a, b in zip(link.identities[::2], link.identities[1::2]):
                    self.assertEqual(a, b)
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "complete")
                self.assertEqual(len(report["points"][0]["receivers"]), 6)
                self.assertEqual(report["points"][1]["receivers"]["practical/ep5"]["blocks"], 6)
                link.calls, link.fail_at = [], 3
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    evaluator.main(args + ["--overwrite"])
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["points"][0]["status"], "complete")
                self.assertEqual(report["points"][1]["receivers"]["practical/ep5"]["blocks"], 4)


class FlatChannel:
    calls = 0

    def sample(self, batch_size):
        self.calls += 1
        m = torch.arange(4).reshape(1, 1, 4, 1, 1, 1, 1)
        k = torch.arange(2).reshape(1, 1, 1, 2, 1, 1, 1)
        b = torch.arange(batch_size).reshape(batch_size, 1, 1, 1, 1, 1, 1)
        h = torch.exp(1j * (2 * torch.pi * m * k / 4 + .2 * b))
        return h.expand(batch_size, 1, 4, 2, 4, 4, 32).contiguous(), {}


@unittest.skipUnless(torch is not None and has_sionna2(), "Requires native Sionna 2.x pilot/CE pipeline")
class NativePaperTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.cfg = paper_fixture()
        path = Path(self.directory.name) / "cov.pt"
        artifact = dict(kind=COVARIANCE_KIND, distribution_id=distribution_id(self.cfg), calibration_seeds=[4000000],
                        cov_mat_freq=regularize_covariance(4 * torch.ones(32, 32, dtype=torch.complex64)),
                        cov_mat_time=regularize_covariance(4 * torch.ones(4, 4, dtype=torch.complex64)))
        torch.save(artifact, path)
        self.cfg["channel_estimation"]["covariance_path"] = str(path)

    def tearDown(self):
        self.directory.cleanup()
        torch.set_num_threads(self.old_threads)

    def test_production_grid_and_codec_contract_without_uma_or_256rx(self):
        from sionna.phy import config
        from sionna.phy.ofdm import ResourceGrid
        from link_level.nr_codec import NRTransportBlockCodec
        config.seed = 314159
        grid = ResourceGrid(14, 192, 30000., num_tx=16, num_streams_per_tx=1,
                            cyclic_prefix_length=14, pilot_pattern="kronecker",
                            pilot_ofdm_symbol_indices=[2, 11], device="cpu")
        codec = NRTransportBlockCodec(int(grid.num_data_symbols), 16, table=1, index=10, device="cpu")
        self.assertEqual((int(grid.num_data_symbols), codec.coded_bits, codec.info_bits), (2304, 9216, 3104))
        pilots = grid.pilot_pattern.pilots
        self.assertTrue(torch.all((pilots.abs() > 0).sum(-1) == 24))
        self.assertTrue(torch.allclose(pilots.abs().square().sum(-1), torch.full((16, 1), 384.), atol=1e-4))

    def test_actual_pilots_grid_tb_and_ebno_identical_between_csi_modes(self):
        perfect = SionnaReferenceLink(self.cfg, channel_provider=FlatChannel())
        practical_cfg = copy.deepcopy(self.cfg)
        practical_cfg["channel_estimation"]["mode"] = "practical"
        practical_cfg["general"]["seed"] = 900  # cannot change pilot sequence
        practical = SionnaReferenceLink(practical_cfg, channel_provider=FlatChannel())
        a, b = perfect.transmit(2, 12., 42), practical.transmit(2, 12., 42)
        for key in ("info", "coded", "x", "h_raw", "y", "no"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(perfect.grid_metadata, practical.grid_metadata)
        self.assertEqual(perfect.codec.metadata(), practical.codec.metadata())
        self.assertEqual((perfect.codec.info_bits, perfect.codec.coded_bits), (80, 256))
        pilot_grid = a["x"][:, :, 0, self.cfg["ofdm"]["dmrs_symbols"]]
        self.assertTrue(torch.all((pilot_grid.abs() > 0).sum(1) == 1))
        self.assertAlmostEqual(pilot_grid.abs().square().mean().item(), 1., places=5)
        expected_no = (4 / 2) * (1 + 4 / 32) / (10 ** 1.2 * 4 * 80 / 256)
        self.assertAlmostEqual(a["no"].item(), expected_no, places=6)

    def test_native_ls_lmmse_uncertainty_and_classical_same_realization(self):
        self.cfg["channel_estimation"]["mode"] = "practical"
        provider = FlatChannel()
        link = SionnaReferenceLink(self.cfg, channel_provider=provider)
        sample = link.transmit(2, 12., 42)
        adapter = ClassicalDetectorAdapter(link, custom_ce_policy="sionna_diagonal", re_chunk=13)
        for csi in ("perfect", "practical"):
            batch = make_detector_batch(link, sample, csi)
            ruu_before = batch["Ruu"].clone()
            outputs = adapter.evaluate(sample, batch)
            self.assertEqual(provider.calls, 1)
            torch.testing.assert_close(batch["Ruu"], ruu_before)
            torch.testing.assert_close(batch["Ruu"], sample["no"] * torch.eye(4, dtype=torch.complex64).expand(2, -1, -1))
            for value in compare_lmmse_outputs(outputs).values():
                self.assertLess(value, 2e-3)
            for name in outputs:
                self.assertTrue(torch.isfinite(outputs[name]["llr"]).all())
                self.assertTrue(torch.equal(outputs[name]["decoded"], sample["info"]), name)
            diagnostics = csi_diagnostics(batch)
            if csi == "practical":
                self.assertGreater(diagnostics["mean_err_var"], 0)
                self.assertLess(diagnostics["data_h_nmse"], .02)
            else:
                self.assertEqual(diagnostics["mean_err_var"], 0)
                self.assertEqual(diagnostics["h_nmse"], 0)


if __name__ == "__main__":
    unittest.main()
