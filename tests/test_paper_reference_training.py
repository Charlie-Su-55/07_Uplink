"""Small CPU checks for training plumbing; no UMa generation or GPU experiment."""

import copy
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from training import train_paper_reference_detector as trainer
from link_level.sionna_ce import distribution_id


ROOT = Path(__file__).resolve().parents[1]


def paper_config():
    # Only a configuration fixture: never instantiate a 256Rx channel locally.
    return dict(mode="paper_reference",
                general=dict(seed=17, device="cuda:0", precision="single", num_ues=16,
                             streams_per_ue=1, carrier_frequency_hz=6.7e9),
                ofdm=dict(num_subcarriers=192, fft_size=192, subcarrier_spacing_hz=30000.,
                          num_ofdm_symbols=14, cyclic_prefix_length=14, omitted_symbols=[],
                          dmrs_symbols=[2, 11], data_symbols=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
                          pilot_pattern="kronecker", pilot_seed=314159),
                bs_array=dict(num_rows_per_panel=8, num_cols_per_panel=16, polarization="dual"),
                ut_array=dict(num_rows_per_panel=2, num_cols_per_panel=1, polarization="dual"),
                topology={"scenario": "uma"}, channel={"normalize_channel": True},
                stream_mapping={"mode": "equal_gain"}, power_control={"mode": "none"},
                interference={"enabled": False},
                channel_estimation=dict(mode="perfect", covariance_path="old.pt", interpolation_order="f-t", rx_chunk_size=4),
                link=dict(noise_mode="sionna_ebno", energy_convention="unit_energy_per_ue",
                          coderate_convention="payload_bits_over_coded_bits"),
                nr=dict(mcs_table=1, mcs_index=10, num_bp_iter=20),
                paper_contract=dict(num_data_symbols=2304, coded_bits_per_ue=9216, info_bits_per_ue=3104))


def arguments(*extra):
    return trainer.parse_args(["--arch", "gt_ep", "--ce-covariance", "prior.pt", "--output-dir", "unused", *extra])


def tensor_batch():
    generator = torch.Generator().manual_seed(71)
    return dict(y=torch.randn(2, 2, 3, 4, dtype=torch.complex64, generator=generator),
                h_hat=torch.randn(2, 2, 3, 4, 2, dtype=torch.complex64, generator=generator),
                err_var=torch.rand(2, 2, 3, 4, 2, generator=generator) * .2,
                n0=torch.tensor(.3), data_indices=torch.tensor([0, 2, 5]),
                coded_bits=torch.randint(0, 2, (2, 2, 12), generator=generator).float(),
                metadata={"csi": "practical"})


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_defaults_and_no_legacy_or_checkpoint_loading_options(self):
        args = arguments()
        self.assertEqual((args.ebno_min_db, args.ebno_max_db, args.val_ebno_db), (-13., -9., -10.5))
        for extra in (("--rx-snr-db", "5"), ("--csi", "perfect"), ("--resume", "legacy.pth"),
                      ("--checkpoint", "legacy.pth"), ("--config", "legacy.yaml"),
                      ("--ebno-min-db", "nan"), ("--ebno-min-db", "0"), ("--steps", "0")):
            with self.subTest(extra=extra), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                arguments(*extra)
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            trainer.parse_args(["--arch", "gt_ep", "--output-dir", "new"])

    def test_runtime_config_preserves_frozen_classical_contract(self):
        cfg = paper_config()
        before = copy.deepcopy(cfg)
        with patch.object(trainer, "load_config", return_value=cfg) as load:
            actual = trainer.training_config(arguments())
        load.assert_called_once_with(trainer.PAPER_CONFIG)
        self.assertEqual(cfg, before)
        self.assertEqual(actual["channel_estimation"]["mode"], "practical")
        self.assertEqual(distribution_id(actual), distribution_id(before))
        for section in ("ofdm", "nr", "paper_contract", "link", "bs_array", "ut_array", "channel", "stream_mapping"):
            self.assertEqual(actual[section], before[section])
        cfg["link"]["rx_snr_db"] = 5.
        with patch.object(trainer, "load_config", return_value=cfg), self.assertRaisesRegex(ValueError, "legacy"):
            trainer.training_config(arguments())

    def test_statistics_use_estimate_uncertainty_and_correct_coded_bit_order(self):
        batch = tensor_batch()  # deliberately no h_true or payload bits
        ordinals = torch.tensor([2, 0])
        actual = trainer.data_re_statistics(batch, ordinals)
        idx = batch["data_indices"][ordinals]
        y = batch["y"].reshape(2, 6, 4)[:, idx]
        h = batch["h_hat"].reshape(2, 6, 4, 2)[:, idx]
        variance = .3 + batch["err_var"].reshape(2, 6, 4, 2)[:, idx].sum(-1)
        expected_z = (h.mH @ (y / variance).unsqueeze(-1)).squeeze(-1)
        expected_gram = h.mH @ (h / variance.unsqueeze(-1))
        torch.testing.assert_close(actual["z"], expected_z.reshape(4, 2))
        torch.testing.assert_close(actual["gram"], expected_gram.reshape(4, 2, 2))
        for b in range(2):
            for j, ordinal in enumerate(ordinals):
                torch.testing.assert_close(actual["bits"][b * 2 + j], batch["coded_bits"][b, :, 4 * ordinal:4 * (ordinal + 1)])
        no_ce = trainer.data_re_statistics(dict(batch, err_var=torch.zeros_like(batch["err_var"])), ordinals)
        self.assertFalse(torch.allclose(no_ce["gram"], actual["gram"]))
        batch["metadata"]["csi"] = "perfect"
        with self.assertRaisesRegex(ValueError, "practical"):
            trainer.data_re_statistics(batch, ordinals)

    def test_sample_uses_one_waveform_and_only_practical_adapter(self):
        calls = []
        sample = object()
        link = SimpleNamespace(mode="paper_reference", csi="practical", codec=SimpleNamespace(qm=4),
                               transmit=lambda *args: calls.append(args) or sample)
        with patch.object(trainer, "make_detector_batch", return_value=tensor_batch()) as adapter:
            first = trainer.sample_re(link, -11., 73, 2)
            torch.manual_seed(999)
            second = trainer.sample_re(link, -11., 73, 2)
            self.assertEqual(calls, [(1, -11., 73), (1, -11., 73)])
            adapter.assert_called_with(link, sample, "practical")
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        link.csi = "perfect"
        with self.assertRaisesRegex(ValueError, "practical"):
            trainer.sample_re(link, -11., 73, 2)
        self.assertEqual(len(calls), 2)

    def test_paired_schedule_has_no_calibration_train_validation_overlap(self):
        args = arguments("--steps", "1000", "--val-channels", "3")
        first = trainer.sampling_schedule(args, [])
        calibration = first[0][:2] + first[1][:1]
        train, val, ebnos = trainer.sampling_schedule(args, calibration)
        self.assertFalse(set(train) & set(val) or (set(train) | set(val)) & set(calibration))
        self.assertEqual(len(set(train)), 1000)
        self.assertTrue(all(-13 <= value <= -9 for value in ebnos))
        self.assertAlmostEqual(sum(ebnos) / len(ebnos), -11., delta=.1)
        args.arch = "detr_ep"
        self.assertEqual((train, val, ebnos), trainer.sampling_schedule(args, calibration))

    def test_llr_sign_and_bce_gradient(self):
        bits = torch.tensor([0., 1., 0., 1.])
        correct = torch.tensor([-8., 8., -8., 8.], requires_grad=True)
        self.assertEqual(trainer.hard_errors(correct, bits), 0)
        self.assertLess(trainer.bit_loss(correct, bits).item(), .001)
        self.assertGreater(trainer.bit_loss(-correct, bits).item(), 7.)
        trainer.bit_loss(correct, bits).backward()
        self.assertTrue(torch.all(correct.grad[bits == 1] < 0))
        self.assertTrue(torch.all(correct.grad[bits == 0] > 0))
        with self.assertRaises(ValueError):
            trainer.bit_loss(correct[:, None], bits)

    def metadata_link(self):
        cfg = paper_config()
        cfg["channel_estimation"]["mode"] = "practical"
        return SimpleNamespace(mode="paper_reference", csi="practical", cfg=cfg,
                               codec=SimpleNamespace(qm=4, coded_bits=9216, info_bits=3104, metadata=lambda: {"qm": 4}),
                               grid_metadata={"num_data_symbols": 2304},
                               covariance_metadata=dict(path="prior.pt", sha256="a" * 64,
                                                        distribution_id=distribution_id(cfg), calibration_seeds=[4000000]))

    def test_checkpoint_provenance_round_trip_and_wrong_prior_rejection(self):
        link = self.metadata_link()
        metadata = trainer.checkpoint_metadata(arguments(), link, {"git_commit": "b" * 40})
        required = {"arch", "bits_per_symbol", "mcs_table", "mcs_index", "train_ebno_min_db", "train_ebno_max_db",
                    "val_ebno_db", "csi", "custom_ce_policy", "ce_covariance_path", "ce_covariance_sha256",
                    "distribution_id", "git_commit", "seed", "validation_metric"}
        self.assertTrue(required <= metadata.keys())
        self.assertEqual(metadata["custom_ce_policy"], "sionna_diagonal")
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            trainer.save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), metadata, 9, {"ber_neural": .2})
            saved = torch.load(path, weights_only=True)
        self.assertEqual((saved["step"], saved["validation_metric_value"], saved["csi"]), (9, .2, "practical"))
        self.assertEqual(saved["distribution_id"], distribution_id(link.cfg))
        for field, value in (("distribution_id", "legacy"), ("sha256", "")):
            invalid = copy.deepcopy(link)
            invalid.covariance_metadata[field] = value
            with self.assertRaisesRegex(ValueError, "provenance"):
                trainer.checkpoint_metadata(arguments(), invalid, {"git_commit": "b" * 40})

    def test_main_saves_final_and_best_without_accepting_old_output(self):
        # Test orchestration with a one-parameter stand-in, not a PHY or model benchmark.
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta = torch.nn.Parameter(torch.zeros(()))

            def forward(self, z, gram, **kwargs):
                return dict(llr=z.real[..., None].expand(-1, -1, 4) + self.delta,
                            correction_rms=self.delta.detach().abs().expand(5),
                            valid_update_fraction=torch.ones(5))

        baseline = TinyModel()
        ep_module = SimpleNamespace(ExpectationPropagationDetector=lambda *a, **kw: baseline)
        tiny = dict(z=torch.ones(2, 2, dtype=torch.complex64),
                    gram=torch.eye(2, dtype=torch.complex64).expand(2, 2, 2), bits=torch.zeros(2, 2, 4))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = ["--arch", "gt_ep", "--ce-covariance", "prior.pt", "--output-dir", str(output),
                    "--device", "cpu", "--steps", "2", "--val-channels", "1", "--val-every", "1"]
            with (patch.object(trainer, "training_config", return_value=paper_config()),
                  patch.object(trainer, "SionnaReferenceLink", return_value=self.metadata_link()),
                  patch.object(trainer, "make_model", return_value=TinyModel()),
                  patch.object(trainer, "sample_re", return_value=tiny) as sample,
                  patch.object(trainer, "provenance", return_value={"git_commit": "b" * 40}),
                  patch.dict(sys.modules, {"detectors.classical.ep": ep_module}),
                  patch("sys.stdout", new_callable=io.StringIO)):
                self.assertEqual(trainer.main(args), 0)
                self.assertEqual(sample.call_count, 3)  # one fixed validation, two training channels
                with self.assertRaises(FileExistsError):
                    trainer.main(args)
            report = json.loads((output / "history.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual([row["step"] for row in report["history"]], [0, 1, 2])
            saved = torch.load(output / "last.pth", weights_only=True)
            self.assertEqual(saved["step"], 2)
            self.assertLess(saved["model_state"]["delta"].item(), 0.)
            self.assertTrue((output / "best.pth").is_file())


class ModelAnchorTests(unittest.TestCase):
    """Exercise unchanged full networks/EP on CPU with a test-only alphabet.

    Only Sionna's constellation dependency is substituted locally. The separate
    native test below checks production constructors with actual Sionna 2.x.
    """

    def test_existing_gt_and_detr_anchor_and_trainable_final_llr(self):
        class TestConstellation:
            def __init__(self, *args, **kwargs):
                axis = torch.tensor([-3., -1., 1., 3.])
                self.points = (axis[:, None] + 1j * axis[None, :]).flatten() / 10 ** .5

        modules = {}
        paths = ("models.graph.gt_ep_detector", "models.baselines.detr_ep_detector", "detectors.classical.ep")
        with patch.dict(sys.modules, {"sionna.phy.mapping": SimpleNamespace(Constellation=TestConstellation)}):
            for name in paths:
                spec = importlib.util.spec_from_file_location("_test_" + name.replace(".", "_"), ROOT / (name.replace(".", "/") + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules[name] = module
        with patch.dict(sys.modules, modules):
            self.check_models()

    def check_models(self):
        from detectors.classical.ep import ExpectationPropagationDetector
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(16)
            cfg = dict(general=dict(precision="single", device="cpu"), modulation=dict(bits_per_symbol=4))
            h = torch.randn(3, 20, 16, dtype=torch.complex64) / 5
            y = torch.randn(3, 20, dtype=torch.complex64)
            validation = dict(z=(h.mH @ y[..., None]).squeeze(-1), gram=h.mH @ h,
                              bits=torch.randint(0, 2, (3, 16, 4)).float())
            ep = ExpectationPropagationDetector(cfg, num_iterations=5)
            for arch in ("gt_ep", "detr_ep"):
                with self.subTest(arch=arch):
                    model = trainer.make_model(cfg, arch)
                    metrics = trainer.validate(model, ep, validation, "cpu", 2, check_anchor=True)
                    self.assertEqual(metrics["ber_neural"], metrics["ber_ep5"])
                    self.assertLessEqual(metrics["anchor_max_abs_llr_difference"], 1e-5)
                    self.assertEqual(metrics["correction_rms"], [0.] * 5)
                    model.train()
                    output = model(validation["z"], validation["gram"])
                    trainer.bit_loss(output["llr"], validation["bits"]).backward()
                    head = model.graph_refiner.logit_head if arch == "gt_ep" else model.refiner.logit_head
                    self.assertTrue(torch.isfinite(head.weight.grad).all())
                    self.assertGreater(head.weight.grad.abs().max().item(), 0.)
                    with torch.no_grad():
                        head.bias[0] = .5
                    with self.assertRaisesRegex(RuntimeError, "anchor"):
                        trainer.validate(model, ep, validation, "cpu", 2, check_anchor=True)
        finally:
            torch.set_num_threads(old_threads)

    def test_native_sionna_model_constructors_and_anchor(self):
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        if version.split(".")[0] != "2":
            self.skipTest("Requires native Sionna 2.x; test-only alphabet is covered separately")
        self.check_models()


if __name__ == "__main__":
    unittest.main()
