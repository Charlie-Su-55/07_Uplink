"""Small synthetic shared-frontend/cache tests; never generate a 256Rx UMa slot."""

import copy
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evaluation import build_paper_soft_cache as builder
from evaluation import paper_soft_common as common
from link_level.detector_adapter import make_detector_batch
from training.train_paper_reference_detector import data_re_statistics


class TinyCodec:
    def decode(self, llr):
        decoded = (llr.squeeze(2)[..., :7] > 0).float()
        return decoded, (decoded.sum(-1).long() % 2) == 0


class TinyLink:
    num_rx, num_ues, device = 4, 2, "cpu"

    def __init__(self):
        generator = torch.Generator().manual_seed(23)
        self.y = torch.randn(1, 3, 4, 4, dtype=torch.complex64, generator=generator)
        self.estimate = torch.randn(1, 3, 4, 4, 2, dtype=torch.complex64, generator=generator)
        self.error = torch.rand(1, 3, 4, 4, 2, generator=generator) * .04
        self.data_indices = torch.tensor([0, 2, 4, 8, 11])
        self.coded = torch.randint(0, 2, (1, 2, 20), generator=generator).float()
        self.codec = TinyCodec()
        self.transmit_calls = self.estimation_calls = self.codec_checks = 0

    def transmit(self, size, ebno, seed):
        self.transmit_calls += 1
        return dict(y=self.y.permute(0, 3, 1, 2).unsqueeze(1), h_eff=self.estimate * 7,
                    info=self.coded[..., :7].clone(), coded=self.coded.clone(), no=torch.tensor(.4),
                    seed=seed, ebno_db=ebno)

    def estimate_channel(self, sample, csi):
        cache = sample.setdefault("csi_estimates", {})
        if csi not in cache:
            self.estimation_calls += 1
            cache[csi] = tuple(value.permute(0, 3, 4, 1, 2).unsqueeze(1).unsqueeze(4)
                               for value in (self.estimate, self.error))
        return cache[csi]

    def receive(self, sample, csi):
        self.estimate_channel(sample, csi)
        llr = (2 * self.coded - 1).unsqueeze(2)
        decoded, crc = self.codec.decode(llr)
        return dict(llr=llr, decoded=decoded, crc_ok=crc)

    def check_codec_identity(self, sample):
        self.codec_checks += 1


class TinyReceiver:
    def __init__(self, capture):
        self.capture = capture

    def __call__(self, matched, gram, return_iterations):
        self.capture.append((id(matched), id(gram)))
        variance = torch.full_like(matched.real, .2)
        mean = matched / (gram.diagonal(dim1=-2, dim2=-1).real + 2)
        llr = matched.real[..., None] + torch.tensor([-.3, -.1, .1, .3])
        return dict(llr=llr, x_hat=mean, posterior_variance=variance)


def settings():
    return dict(re_chunk=2, features=dict(ce_epsilon=1e-12, residual_atol=1e-5, residual_rtol=5e-6))


class CacheFeatureTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def batch(self, link):
        sample = link.transmit(1, -10.5, 81)
        return sample, self.make_batch(link, sample, "practical")

    @staticmethod
    def make_batch(link, sample, csi):
        estimate, error = link.estimate_channel(sample, csi)
        return dict(y=link.y, h_hat=estimate[:, 0, :, :, 0].permute(0, 3, 4, 1, 2),
                    err_var=error[:, 0, :, :, 0].permute(0, 3, 4, 1, 2), h_true=sample["h_eff"],
                    n0=sample["no"], Ruu=sample["no"] * torch.eye(link.num_rx, dtype=torch.complex64)[None],
                    coded_bits=sample["coded"], bits=sample["info"], data_indices=link.data_indices,
                    metadata={"csi": "practical"})

    def test_statistics_match_existing_training_frontend_and_no_oracle_features(self):
        link = TinyLink()
        sample, batch = self.batch(link)
        ordinals = torch.tensor([0, 2, 4])
        actual = builder.shared_statistics(batch, ordinals, 1e-12)
        expected = data_re_statistics(batch, ordinals)
        self.assertTrue(torch.equal(actual["z"].reshape(-1, 2), expected["z"]))
        self.assertTrue(torch.equal(actual["gram"].reshape(-1, 2, 2), expected["gram"]))
        changed = copy.deepcopy(batch)
        for name in ("h_true", "bits", "coded_bits"):
            changed[name].fill_(float("nan"))
        changed["realized_noise"] = torch.full_like(batch["y"], float("nan"))
        repeated = builder.shared_statistics(changed, ordinals, 1e-12)
        for name in actual:
            self.assertTrue(torch.equal(actual[name], repeated[name]))
        indices = batch["data_indices"][ordinals]
        estimate = batch["h_hat"].reshape(1, -1, 4, 2)[:, indices]
        variance = batch["err_var"].reshape_as(batch["h_hat"]).reshape(1, -1, 4, 2)[:, indices]
        expected_eta = variance.sum(-2) / (estimate.abs().square().sum(-2) + variance.sum(-2) + 1e-12)
        self.assertTrue(torch.equal(actual["eta"], expected_eta))

    def test_residual_matches_direct_whitened_posterior_expectation_and_rejects_negative(self):
        generator = torch.Generator().manual_seed(74)
        received = torch.randn(1, 5, 4, dtype=torch.complex64, generator=generator)
        channel = torch.randn(1, 5, 4, 2, dtype=torch.complex64, generator=generator)
        mean = torch.randn(1, 5, 2, dtype=torch.complex64, generator=generator)
        variance = torch.rand(1, 5, 2, generator=generator)
        gram = channel.mH @ channel
        statistics = dict(z=(channel.mH @ received[..., None]).squeeze(-1), gram=gram,
                          received_energy=received.abs().square().sum(-1))
        residual = builder.posterior_residual(statistics, mean, variance, 4, 1e-5, 5e-6)
        expected = ((received - (channel @ mean[..., None]).squeeze(-1)).abs().square().sum(-1)
                    + (channel.abs().square().sum(-2) * variance).sum(-1)) / 4
        torch.testing.assert_close(residual, expected, atol=1e-5, rtol=5e-6)
        bad = dict(statistics, received_energy=torch.full((1, 5), -1e6))
        with self.assertRaisesRegex(ValueError, "negative values"):
            builder.posterior_residual(bad, mean, variance, 4, 1e-5, 5e-6)

    def test_one_transmit_one_estimate_and_one_frontend_per_chunk_all_methods_share_statistics(self):
        link, captured = TinyLink(), []
        ep = TinyReceiver(captured)
        models = {name: TinyReceiver(captured) for name in common.METHODS[2:]}
        with (patch.object(builder, "make_detector_batch", side_effect=self.make_batch),
              patch.object(builder, "shared_statistics", wraps=builder.shared_statistics) as frontend):
            payload = builder.build_channel(link, models, ep, "development", -10.5, 82, "tiny", settings(), True)
        self.assertEqual((link.transmit_calls, link.estimation_calls, link.codec_checks), (1, 1, 1))
        self.assertEqual(frontend.call_count, 3)
        self.assertEqual(len(captured), 15)
        for start in range(0, 15, 5):
            self.assertEqual(len(set(captured[start:start + 5])), 1)
        self.assertEqual(payload["features"]["gram"].shape, (1, 5, 2, 2))
        for forbidden in ("y", "h_hat", "h_true", "realized_noise", "a1_candidates"):
            self.assertNotIn(forbidden, payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "channel.pt"
            common.atomic_torch_save(path, payload,
                verify=lambda temporary: builder.verify_cache_decode(temporary, link.codec, "cpu", "tiny"))
            loaded = builder.verify_cache_decode(path, link.codec, "cpu", "tiny")
            self.assertTrue(torch.equal(loaded["methods"]["ep5"]["decoder_input_llr"], payload["methods"]["ep5"]["decoder_input_llr"]))
            self.assertTrue(torch.equal(loaded["coded_bits"], link.coded))
            modified = copy.deepcopy(payload)
            modified["methods"]["ep5"]["crc_ok"] = ~modified["methods"]["ep5"]["crc_ok"]
            corrupted = Path(directory) / "incorrect_crc.pt"
            torch.save(modified, corrupted)
            with self.assertRaisesRegex(ValueError, "decode mismatch"):
                builder.verify_cache_decode(corrupted, link.codec, "cpu", "tiny")

    def test_repeated_cache_features_and_llrs_are_deterministic(self):
        values = []
        for _ in range(2):
            capture = []
            with patch.object(builder, "make_detector_batch", side_effect=self.make_batch):
                values.append(builder.build_channel(TinyLink(), {name: TinyReceiver(capture) for name in common.METHODS[2:]},
                              TinyReceiver(capture), "calibration", -10.5, 81, "tiny", settings()))
        for key in values[0]["features"]:
            self.assertTrue(torch.equal(values[0]["features"][key], values[1]["features"][key]))
        for name in common.METHODS:
            self.assertTrue(torch.equal(values[0]["methods"][name]["llr"], values[1]["methods"][name]["llr"]))

    def test_reuse_complete_cache_never_constructs_link_or_models(self):
        config = common.load_experiment_config(mode="smoke")
        identity = dict(experiment_config=config)
        checked = dict(identity=identity, identity_id=common.digest(identity), seed_plan={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir()
            common.save_json(root / "manifest.json", dict(schema_version=1, status="complete", channels=[], **checked))
            args = SimpleNamespace(config=common.DEFAULT_CONFIG, mode="smoke", run_id="test", split="all", preflight=False, reuse_cache=True)
            with (patch.object(builder, "preflight", return_value=checked),
                  patch.object(builder, "run_paths", return_value={"cache": root}),
                  patch.object(builder, "load_complete_cache") as validate,
                  patch.object(builder, "SionnaReferenceLink") as link,
                  patch.object(builder, "load_neural_checkpoint") as load,
                  patch("sys.stdout", new_callable=io.StringIO)):
                self.assertEqual(builder.run(args), 0)
                validate.assert_called_once_with(root)
                link.assert_not_called()
                load.assert_not_called()
                args.reuse_cache = False
                with self.assertRaises(FileExistsError):
                    builder.run(args)


class NativeCacheTests(unittest.TestCase):
    def test_small_native_practical_cache_matches_frozen_ep_and_tb_decoder(self):
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        if version != "2.0.1":
            self.skipTest("Requires server Sionna 2.0.1; local native link acceptance is not claimed")
        from tests.test_paper_reference import FlatChannel, paper_fixture
        from detectors.classical.ep import ExpectationPropagationDetector
        from link_level.detector_adapter import ClassicalDetectorAdapter
        from link_level.sionna_ce import COVARIANCE_KIND, distribution_id, regularize_covariance

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                config = paper_fixture()
                config["channel_estimation"]["mode"] = "practical"
                covariance = Path(directory) / "synthetic_prior.pt"
                torch.save(dict(kind=COVARIANCE_KIND, distribution_id=distribution_id(config), calibration_seeds=[4000000],
                                cov_mat_freq=regularize_covariance(4 * torch.ones(32, 32, dtype=torch.complex64)),
                                cov_mat_time=regularize_covariance(4 * torch.ones(4, 4, dtype=torch.complex64))), covariance)
                config["channel_estimation"]["covariance_path"] = str(covariance)
                link = builder.SionnaReferenceLink(config, channel_provider=FlatChannel())
                ep = ExpectationPropagationDetector(dict(config, modulation={"bits_per_symbol": 4}), num_iterations=5)
                payload = builder.build_channel(link, {name: ep for name in common.METHODS[2:]}, ep,
                                                 "calibration", 12., 42, "native", dict(settings(), re_chunk=13), True)
                sample = link.transmit(1, 12., 42)
                batch = make_detector_batch(link, sample, "practical")
                classical = ClassicalDetectorAdapter(link, detectors=("sionna_lmmse", "ep5"),
                                                      custom_ce_policy="sionna_diagonal", re_chunk=13).evaluate(sample, batch)
                for name in ("sionna_lmmse", "ep5"):
                    torch.testing.assert_close(payload["methods"][name]["decoder_input_llr"], classical[name]["llr"].cpu(), atol=1e-5, rtol=1e-5)
                    self.assertTrue(torch.equal(payload["methods"][name]["decoded_payload_bits"], classical[name]["decoded"].cpu()))
                path = Path(directory) / "native_channel.pt"
                common.atomic_torch_save(path, payload, verify=lambda temporary: builder.verify_cache_decode(temporary, link.codec, "cpu", "native"))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
