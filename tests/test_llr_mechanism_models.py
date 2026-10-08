"""CPU-only matched-capacity masks and fixed N0 interpolation tests."""

import copy
import unittest

import torch

from models.conditional_llr_scaling import scale_llr
from models.llr_mechanism_controls import CANDIDATES, MASKED_FEATURES, MechanismControl, interpolate_noise_alpha


def noise_fixture():
    return dict(alpha_min=0.05, alpha_max=8.0, interpolation="linear_log_alpha_log_n0_endpoint_hold",
                knots=[dict(n0=1.0, alpha=[1.0] * 4), dict(n0=4.0, alpha=[4.0] * 4),
                       dict(n0=16.0, alpha=[0.25] * 4)])


class MechanismModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(175)
        self.nodes = torch.randn(7, 4, 6, generator=generator)
        self.llr = torch.randn(7, 4, 4, generator=generator)

    def test_matched_parameter_counts_initial_identity_and_gradients(self):
        for candidate in CANDIDATES:
            model = MechanismControl(candidate)
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 28 if candidate == "affine_all_features" else 1412)
            alpha = model(self.nodes)
            self.assertTrue(torch.equal(alpha, torch.ones_like(self.llr)))
            self.assertTrue(torch.equal(scale_llr(self.llr, alpha), self.llr))
            (alpha * self.llr).sum().backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_masked_features_cannot_affect_nonzero_head_outputs(self):
        for candidate in CANDIDATES:
            model = MechanismControl(candidate)
            with torch.no_grad():
                model.head.weight.normal_(0, 0.2)
            baseline = model(self.nodes)
            for feature in MASKED_FEATURES[candidate]:
                changed = self.nodes.clone()
                changed[..., feature] = 999
                self.assertTrue(torch.equal(model(changed), baseline))
            if candidate == "affine_all_features":
                with torch.no_grad():
                    model.head.weight.fill_(0.01)
                for feature in range(6):
                    changed = torch.zeros_like(self.nodes)
                    changed[..., feature] = 1
                    self.assertFalse(torch.equal(model(changed), model(torch.zeros_like(changed))))

    def test_permutation_equivariance_and_positive_hard_decisions(self):
        permutation = torch.tensor([3, 1, 0, 2])
        for candidate in CANDIDATES:
            model = MechanismControl(candidate)
            with torch.no_grad():
                model.head.weight.normal_(0, 0.2)
                model.head.bias.copy_(torch.tensor([-100.0, 100.0, 0.1, 0.0]))
            alpha = model(self.nodes)
            torch.testing.assert_close(model(self.nodes[:, permutation]), alpha[:, permutation], atol=0, rtol=0)
            self.assertTrue(((alpha >= .05) & (alpha <= 8)).all())
            self.assertTrue(torch.equal(scale_llr(self.llr, alpha) > 0, self.llr > 0))

    def test_noise_knots_log_interpolation_and_endpoint_hold(self):
        fitted = noise_fixture()
        for noise, expected in ((0.2, 1), (1, 1), (2, 2), (4, 4), (8, 1), (16, .25), (100, .25)):
            alpha = interpolate_noise_alpha(noise, fitted)
            self.assertEqual(alpha.dtype, torch.float32)
            torch.testing.assert_close(alpha, torch.full((4,), float(expected)), atol=1e-7, rtol=1e-7)
        changed = copy.deepcopy(fitted)
        changed["confirmation_labels"] = "These never enter interpolation"
        self.assertTrue(torch.equal(interpolate_noise_alpha(2, fitted), interpolate_noise_alpha(2, changed)))

    def test_noise_interpolation_rejects_invalid_contracts(self):
        for noise in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                interpolate_noise_alpha(noise, noise_fixture())
        fitted = noise_fixture()
        fitted["knots"][1]["n0"] = 1
        with self.assertRaises(ValueError):
            interpolate_noise_alpha(2, fitted)
        fitted = noise_fixture()
        fitted["knots"][1]["alpha"][0] = 10
        with self.assertRaises(ValueError):
            interpolate_noise_alpha(2, fitted)


if __name__ == "__main__":
    unittest.main()
