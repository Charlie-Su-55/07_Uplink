"""CPU checks for shared conditional scales; no native link or weights required."""

import unittest

import torch

from models.conditional_llr_scaling import CANDIDATES, ConditionalScale, bound_statistics, scale_llr


class ConditionalModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(124)
        self.nodes = torch.randn(5, 4, 6, generator=generator)
        self.edges = torch.randn(5, 4, 4, 3, generator=generator)
        self.llr = torch.randn(5, 4, 4, generator=generator)
        self.llr[0, 0, 0] = 0

    def test_identity_initialization_and_finite_gradients(self):
        for candidate in CANDIDATES:
            with self.subTest(candidate=candidate):
                model = ConditionalScale(candidate)
                alpha = model(self.nodes, self.edges)
                self.assertTrue(torch.equal(alpha, torch.ones_like(self.llr)))
                self.assertTrue(torch.equal(scale_llr(self.llr, alpha), self.llr))
                (alpha * self.llr).sum().backward()
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()))
                self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)

    def test_nontrivial_permutation_equivariance(self):
        permutation = torch.tensor([2, 0, 3, 1])
        for candidate in CANDIDATES:
            with self.subTest(candidate=candidate):
                model = ConditionalScale(candidate).eval()
                with torch.no_grad():
                    model.head.weight.normal_(0, 0.2)
                    model.head.bias.normal_(0, 0.2)
                direct = model(self.nodes, self.edges)
                permuted = model(self.nodes[:, permutation], self.edges[:, permutation][:, :, permutation])
                self.assertGreater(float(direct.std()), 0)
                torch.testing.assert_close(permuted, direct[:, permutation], atol=1e-6, rtol=1e-6)

    def test_scale_bounds_and_hard_bits_survive_saturation(self):
        model = ConditionalScale("mlp")
        with torch.no_grad():
            model.head.bias.copy_(torch.tensor([-1000.0, 1000.0, 0.0, 0.2]))
        alpha = model(self.nodes, self.edges)
        statistics = bound_statistics(alpha)
        self.assertEqual(statistics["lower_count"], 20)
        self.assertEqual(statistics["upper_count"], 20)
        self.assertTrue(torch.equal(scale_llr(self.llr, alpha) > 0, self.llr > 0))
        self.assertTrue((alpha >= 0.05).all())
        self.assertTrue((alpha <= 8.0).all())

    def test_ablation_and_affine_feature_restrictions(self):
        for candidate in ("affine", "mlp_no_residual_eta"):
            model = ConditionalScale(candidate)
            with torch.no_grad():
                model.head.weight.fill_(0.1)
            changed = self.nodes.clone()
            indices = slice(0, 4) if candidate == "affine" else slice(4, 6)
            changed[..., indices] = 90
            self.assertTrue(torch.equal(model(self.nodes, self.edges), model(changed, self.edges)))

    def test_graph_uses_edges_and_single_user_remains_finite(self):
        model = ConditionalScale("graph")
        with torch.no_grad():
            model.head.weight.normal_(0, 0.5)
        self.assertFalse(torch.equal(model(self.nodes, self.edges), model(self.nodes, self.edges + 1)))
        self.assertTrue(torch.isfinite(model(self.nodes[:, :1], self.edges[:, :1, :1])).all())

    def test_reject_invalid_shapes_and_nonpositive_scales(self):
        model = ConditionalScale("graph")
        with self.assertRaises(ValueError):
            model(self.nodes[..., :5], self.edges)
        with self.assertRaises(ValueError):
            model(self.nodes, self.edges[:, :, :2])
        with self.assertRaises(ValueError):
            scale_llr(self.llr, torch.zeros_like(self.llr))
        with self.assertRaises(ValueError):
            scale_llr(self.llr, torch.ones_like(self.llr, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
