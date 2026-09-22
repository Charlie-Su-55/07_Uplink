"""Synthetic CPU tests for Flow contracts; no UMa generation or real training."""
import ast
import copy
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from models.baselines.flow_matching_detector import FlowMatchingDetector, flow_from_checkpoint, square_qam

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, namespace=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = dict(torch=torch, copy=copy, json=json, Path=Path)
    ns.update(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), ns)
    return ns


@unittest.skipIf(torch is None, "Synthetic Flow tensor tests require PyTorch")
class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)

    def model(self, qm=4, streams=3, **kwargs):
        cfg = {"general": {"num_ues": streams, "streams_per_ue": 1}, "modulation": {"bits_per_symbol": qm}}
        options = dict(d_model=16, num_heads=2, num_layers=1,
                       ffn_dim=32, num_samples=8, sample_chunk=3, re_chunk=2)
        options.update(kwargs)
        return FlowMatchingDetector(cfg, **options)

    def statistics(self, n=5, streams=3):
        h = torch.randn(n, 7, streams, dtype=torch.complex64)
        y = torch.randn(n, 7, dtype=torch.complex64)
        return (h.mH @ y[..., None]).squeeze(-1), h.mH @ h

    def test_qam_labels_power_and_sign(self):
        for qm in (2, 4, 6):
            model = self.model(qm, streams=1).eval()
            m = 2 ** qm
            expected_first = {2: 1 / 2 ** .5, 4: 1 / 10 ** .5, 6: 3 / 42 ** .5}[qm]
            self.assertAlmostEqual(model.points[0].real.item(), expected_first, places=6)
            self.assertAlmostEqual(model.points.abs().square().mean().item(), 1, places=6)
            z = 50 * model.points[:, None]
            gram = torch.full((m, 1, 1), 50, dtype=torch.complex64)
            out = model(z, gram)
            self.assertTrue(torch.equal((out["llr"][:, 0] > 0), model.bit_table.bool()))
            self.assertTrue(torch.equal(out["llr"], out["lmmse_llr"]))

    def test_single_stream_lmmse_matches_exact_app_not_max_log(self):
        model = self.model(streams=1).eval()
        z = torch.tensor([[.1 + .5j], [.8 - .2j]], dtype=torch.complex64)
        g = torch.full((2, 1, 1), .75, dtype=torch.complex64)
        llr = model(z, g)["llr"]
        metric = -.75 * model.points.abs().square() + 2 * (z[..., None] * model.points.conj()).real
        expected = torch.stack([
            torch.logsumexp(metric[..., model.bit_table[:, b] == 1], -1)
            - torch.logsumexp(metric[..., model.bit_table[:, b] == 0], -1)
            for b in range(4)], -1)
        torch.testing.assert_close(llr, expected, rtol=2e-5, atol=2e-6)

    def test_modulation_and_grid_dimensions(self):
        for qm in (2, 4, 6):
            model = self.model(qm).eval()
            z, g = self.statistics(12)
            z = z.reshape(2, 2, 3, 3).transpose(1, 2)
            g = g.reshape(2, 2, 3, 3, 3).transpose(1, 2)
            result = model(z, g, return_iterations=(5,))
            for value in result.values():
                self.assertEqual(value.shape, (2, 3, 2, 3, qm))
                self.assertTrue(torch.isfinite(value).all())
            flat = model(z.reshape(-1, 3), g.reshape(-1, 3, 3))
            torch.testing.assert_close(result["llr"].reshape(-1, 3, qm), flat["llr"])

    def test_sixteen_stream_contract(self):
        model = self.model(streams=16).eval()
        g = torch.eye(16, dtype=torch.complex64)[None]
        z = torch.ones(1, 16, dtype=torch.complex64)
        self.assertEqual(model(z, g)["llr"].shape, (1, 16, 4))

    def test_raw_frontend_matches_statistics_and_unit_scaling(self):
        from data.preprocessing.whitening import CovarianceAwareFrontEnd
        model = self.model().eval()
        y = torch.randn(2, 2, 3, 7, dtype=torch.complex64)
        h = torch.randn(2, 2, 3, 7, 3, dtype=torch.complex64)
        a = torch.randn(2, 7, 7, dtype=torch.complex64)
        r = a @ a.mH + torch.eye(7, dtype=torch.complex64)
        stats = CovarianceAwareFrontEnd()(y, h, r)
        expected = model(stats["z"], stats["gram"])["llr"]
        actual = model.detect(y, h, r)["llr"]
        torch.testing.assert_close(actual, expected)
        scaled = model.detect(y * 3, h * 3, r * 9)["llr"]
        torch.testing.assert_close(scaled, expected, rtol=5e-4, atol=1e-4)
        with self.assertRaises(ValueError):
            model.detect(y, h, r[:1])

    def test_flow_and_llr_head_receive_finite_gradients(self):
        model = self.model().train()
        z, g = self.statistics()
        bits = torch.randint(2, (5, 3, 4)).float()
        result = model.training_loss(z, g, bits)
        result["loss"].backward()
        for group in (model.proposal_head, model.blocks, model.context_in, model.llr_head):
            grads = [p.grad for p in group.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
            self.assertGreater(sum(float(grad.abs().sum()) for grad in grads), 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.step()
        self.assertGreater(float((model(z, g)["llr"] - model(z, g)["lmmse_llr"]).abs().max()), 0)

    def test_chunk_invariance_and_no_simulator_rng_consumption(self):
        model = self.model().eval()
        torch.nn.init.normal_(model.llr_head[-1].weight, std=.02)
        z, g = self.statistics()
        state = torch.random.get_rng_state().clone()
        reference = model(z, g)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        model.re_chunk, model.sample_chunk = 4, 8
        actual = model(z, g)
        for key in reference:
            torch.testing.assert_close(actual[key], reference[key], rtol=2e-4, atol=2e-5)
        split = torch.cat([model(z[i:i+1], g[i:i+1])["llr"] for i in range(5)])
        torch.testing.assert_close(split, reference["llr"], rtol=2e-4, atol=2e-5)

    def test_proposal_stream_equivariance(self):
        model = self.model().eval()
        z, g = self.statistics(2)
        state = torch.tensor([[16, 2, 16], [1, 16, 7]])
        perm = torch.tensor([2, 0, 1])
        expected = model._proposal(state, model._context(z, g))[:, perm]
        actual = model._proposal(state[:, perm], model._context(z[:, perm], g[:, perm][:, :, perm]))
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_invalid_inputs_fail(self):
        model = self.model()
        z, g = self.statistics()
        invalid = [(z.real, g), (z, g[:, :2]), (z[:0], g[:0]),
                   (z[:, :2], g), (z * float("nan"), g)]
        for zz, gg in invalid:
            with self.subTest(shape=zz.shape), self.assertRaises(ValueError):
                model(zz, gg)
        with self.assertRaises(ValueError):
            model.training_loss(z, g, torch.ones(5, 3, 2))
        with self.assertRaises(ValueError):
            model.training_loss(z, g, torch.full((5, 3, 4), .5))
        with self.assertRaises(ValueError):
            self.model(sample_chunk=0)
        with self.assertRaises(ValueError):
            square_qam(3)

    def test_checkpoint_roundtrip_and_provenance(self):
        model = self.model().eval()
        cfg = {"general": {"num_ues": 3, "streams_per_ue": 1}, "modulation": {"bits_per_symbol": 4}}
        checkpoint = dict(arch="flow_matching", implementation_id=model.implementation_id,
                          model_config=model.model_config, bits_per_symbol=4,
                          csi="lmmseH", covariance="estimated_Ruu", model_state=model.state_dict())
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        rng = torch.random.get_rng_state().clone()
        loaded = flow_from_checkpoint(cfg, checkpoint).eval()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        z, g = self.statistics()
        torch.testing.assert_close(model(z, g)["raw_flow_llr"], loaded(z, g)["raw_flow_llr"])
        for key, wrong in (("arch", "gt_ep"), ("bits_per_symbol", 6), ("csi", "trueH"),
                           ("implementation_id", "partner"), ("model_config", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                flow_from_checkpoint(cfg, dict(checkpoint, **{key: wrong}))

    def test_bler_mapping_and_dynamic_chunk_adapter(self):
        ns = extract("evaluation/evaluate_mcs_bler.py",
                     {"read_checkpoint_map", "load_specialist_models", "run_neural_chunks"},
                     {"SPECIALIST_CHECKPOINTS": {}, "load_neural_model": lambda *args: args[2]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            path.write_text(json.dumps({"1:11": {"gt": "g", "detr": "d", "flow": "f"}}))
            mapping = ns["read_checkpoint_map"](path)
            models = ns["load_specialist_models"]({}, 1, 11, 4, "cpu", mapping)
            self.assertEqual(set(models), {"gt", "detr", "flow"})
            path.write_text(json.dumps({"1:11": {"gt": "g", "detr": "d", "unknown": "f"}}))
            with self.assertRaises(ValueError):
                ns["read_checkpoint_map"](path)
        model = self.model().eval()
        z, g = self.statistics()
        llr = ns["run_neural_chunks"](model, z, g, 4, 2)
        self.assertEqual(llr.shape, (5, 3, 4))
        torch.testing.assert_close(llr, model(z, g)["llr"])

    def test_coded_grid_roundtrip_order(self):
        ns = extract("evaluation/evaluate_mcs_bler.py", {"llr_grid_to_codeword"})
        runtime = SimpleNamespace(num_coded_bits=2 * 3 * 4)
        llr = torch.arange(2 * 3 * 3 * 4).reshape(1, 2, 3, 3, 4)
        codeword = ns["llr_grid_to_codeword"](llr, runtime, 3)
        for user in range(3):
            torch.testing.assert_close(codeword[0, user], llr[0, :, :, user].reshape(-1))

    def test_sionna_app_anchor_when_server_available(self):
        try:
            version = importlib.metadata.version("sionna")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("Sionna 2 server environment required")
        if int(version.split(".")[0]) < 2:
            self.skipTest("Sionna 2 server environment required")
        from sionna.phy.mapping import Constellation
        from detectors.classical.lmmse import LMMSESoftDetector
        for qm in (2, 4, 6):
            cfg = {"general": {"device": "cpu", "precision": "single"},
                   "modulation": {"bits_per_symbol": qm}}
            model = self.model(qm).eval()
            constellation = Constellation("qam", qm, device="cpu", precision="single")
            torch.testing.assert_close(model.points, constellation.points)
            y = torch.randn(1, 1, 2, 7, dtype=torch.complex64)
            h = torch.randn(1, 1, 2, 7, 3, dtype=torch.complex64)
            r = torch.eye(7, dtype=torch.complex64)[None]
            reference = LMMSESoftDetector(cfg)(y, h, r)["llr"]
            torch.testing.assert_close(model.detect(y, h, r)["llr"], reference, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
