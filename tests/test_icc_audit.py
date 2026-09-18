"""CPU control-flow tests. AST extraction avoids importing the GPU/Sionna stack.

The end-to-end sweep test uses synthetic point callbacks, not a radio simulator.
Run with: python -m unittest discover -s tests -v
"""
import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names), names
    ns = dict(math=math, Path=Path, warnings=warnings, copy=copy, json=json, csv=csv, hashlib=hashlib)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"SPECIALIST_CHECKPOINTS", "DETECTOR_LABELS", "MOD_NAMES"}:
                ns[node.targets[0].id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), ns)
    return ns


def point(snr, bler, n=1000):
    return {"snr_db": snr, "x_bler": bler, "total_blocks": n}


class BLERTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions("evaluation/evaluate_mcs_bler.py", {
            "interpolate_snr_at_bler", "extend_bler_grid", "choose_formal_grid", "read_checkpoint_map",
            "save_csv", "extract_state", "load_neural_model", "load_specialist_models", "main",
            "parse_int_list", "parse_float_list",
        })
        self.interp = self.ns["interpolate_snr_at_bler"]

    def test_interpolation_boundaries(self):
        self.assertEqual(self.interp([point(8, .1)], "x_bler", .1), 8)
        self.assertIsNone(self.interp([], "x_bler", .1))
        self.assertAlmostEqual(self.interp([point(9, .01), point(7, 1)], "x_bler", .1), 8)
        self.assertIsNone(self.interp([point(7, .09), point(8, .01)], "x_bler", .1))
        self.assertIsNone(self.interp([point(0, .2, 10), point(1, 0, 10)], "x_bler", .01))
        value = self.interp([point(0, .2), point(1, 0)], "x_bler", .1)
        self.assertTrue(0 <= value <= 1)

    def test_reject_invalid_observations(self):
        for target in (0, 1, -1, float("nan")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.interp([point(1, .1)], "x_bler", target)
        for observations in ([point(1, float("nan"))], [point(1, 1.1)], [point(1, .1, 0)],
                             [point(float("inf"), .1)], [point(1, .2), point(1, .1)]):
            with self.subTest(points=observations), self.assertRaises(ValueError):
                self.interp(observations, "x_bler", .1)

    def test_low_high_extension_and_shared_points(self):
        points = [{"snr_db": 8., "a_bler": .02, "b_bler": .8, "total_blocks": 1000}]
        calls, saves = [], []
        def evaluate(snr):
            calls.append(snr)
            return {"snr_db": snr, "a_bler": .2 if snr < 8 else .02,
                    "b_bler": .01 if snr > 8 else .8, "total_blocks": 1000}
        messages = self.ns["extend_bler_grid"](points, ["a_bler", "b_bler"], .1, .5, 4,
                                               evaluate, lambda: saves.append(len(points)))
        self.assertEqual(calls, [7.5, 8.5])
        self.assertEqual(saves, [2, 3])
        self.assertEqual(messages, [])
        self.assertIsNotNone(self.interp(points, "a_bler", .1))
        self.assertIsNotNone(self.interp(points, "b_bler", .1))

    def test_extension_budget_and_resolution_warning(self):
        for points, limit, expected_calls in (([point(8, .01)], 2, 2), ([point(8, .01)], 0, 0),
                                             ([point(0, .2, 10), point(1, 0, 10)], 3, 0)):
            calls = []
            def evaluate(snr):
                calls.append(snr)
                return point(snr, .01)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                messages = self.ns["extend_bler_grid"](points, ["x_bler"], .01 if len(points) == 2 else .1,
                                                       .5, limit, evaluate)
            self.assertEqual(len(calls), expected_calls)
            self.assertTrue(messages and caught)

    def test_checkpoint_mapping_uses_table_and_mcs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps({"1:11": {"gt": "first", "detr": "second"},
                                        "2:5": {"gt": "third", "detr": "fourth"}}))
            mapping = self.ns["read_checkpoint_map"](path)
            self.ns["load_neural_model"] = lambda cfg, qm, arch, path, device, table, mcs: (path, table, mcs)
            load = self.ns["load_specialist_models"]
            self.assertEqual(load({}, 1, 11, 4, "cpu", mapping)["gt"], ("first", 1, 11))
            self.assertEqual(load({}, 2, 5, 4, "cpu", mapping)["gt"], ("third", 2, 5))
            with self.assertRaises(KeyError):
                load({}, 1, 19, 6, "cpu", mapping)

    def test_checkpoint_provenance_rejects_wrong_specialist(self):
        class Model:
            def load_state_dict(self, state, strict):
                self.loaded = (state, strict)
            def to(self, device):
                return self
            def eval(self):
                pass
        state = {"model_state": {}, "arch": "gt_ep", "bits_per_symbol": 4, "csi": "lmmseH",
                 "covariance": "estimated_Ruu", "mcs_table": 1, "mcs_index": 11}
        self.ns.update(torch=SimpleNamespace(load=lambda *a, **k: state), make_neural_model=lambda *a: Model())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pth"
            path.write_bytes(b"metadata-test-only")
            load = self.ns["load_neural_model"]
            cfg = {"modulation": {"bits_per_symbol": 4}}
            with contextlib.redirect_stdout(io.StringIO()):
                model = load(cfg, 4, "gt", path, "cpu", 1, 11)
            self.assertTrue(model.loaded[1])
            self.assertEqual(model.checkpoint_metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            for key, wrong in (("mcs_index", 5), ("mcs_table", 2), ("csi", "trueH"),
                               ("bits_per_symbol", 6), ("arch", "detr_ep"), ("covariance", "true_Ruu")):
                original = state[key]
                state[key] = wrong
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load(cfg, 4, "gt", path, "cpu", 1, 11)
                state[key] = original

    def run_sweep_fixture(self, directory, interrupt=False, extra=()):
        calls = []
        base_cfg = {"general": {"device": "cpu", "precision": "single", "num_ues": 16, "streams_per_ue": 1},
                    "modulation": {"bits_per_symbol": 4}, "link": {}}
        runtime = SimpleNamespace(qm=4, target_rate=.369, spectral_efficiency=1.476,
                                  tb_size=2856, num_coded_bits=7680)
        models = {name: SimpleNamespace(checkpoint_metadata={"path": name}) for name in ("gt", "detr")}
        def evaluate(base_cfg, runtime, models, snr_db, channels, seed, gt_chunk, verbose=True):
            calls.append((snr_db, channels, seed))
            if interrupt and len(calls) == 2:
                raise RuntimeError("synthetic interruption")
            result = {"snr_db": snr_db, "channels": channels, "total_blocks": channels * 16,
                      "seed": seed, "channel_counts": []}
            for name in ("lmmse", "ep5", "gt", "detr"):
                crossing = 7.25 if name in {"gt", "detr"} else 8
                result[f"{name}_bler"] = min(1., .1 * 10 ** (crossing - snr_db))
                result[f"{name}_ber"] = result[f"{name}_bler"] / 10
                result[f"{name}_block_errors"] = round(result[f"{name}_bler"] * channels * 16)
                result[f"{name}_bit_errors"] = result[f"{name}_block_errors"]
            return result
        self.ns.update(argparse=argparse, sys=sys, as_int=int,
            subprocess=SimpleNamespace(check_output=lambda argv, **k: "fixture" if argv[1] == "rev-parse" else "", DEVNULL=None, CalledProcessError=RuntimeError),
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                  set_float32_matmul_precision=lambda *a: None, manual_seed=lambda *a: None,
                                  __version__="fixture"),
            importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda *a: "fixture")),
            sionna_config=SimpleNamespace(), load_yaml=lambda path: base_cfg if path == "system" else {"system_config": "system"},
            UplinkUMADataset=lambda cfg: SimpleNamespace(channel=SimpleNamespace(resource_grid=SimpleNamespace(data_symbols=range(10), num_subcarriers=192))),
            decode_mcs_index=lambda *a, **k: (4, .369), build_mcs_runtime=lambda **k: runtime,
            load_specialist_models=lambda *a: models, codec_identity_test=lambda *a: None,
            evaluate_snr_point=evaluate)
        cache = Path("data/cache/uma_lmmse_ft_cov.pt")
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cache-fixture")
        for paths in self.ns["SPECIALIST_CHECKPOINTS"].values():
            for item in paths.values():
                path = Path(item)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"checkpoint-fixture")
        argv = ["fixture", "--coarse-snrs", "8,9", "--coarse-channels", "10", "--formal-channels", "100",
                "--formal-margin-db", "0", "--formal-step-db", "0.5", "--max-bracket-extensions", "4",
                "--output-dir", str(Path(directory) / "out"), *extra]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            if interrupt:
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    self.ns["main"]()
            else:
                self.ns["main"]()
        payload = json.loads((Path(directory) / "out/mcs_bler_results.json").read_text())
        return calls, payload

    def test_formal_bracketing_main_and_seed_consistency(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            calls, payload = self.run_sweep_fixture(directory)
            result = payload["results"][0]
            self.assertEqual(result["status"], "complete")
            self.assertAlmostEqual(result["snr_at_target_db"]["gt"], 7.25)
            self.assertEqual([call[0] for call in calls], [8., 9., 8., 7.5, 7.])
            self.assertEqual([call[1] for call in calls], [10, 10, 100, 100, 100])
            self.assertEqual(len({call[2] for call in calls}), 1)
            self.assertEqual(len(result["formal_points"]), 3)

    def test_partial_sweep_survives_interruption(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            _, payload = self.run_sweep_fixture(directory, interrupt=True)
            result = payload["results"][0]
            self.assertEqual(result["status"], "running")
            self.assertEqual(len(result["coarse_points"]), 1)
            self.assertEqual(result["formal_points"], [])

    def test_skip_formal_never_launches_extensions(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calls, payload = self.run_sweep_fixture(directory, extra=("--skip-formal",))
            self.assertEqual(len(calls), 2)
            self.assertEqual(payload["results"][0]["mode"], "coarse_diagnostic")
            self.assertEqual(payload["results"][0]["status"], "unresolved")

    def test_operating_point_schema_and_no_extension(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            calls, payload = self.run_sweep_fixture(directory, extra=("--operating-point",))
            result = payload["results"][0]
            self.assertEqual(calls, [(8., 100, 20260911 + 100000 + 11000)])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["mode"], "operating_point")
            with open(Path(directory) / "out/mcs_bler_curves.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertNotIn("channel_counts", rows[0])
            self.assertEqual(json.loads(rows[0]["checkpoints"])["gt"]["path"], "gt")



class ArtifactTests(unittest.TestCase):
    def test_training_refuses_silent_overwrite(self):
        check = functions("training/train_gt_detr_lmmse.py", {"check_output_paths"})["check_output_paths"]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            ckpt, history = out / "best.pth", out / "history.json"
            check(out, history)
            ckpt.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                check(out, history)
            check(out, history, fresh=True)
            self.assertEqual(ckpt.read_bytes(), b"keep")
            ckpt.unlink()
            history.write_text("keep history")
            with self.assertRaises(FileExistsError):
                check(out, history)

    def test_ber_name_reflects_runtime(self):
        ns = functions("evaluation/evaluate_gt_detr_lmmse.py", {"default_output_path"})
        args = SimpleNamespace(snr_db=8., bits_per_symbol=6, channels=7, re_per_channel=32, seed=123)
        self.assertEqual(ns["default_output_path"](args).name, "lmmseH_gt_vs_detr_64qam_7ch_32re_8db_seed123.json")

    def test_no_dangling_local_imports_or_ua(self):
        roots = {"data", "detectors", "models", "training", "evaluation", "link_level"}
        for root in roots:
            for path in (ROOT / root).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in roots:
                        target = ROOT.joinpath(*node.module.split("."))
                        self.assertTrue(target.with_suffix(".py").exists() or target.is_dir(), (path, node.module))
        for path in ("training/train_gt_detr_lmmse.py", "models/graph/gt_ep_detector.py"):
            source = (ROOT / path).read_text()
            self.assertNotIn("ce_uncertainty", source)
            self.assertNotIn("ua_gt_ep", source)


if __name__ == "__main__":
    unittest.main()
