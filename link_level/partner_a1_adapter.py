"""Read-only adapter for the external, frozen A1_PARTNER_MINIMAL bundle.

The package's legacy one-click data source is never imported. Only its original
load_model/predict implementation is used; no proposal/readout code is copied.
"""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import torch

from link_level.detector_adapter import whiten_diagonal_uncertainty
from link_level.sionna_ce import distribution_id


DEFAULT_A1_ROOT = "../A1_PARTNER_MINIMAL"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(root):
    """Check package bytes before importing external Python or loading weights."""
    root = Path(root).resolve()
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    required = {"MODEL.json", "model.pt", "src/r3p5_invariant/portable/runtime.py"}
    if not required <= manifest.keys():
        raise ValueError("Incomplete A1 bundle manifest.")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError(f"A1 bundle identity mismatch: {name}")
    # An unlisted Python module must not bypass the checked package identity.
    for path in (root / "src").rglob("*.py"):
        if path.relative_to(root).as_posix() not in manifest:
            raise ValueError(f"Unlisted A1 Python source: {path}")
    record = json.loads((root / "MODEL.json").read_text(encoding="utf-8"))
    if (record.get("format"), record.get("variant"), record.get("step")) != ("a1_partner_minimal_v1", "combined", 6000):
        raise ValueError("Expected frozen A1 combined/6000-step bundle.")
    if record.get("model_file_sha256") != manifest["model.pt"]:
        raise ValueError("A1 model file differs from MODEL.json.")
    return dict(root=str(root), manifest_sha256=file_sha256(root / "BUNDLE_MANIFEST.json"),
                model=record, evaluation_role="external_frozen_transfer_control",
                mode_b_training_distribution_verified=False,
                a1_training_eval_seed_independence="unknown: bundle has no training seed list")


def load_runtime(root):
    root = Path(root).resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "r3p5_invariant" or name.startswith("r3p5_invariant."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root / "src"):
                raise ValueError("Another r3p5_invariant package is already imported.")
    sys.path.insert(0, str(root / "src"))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        runtime = importlib.import_module("r3p5_invariant.portable.runtime")
        common = importlib.import_module("r3p5_invariant.portable.common")
        metrics = importlib.import_module("r3p5_invariant.evaluation.point_metrics")
    finally:
        sys.dont_write_bytecode = previous
        sys.path.pop(0)
    return runtime, common, metrics


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def a1_inputs(batch, ordinals, policy):
    """Labels/truth never enter A1. The original batch/Ruu are left untouched."""
    if batch["metadata"]["csi"] != "practical":
        raise ValueError("A1 comparison requires practical CSI.")
    b, t, f, m = batch["y"].shape
    k = batch["h_hat"].shape[-1]
    idx = batch["data_indices"][ordinals.to(batch["data_indices"].device)]
    y = batch["y"].reshape(b, t * f, m)[:, idx].unsqueeze(1)
    h = batch["h_hat"].reshape(b, t * f, m, k)[:, idx].unsqueeze(1)
    if policy == "sionna_diagonal":
        error = batch["err_var"].reshape(b, t * f, m, k)[:, idx].unsqueeze(1)
        y, h = whiten_diagonal_uncertainty(y, h, error, batch["n0"])
        ruu = torch.eye(m, dtype=h.dtype, device=h.device).expand(b, -1, -1)
    elif policy == "thermal_only":
        # Explicit auxiliary receiver: CE error is handled only inside A1.
        ruu = batch["Ruu"]
    else:
        raise ValueError("Unknown A1 CE input policy.")
    return dict(y=y, h=h, ruu=ruu)


class PartnerA1Adapter:
    def __init__(self, root, device):
        from link_level.sionna_ebno import require_sionna2

        self.root, self.device = Path(root).resolve(), torch.device(device)
        self.metadata = verify_bundle(self.root)
        if require_sionna2() != "2.0.1":
            raise RuntimeError("This A1 bundle is qualified with Sionna 2.0.1; use the existing server environment.")
        self.runtime, self.common, self.metrics = load_runtime(self.root)
        self.common.numeric_setup(self.device)
        self.model, record = self.runtime.load_model(self.root, self.device)
        if record != self.metadata["model"]:
            raise ValueError("A1 metadata changed during load.")
        self.metadata["runtime"] = self.common.environment(self.device)

    @torch.inference_mode()
    def check_constellation(self, link):
        labels = self.model.table.bit_table.to(self.device).reshape(1, 1, -1)
        actual = link.codec.mapper(labels).reshape(-1)
        torch.testing.assert_close(actual, self.model.table.points, atol=2e-6, rtol=2e-6)

    @torch.inference_mode()
    def evaluate(self, batch, ordinals, policy, seed):
        synchronize(self.device)
        start = time.perf_counter()
        values = a1_inputs(batch, ordinals, policy)
        synchronize(self.device)
        frontend_seconds = time.perf_counter() - start
        # Original K256 sampler, with K64 as its exact RB prefix; no new sampler.
        predictions, timing = self.runtime.predict(self.model, values, self.device, seed)
        expected = (batch["y"].shape[0], 1, len(ordinals), 16, 4)
        for llr in predictions.values():
            if tuple(llr.shape) != expected or not torch.isfinite(llr).all():
                raise ValueError("Invalid A1/native baseline output.")
        timing.update(ce_adapter_seconds=frontend_seconds, input_policy=policy,
                      sampling_seed=seed, llr_convention="log(P1/P0)",
                      timing_note="K64/K256 share one inference; no separate K64 latency is measured")
        return {name: value[:, 0] for name, value in predictions.items()}, timing


def validate_neural_checkpoint(checkpoint, arch, cfg, covariance_sha256, evaluation_seeds):
    expected = dict(kind="paper_reference_detector_v1", arch=arch, bits_per_symbol=4,
                    mcs_table=1, mcs_index=10, csi="practical", custom_ce_policy="sionna_diagonal",
                    distribution_id=distribution_id(cfg), ce_covariance_sha256=covariance_sha256,
                    llr_convention="log(P1/P0)")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Legacy/mismatched {arch} checkpoint: {key}.")
    if distribution_id(checkpoint["system_config"]) != expected["distribution_id"]:
        raise ValueError("Checkpoint system_config does not match its distribution hash.")
    for key in ("train_seeds", "validation_seeds"):
        if not isinstance(checkpoint.get(key), list) or not checkpoint[key]:
            raise ValueError(f"Checkpoint is missing {key} provenance.")
        if set(evaluation_seeds).intersection(checkpoint[key]):
            raise ValueError(f"Evaluation seeds overlap checkpoint {key}.")
    if not isinstance(checkpoint.get("step"), int) or checkpoint["step"] < 0:
        raise ValueError("Missing/invalid checkpoint training step.")


def load_neural_checkpoint(path, arch, link, evaluation_seeds):
    from training.train_paper_reference_detector import make_model

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_neural_checkpoint(checkpoint, arch, link.cfg, link.covariance_metadata["sha256"], evaluation_seeds)
    model_cfg = dict(link.cfg, modulation={"bits_per_symbol": 4})
    model = make_model(model_cfg, arch).to(link.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    metadata = {key: value for key, value in checkpoint.items() if key not in ("model_state", "optimizer_state")}
    metadata.update(path=str(path), sha256=file_sha256(path),
                    evaluation_role="trained_Mode_B" if checkpoint["step"] else "untrained_EP_anchor")
    return model.eval().requires_grad_(False), metadata
