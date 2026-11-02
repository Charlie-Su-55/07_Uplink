"""Thin, hash-checked adapter for the external frozen R3P5 reference6000 bundle."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data/cache/external_bundles/R3P5_REFERENCE6000_889cf4abd6ed_b6a01b52/R3P5_REFERENCE6000"
ARCHIVE_SHA256 = "f7ef4e6d5fc74a1ae7d631ce4f4c7e4e5d9c1bb3e7c32c3e1966b41a629bcf98"


class R3P5Reference6000Adapter:
    """Preserves native sampling, correction, K64 readout, and positive LLR convention."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, device: str | torch.device = "cuda:0"):
        self.root = Path(root).resolve()
        self.device = torch.device(device)
        self.archive_sha256 = ARCHIVE_SHA256
        self.manifest = json.loads((self.root / "BUNDLE_MANIFEST.json").read_text())
        for relative, expected in self.manifest.items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Unsafe or missing bundle member: {relative}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Bundle member hash mismatch: {relative}")
        self.metadata = json.loads((self.root / "MODEL.json").read_text())
        if self.metadata.get("format") != "reference6000_partner_v1":
            raise ValueError("Unexpected external model identity")
        # The package's public loader verifies all files and prevents importing
        # a conflicting same-name package from another source tree.
        if "r3p5_invariant" in sys.modules:
            raise ImportError("R3P5 requires a fresh Python process to avoid module identity collision")
        spec = importlib.util.spec_from_file_location("_uplink_r3p5_receiver", self.root / "r3p5_receiver.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load bundled public receiver")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.receiver = module.Receiver(device=str(self.device))

    def check_constellation(self, points: torch.Tensor, labels: torch.Tensor) -> None:
        self.receiver.check_constellation(points, labels)

    @torch.inference_mode()
    def predict(self, y: torch.Tensor, hhat: torch.Tensor, err_var: torch.Tensor,
                n0: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, dict]:
        """Convert only axes/device/dtype and documented diagonal-CE scale.

        Inputs are Mode-B data REs in [B,T,F,...] order. CE uncertainty is
        represented by the same diagonal approximation as the native equalizer:
        v=N0+sum_UE(err_var). Ruu presented to the model is I after joint scaling.
        """
        if y.ndim != 4 or hhat.ndim != 5 or hhat.shape[:-1] != y.shape:
            raise ValueError("Expected y[B,T,F,Rx], Hhat[B,T,F,Rx,S]")
        if y.shape[-1] != 256 or not 1 <= hhat.shape[-1] <= 16 or y.shape[1] * y.shape[2] != 2304:
            raise ValueError("Mode-B external receiver requires Rx=256, 1..16 streams, and all 2304 data REs")
        if err_var.shape != hhat.shape:
            raise ValueError("err_var must match Hhat exactly")
        variance = n0.reshape(-1, 1, 1, 1) + err_var.sum(-1, keepdim=False)
        if variance.shape != y.shape or not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
            raise ValueError("Invalid shared thermal-plus-CE diagonal variance")
        scale = variance.sqrt()
        yw = (y / scale).to(device=self.device, dtype=torch.complex64).contiguous()
        hw = (hhat / scale.unsqueeze(-1)).to(device=self.device, dtype=torch.complex64).contiguous()
        ruu = torch.eye(256, dtype=torch.complex64, device=self.device).expand(y.shape[0], -1, -1).contiguous()
        llr = self.receiver(yw, hw, ruu, seed=int(seed))
        expected = (*y.shape[:-1], hhat.shape[-1], 4)
        if llr.shape != expected or llr.dtype != torch.float32 or not bool(torch.isfinite(llr).all()):
            raise ValueError("External receiver returned an invalid LLR tensor")
        return llr, {"native_seed": int(seed), "shape": list(llr.shape), "dtype": str(llr.dtype),
                     "scale": "sqrt(N0+sum(err_var)); Ruu=I", "native_audit": self.receiver.last_audit}
