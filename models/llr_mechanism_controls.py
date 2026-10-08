"""Matched-capacity feature masks and train-only noise-conditioned LLR scales."""

import math

import torch
from torch import nn

from models.conditional_llr_scaling import ConditionalScale


CANDIDATES = ("full_mlp", "mlp_without_residual", "mlp_without_eta", "mlp_without_both", "affine_all_features")
MASKED_FEATURES = {"full_mlp": (), "mlp_without_residual": (4,), "mlp_without_eta": (5,),
                   "mlp_without_both": (4, 5), "affine_all_features": ()}


class MechanismControl(ConditionalScale):
    def __init__(self, candidate, *, alpha_min=0.05, alpha_max=8.0, hidden=32):
        if candidate not in CANDIDATES:
            raise ValueError(f"Unknown mechanism control: {candidate}")
        super().__init__("mlp", alpha_min=alpha_min, alpha_max=alpha_max, hidden=hidden)
        self.control_candidate = candidate
        mask = torch.ones(6, dtype=torch.float32)
        for index in MASKED_FEATURES[candidate]:
            mask[index] = 0
        self.register_buffer("feature_mask", mask)
        if candidate == "affine_all_features":
            self.encoder = nn.Identity()
            self.head = nn.Linear(6, 4)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, nodes, edges=None):
        if nodes.ndim != 3 or nodes.shape[-1] != 6:
            raise ValueError("Mask only standardized nodes [RE,UE,6].")
        return super().forward(nodes * self.feature_mask, edges)


def make_model(candidate, config):
    settings = config["model"]
    return MechanismControl(candidate, alpha_min=settings["alpha_min"], alpha_max=settings["alpha_max"],
                            hidden=settings["hidden"])


def interpolate_noise_alpha(n0, fitted):
    """Interpolate log alpha versus actual log N0; hold the two endpoint values."""
    value = float(n0)
    lower, upper = float(fitted["alpha_min"]), float(fitted["alpha_max"])
    if (not math.isfinite(value) or value <= 0 or not 0 < lower < 1 < upper
            or fitted.get("interpolation") != "linear_log_alpha_log_n0_endpoint_hold"):
        raise ValueError("Invalid fixed N0 interpolation contract.")
    knots = fitted["knots"]
    noises = [float(knot["n0"]) for knot in knots]
    if (len(knots) != 3 or any(not math.isfinite(noise) or noise <= 0 for noise in noises)
            or any(second <= first for first, second in zip(noises, noises[1:]))):
        raise ValueError("Expected three strictly increasing positive training N0 knots.")
    alphas = torch.tensor([knot["alpha"] for knot in knots], dtype=torch.float64)
    if alphas.shape != (3, 4) or not torch.isfinite(alphas).all() or (alphas < lower).any() or (alphas > upper).any():
        raise ValueError("Invalid bounded per-bit training knot scales.")
    if value <= noises[0]:
        alpha = alphas[0]
    elif value >= noises[-1]:
        alpha = alphas[-1]
    else:
        right = next(index for index, noise in enumerate(noises) if noise >= value)
        if value == noises[right]:
            alpha = alphas[right]
        else:
            fraction = (math.log(value) - math.log(noises[right - 1])) / (math.log(noises[right]) - math.log(noises[right - 1]))
            alpha = ((1 - fraction) * alphas[right - 1].log() + fraction * alphas[right].log()).exp()
    return alpha.to(torch.float32).clamp(lower, upper)
