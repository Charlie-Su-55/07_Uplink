"""Small shared positive LLR scales from cached, non-oracle receiver features."""

import math

import torch
from torch import nn


CANDIDATES = ("affine", "mlp", "graph", "mlp_no_residual_eta")
FEATURE_NAMES = ("mean_real", "mean_imag", "log_variance", "log_gram_diagonal",
                 "log_residual", "eta")


class MessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * hidden + 3, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())

    def forward(self, nodes, edges):
        users = nodes.shape[-2]
        target = nodes.unsqueeze(-2).expand(*nodes.shape[:-2], users, users, nodes.shape[-1])
        source = nodes.unsqueeze(-3).expand_as(target)
        messages = self.message(torch.cat((target, source, edges), dim=-1))
        mask = ~torch.eye(users, device=nodes.device, dtype=torch.bool)
        pooled = (messages * mask[..., None]).sum(-2) / max(users - 1, 1)
        return self.update(torch.cat((nodes, pooled), dim=-1))


class ConditionalScale(nn.Module):
    def __init__(self, candidate, *, alpha_min=0.05, alpha_max=8.0, hidden=32):
        super().__init__()
        if candidate not in CANDIDATES:
            raise ValueError(f"Unknown conditional scale candidate: {candidate}")
        if not 0 < alpha_min < 1 < alpha_max or hidden < 1:
            raise ValueError("Positive scale bounds must contain one; hidden width must be positive.")
        self.candidate = candidate
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        self.log_bounds = (math.log(self.alpha_min), math.log(self.alpha_max))
        if candidate == "affine":
            self.encoder = nn.Identity()
            self.head = nn.Linear(2, 4)
        elif candidate == "graph":
            self.encoder = nn.Sequential(nn.Linear(6, hidden), nn.SiLU())
            self.message_layers = nn.ModuleList((MessageLayer(hidden), MessageLayer(hidden)))
            self.head = nn.Linear(hidden, 4)
        else:
            features = 4 if candidate == "mlp_no_residual_eta" else 6
            self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(),
                                         nn.Linear(hidden, hidden), nn.SiLU())
            self.head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, nodes, edges):
        if nodes.ndim != 3 or nodes.shape[-1] != 6 or nodes.dtype != torch.float32:
            raise ValueError("Expected float32 nodes [RE,UE,6].")
        if not torch.isfinite(nodes).all():
            raise ValueError("Non-finite inference features.")
        if self.candidate == "affine":
            encoded = nodes[..., 4:6]
        elif self.candidate == "graph":
            if edges.shape != (*nodes.shape[:2], nodes.shape[1], 3) or edges.dtype != nodes.dtype:
                raise ValueError("Expected normalized Gram edges [RE,UE,UE,3].")
            if not torch.isfinite(edges).all():
                raise ValueError("Non-finite Gram edges.")
            encoded = self.encoder(nodes)
            for layer in self.message_layers:
                encoded = layer(encoded, edges)
        else:
            encoded = self.encoder(nodes[..., :4] if self.candidate == "mlp_no_residual_eta" else nodes)
        log_scale = self.head(encoded).clamp(*self.log_bounds)
        return log_scale.exp().clamp(self.alpha_min, self.alpha_max)


def make_model(candidate, config):
    settings = config["model"]
    return ConditionalScale(candidate, alpha_min=settings["alpha_min"],
                            alpha_max=settings["alpha_max"], hidden=config["model"]["hidden"])


def scale_llr(llr, alpha):
    if llr.shape != alpha.shape or llr.dtype != torch.float32 or alpha.dtype != llr.dtype:
        raise ValueError("Conditional scales must match float32 [RE,UE,Qm] LLRs.")
    if not torch.isfinite(llr).all() or not torch.isfinite(alpha).all() or (alpha <= 0).any():
        raise ValueError("Require finite LLRs and finite positive scales.")
    scaled = llr * alpha
    if not torch.isfinite(scaled).all() or not torch.equal(scaled > 0, llr > 0):
        raise ValueError("Scaling changed hard decisions or produced non-finite values.")
    return scaled


def bound_statistics(alpha, alpha_min=0.05, alpha_max=8.0, tolerance=0.0):
    if not torch.isfinite(alpha).all() or (alpha < alpha_min).any() or (alpha > alpha_max).any():
        raise ValueError("Conditional scale outside predeclared positive bounds.")
    return dict(min=float(alpha.min()), max=float(alpha.max()),
                lower_count=int((alpha <= alpha_min).sum()), upper_count=int((alpha >= alpha_max).sum()),
                near_lower_count=int((alpha <= alpha_min + tolerance).sum()),
                near_upper_count=int((alpha >= alpha_max - tolerance).sum()),
                saturation_tolerance=tolerance, count=alpha.numel())
