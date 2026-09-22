"""Discrete masked posterior flow control, adapted from the partner Flow package.

The partner's order-agnostic MASK -> QAM process and Gram-edge attention are
retained. This implementation consumes the same whitened z/G as GT/DETR, uses
APP LMMSE anchoring, and streams conditional marginals instead of storing K
complete candidate sets on the CPU. Partner checkpoints are not compatible.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def square_qam(bits_per_symbol):
    """Normalized recursive Gray PAM, with natural-binary row labels (Sionna)."""
    if bits_per_symbol not in (2, 4, 6):
        raise ValueError("bits_per_symbol must be 2, 4 or 6")
    labels = ((torch.arange(2 ** bits_per_symbol)[:, None]
               >> torch.arange(bits_per_symbol - 1, -1, -1)) & 1).float()

    def pam(bits):
        sign = 1 - 2 * bits[..., 0]
        return sign if bits.shape[-1] == 1 else sign * (2 ** (bits.shape[-1] - 1) - pam(bits[..., 1:]))

    points = torch.complex(pam(labels[:, 0::2]), pam(labels[:, 1::2]))
    return points / points.abs().square().mean().sqrt(), labels


class GramAttentionBlock(nn.Module):
    """Partner-style stream attention with physical edge bias and gating."""
    def __init__(self, width, heads, ffn_dim):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_bias = nn.Linear(4, heads)
        self.edge_gate = nn.Linear(4, heads)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ffn_dim),
                                 nn.SiLU(), nn.Linear(ffn_dim, width))

    def forward(self, x, edge):
        n, s, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(n, s, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge).tanh().permute(0, 3, 1, 2)
        weights = scores.softmax(-1) * self.edge_gate(edge).sigmoid().permute(0, 3, 1, 2)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        x = x + self.out((weights @ v).transpose(1, 2).reshape(n, s, width))
        return x + self.ffn(x)


class FlowMatchingDetector(nn.Module):
    """z[...,S], G[...,S,S] -> dict(llr[...,S,Qm]), positive LLR = bit one.

    z=H^H Ruu^-1 y and G=H^H Ruu^-1 H already include noise whitening.
    Only active streams are accepted; slice padded streams before calling.
    return_iterations is accepted for the common evaluator, not an EP schedule.
    """
    implementation_id = "uplink-masked-flow-v1"
    CONTEXT_FEATURES = 10
    DYNAMIC_FEATURES = 9

    def __init__(self, cfg, num_users=None, d_model=128, num_heads=8,
                 num_layers=3, ffn_dim=256, num_samples=64, sample_chunk=16,
                 re_chunk=16, sampling_seed=1729, max_llr_correction=20.0):
        super().__init__()
        self.bits_per_symbol = int(cfg["modulation"]["bits_per_symbol"])
        general = cfg.get("general", {})
        self.num_users = int(num_users if num_users is not None else
                             general.get("num_ues", 16) * general.get("streams_per_ue", 1))
        for name, value in dict(num_users=self.num_users, d_model=d_model, num_heads=num_heads,
                                num_layers=num_layers, ffn_dim=ffn_dim, num_samples=num_samples,
                                sample_chunk=sample_chunk, re_chunk=re_chunk).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads or not math.isfinite(max_llr_correction) or max_llr_correction <= 0:
            raise ValueError("Invalid attention dimensions or max_llr_correction")
        self.num_samples, self.sample_chunk, self.re_chunk = num_samples, sample_chunk, re_chunk
        self.sampling_seed = int(sampling_seed)
        self.max_llr_correction = float(max_llr_correction)
        self.model_config = dict(num_users=self.num_users, d_model=d_model, num_heads=num_heads,
                                 num_layers=num_layers, ffn_dim=ffn_dim, num_samples=num_samples,
                                 sample_chunk=sample_chunk, re_chunk=re_chunk,
                                 sampling_seed=self.sampling_seed, max_llr_correction=max_llr_correction)
        points, labels = square_qam(self.bits_per_symbol)
        self.register_buffer("points", points)
        self.register_buffer("bit_table", labels)
        self.mask_token = len(points)
        self.state_embedding = nn.Embedding(self.mask_token + 1, 32)
        self.context_in = nn.Sequential(nn.Linear(self.CONTEXT_FEATURES, d_model), nn.SiLU())
        self.state_in = nn.Linear(32 + self.DYNAMIC_FEATURES, d_model)
        self.blocks = nn.ModuleList(GramAttentionBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers))
        self.proposal_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.mask_token))
        self.physical_gate = nn.Linear(d_model, 1)
        nn.init.normal_(self.proposal_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.proposal_head[-1].bias)
        nn.init.zeros_(self.physical_gate.weight)
        nn.init.constant_(self.physical_gate.bias, math.log(0.15 / 0.85))
        self.llr_head = nn.Sequential(
            nn.Linear(self.CONTEXT_FEATURES + 3 * self.bits_per_symbol + 1, d_model),
            nn.SiLU(), nn.LayerNorm(d_model), nn.Linear(d_model, self.bits_per_symbol))
        nn.init.zeros_(self.llr_head[-1].weight)
        nn.init.zeros_(self.llr_head[-1].bias)

        # Fixed randomized quadrature shared across REs: independent of execution
        # chunks and the simulator's global RNG. Persisted with the checkpoint.
        draws = torch.quasirandom.SobolEngine(2 * self.num_users, scramble=True,
                                              seed=self.sampling_seed).draw(num_samples)
        self.register_buffer("sampling_draws", draws)

    def _flatten(self, z, gram):
        if z.ndim < 2 or not z.is_complex() or not gram.is_complex():
            raise ValueError("Expected complex z[...,S] and gram[...,S,S], including an RE axis")
        s = z.shape[-1]
        if not 1 <= s <= self.num_users or gram.shape != (*z.shape, s):
            raise ValueError(f"Incompatible z/G shapes: {tuple(z.shape)}, {tuple(gram.shape)}")
        if z.numel() == 0 or z.device != gram.device or z.device != self.points.device:
            raise ValueError("Inputs must be nonempty and on the model device")
        if not torch.isfinite(z).all() or not torch.isfinite(gram).all():
            raise ValueError("z/G contain NaN or Inf")
        return z.reshape(-1, s).to(torch.complex64), gram.reshape(-1, s, s).to(torch.complex64)

    def _llr(self, logits):
        return torch.stack([
            torch.logsumexp(logits[..., self.bit_table[:, b].bool()], -1)
            - torch.logsumexp(logits[..., ~self.bit_table[:, b].bool()], -1)
            for b in range(self.bits_per_symbol)], -1)

    def _context(self, z, gram):
        # Ruu already contains thermal noise; the whitened regularizer is I.
        gram = (gram + gram.mH) * 0.5
        s = z.shape[-1]
        eye = torch.eye(s, device=z.device, dtype=gram.dtype)
        chol = torch.linalg.cholesky(gram + eye)
        biased = torch.cholesky_solve(z[..., None], chol).squeeze(-1)
        effective = torch.cholesky_solve(gram, chol)
        gain = effective.diagonal(dim1=-2, dim2=-1).real.clamp(1e-7, 1 - 1e-7)
        symbols = biased / gain
        variance = gain.reciprocal() - 1
        logits = -(symbols[..., None] - self.points).abs().square() / variance[..., None]
        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
        scale = diag.sqrt()
        rho = gram / (scale[..., :, None] * scale[..., None, :])
        off = rho.abs() * (1 - eye.real)
        residual = (z - (gram @ biased[..., None]).squeeze(-1)) / scale
        node = torch.stack((symbols.real, symbols.imag, variance.log(), diag.log(),
                            residual.real, residual.imag, off.sum(-1), off.amax(-1),
                            gain, torch.log1p(z.abs().square() / diag)), -1).float()
        edge = torch.stack((rho.real, rho.imag, rho.abs(), off), -1).float()
        return dict(z=z, gram=gram, diag=diag, scale=scale, symbols=symbols,
                    variance=variance, posterior_variance=1 - gain, biased=biased,
                    node=node, edge=edge, encoded=self.context_in(node), lmmse_llr=self._llr(logits))

    def _proposal(self, state, context):
        known = state != self.mask_token
        current = self.points[state.clamp_max(self.mask_token - 1)]
        completion = torch.where(known, current, context["biased"])
        residual = (context["z"] - (context["gram"] @ completion[..., None]).squeeze(-1)) / context["scale"]
        fraction = known.float().mean(-1, keepdim=True).expand_as(known)
        dynamic = torch.stack((completion.real, completion.imag, residual.real, residual.imag,
                               known.float(), fraction, torch.sin(math.pi * fraction),
                               torch.cos(math.pi * fraction), residual.abs()), -1)
        x = context["encoded"] + self.state_in(torch.cat((self.state_embedding(state), dynamic), -1))
        for block in self.blocks:
            x = block(x, context["edge"])
        coupled = (context["gram"] @ completion[..., None]).squeeze(-1) - context["diag"] * completion
        matched = context["z"] - coupled
        energy = (context["diag"][..., None] * self.points.abs().square()
                  - 2 * (matched[..., None] * self.points.conj()).real)
        # Marginalize unresolved-stream uncertainty approximately instead of
        # treating every LMMSE-filled prefix entry as an exact transmitted symbol.
        unknown_var = context["posterior_variance"] * (~known)
        off_power = context["gram"].abs().square() - torch.diag_embed(context["diag"].square())
        uncertainty = (off_power.clamp_min(0) @ unknown_var[..., None]).squeeze(-1)
        noise_scale = 1 + uncertainty / context["diag"]
        physical = -energy / noise_scale[..., None]
        physical = (physical - physical.amax(-1, keepdim=True)).clamp_min(-60)
        return self.proposal_head(x) + self.physical_gate(x).sigmoid() * physical

    @torch.no_grad()
    def _marginals(self, context):
        """Rao-Blackwell mean/variance; O(RE_chunk * K_chunk), no CPU transfers."""
        n, s = context["z"].shape
        q = self.bits_per_symbol
        total = context["z"].real.new_zeros(n, s, q)
        total_square = torch.zeros_like(total)
        entropy = total[..., 0].clone()
        for start in range(0, self.num_samples, self.sample_chunk):
            draws = self.sampling_draws[start:start + self.sample_chunk]
            count = len(draws)
            order = draws[:, :s].argsort(-1)
            uniforms = draws[:, self.num_users:self.num_users + s]
            expanded = {key: value[:, None].expand(n, count, *value.shape[1:]).reshape(n * count, *value.shape[1:])
                        for key, value in context.items()}
            state = torch.full((n * count, s), self.mask_token, device=total.device, dtype=torch.long)
            rows = torch.arange(n * count, device=total.device)
            probability = total.new_zeros(n * count, s, q)
            local_entropy = total.new_zeros(n * count, s)
            for depth in range(s):
                selected = order[:, depth].repeat(n)
                logits = self._proposal(state, expanded)[rows, selected]
                probs = logits.softmax(-1)
                probability[rows, selected] = probs @ self.bit_table
                local_entropy[rows, selected] = -(probs * probs.clamp_min(1e-30).log()).sum(-1) / math.log(self.mask_token)
                u = uniforms[:, depth].repeat(n)
                tokens = (u[:, None] > probs.cumsum(-1)).sum(-1).clamp_max(self.mask_token - 1)
                state[rows, selected] = tokens
            probability = probability.reshape(n, count, s, q)
            total += probability.sum(1)
            total_square += probability.square().sum(1)
            entropy += local_entropy.reshape(n, count, s).sum(1)
        mean = total / self.num_samples
        std = (total_square / self.num_samples - mean.square()).clamp_min(0).sqrt()
        # No hard-count zero probabilities, even at small K.
        mean = mean.clamp(1e-7, 1 - 1e-7)
        return torch.logit(mean), std, entropy / self.num_samples

    def _detect_context(self, context):
        flow_llr, std, entropy = self._marginals(context)
        base = context["lmmse_llr"]
        features = torch.cat((context["node"], base.clamp(-20, 20) / 10,
                              flow_llr / 10, std, entropy[..., None]), -1)
        correction = self.max_llr_correction * self.llr_head(features).tanh()
        return {"llr": base + correction, "lmmse_llr": base, "raw_flow_llr": flow_llr}

    def forward(self, z, gram, return_iterations=None):
        leading = z.shape
        z, gram = self._flatten(z, gram)
        parts = [self._detect_context(self._context(z[i:i + self.re_chunk], gram[i:i + self.re_chunk]))
                 for i in range(0, len(z), self.re_chunk)]
        return {key: torch.cat([part[key] for part in parts]).reshape(*leading, self.bits_per_symbol)
                for key in parts[0]}

    def training_loss(self, z, gram, bits, proposal_weight=1.0):
        """Uniform random prefix depth + symbol CE; sampled final LLR + bit BCE.

        Discrete trajectories are detached. CE trains the flow proposal and BCE
        trains the zero-initialized soft residual head on deployment inputs.
        """
        if bits.shape != (*z.shape, self.bits_per_symbol):
            raise ValueError("bits must have shape [...,S,Qm] aligned with z")
        if bits.device != z.device or not ((bits == 0) | (bits == 1)).all():
            raise ValueError("bits must be binary and on the input device")
        if not math.isfinite(proposal_weight) or proposal_weight <= 0:
            raise ValueError("proposal_weight must be positive and finite")
        z, gram = self._flatten(z, gram)
        bits = bits.reshape(*z.shape, self.bits_per_symbol).float()
        powers = 2 ** torch.arange(self.bits_per_symbol - 1, -1, -1, device=z.device)
        truth = (bits.long() * powers).sum(-1)
        context = self._context(z, gram)
        n, s = z.shape
        order = torch.rand(n, s, device=z.device).argsort(-1)
        depth = torch.randint(s, (n,), device=z.device)
        ranks = order.argsort(-1)
        state = torch.where(ranks < depth[:, None], truth, self.mask_token)
        selected = order.gather(-1, depth[:, None]).squeeze(-1)
        rows = torch.arange(n, device=z.device)
        ce = F.cross_entropy(self._proposal(state, context)[rows, selected], truth[rows, selected])
        # Forward uses the same bounded RE and trajectory tiles as deployment.
        out = self(z, gram)
        bce = F.binary_cross_entropy_with_logits(out["llr"], bits)
        return {"loss": bce + float(proposal_weight) * ce, "bce": bce,
                "proposal_ce": ce, "llr": out["llr"]}

    def detect(self, y, h, ruu):
        """Optional raw-grid adapter: [B,T,F,Nr], [B,T,F,Nr,S], [B,Nr,Nr].

        Supply practical h_hat_lmmse and ruu_hat; select data OFDM symbols first.
        """
        from data.preprocessing.whitening import CovarianceAwareFrontEnd
        if ruu.shape != (y.shape[0], y.shape[-1], y.shape[-1]):
            raise ValueError("Ruu must be [B,Nr,Nr] aligned with y")
        stats = CovarianceAwareFrontEnd()(y, h, ruu)
        return self(stats["z"], stats["gram"])


def flow_from_checkpoint(cfg, checkpoint):
    """Strict architecture/receiver contract for this new control's checkpoints."""
    expected = {"arch": "flow_matching", "implementation_id": FlowMatchingDetector.implementation_id,
                "bits_per_symbol": int(cfg["modulation"]["bits_per_symbol"]),
                "csi": "lmmseH", "covariance": "estimated_Ruu"}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Flow checkpoint {key}={checkpoint.get(key)!r}, expected {value!r}")
    if not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("Flow checkpoint requires model_config")
    # Preserve the simulator RNG when adding this optional control.
    with torch.random.fork_rng(devices=[]):
        model = FlowMatchingDetector(cfg, **checkpoint["model_config"])
    general = cfg.get("general", {})
    streams = int(general.get("num_ues", model.num_users)) * int(general.get("streams_per_ue", 1))
    if model.num_users != streams:
        raise ValueError("Flow checkpoint stream capacity does not match the configured system")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model
