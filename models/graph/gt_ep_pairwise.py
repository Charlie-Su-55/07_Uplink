#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from sionna.phy.mapping import Constellation


class PairwiseGraphTransformerLayer(nn.Module):
    def __init__(self, d_model=128, num_heads=8, edge_dim=48, ffn_dim=256, dropout=0.05):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, num_heads),
        )

        self.edge_value = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, d_model),
        )

        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_state, cross_mask):
        b, k, _ = x.shape

        qkv = self.qkv(self.norm1(x)).view(b, k, 3, self.num_heads, self.head_dim)
        q, key, value = qkv.unbind(dim=2)

        logits = torch.einsum("bihd,bjhd->bhij", q, key) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge_state).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~cross_mask[:, None], -1e4)

        attn = torch.softmax(logits, dim=-1)

        edge_value = self.edge_value(edge_state).view(b, k, k, self.num_heads, self.head_dim)
        value_j = value[:, None, :, :, :]
        messages = value_j + edge_value

        weights = attn.permute(0, 2, 3, 1).unsqueeze(-1)
        aggregate = (weights * messages).sum(dim=2).reshape(b, k, self.d_model)

        x = x + self.dropout(self.out_proj(aggregate))
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class PairwiseSymbolGraphRefiner(nn.Module):
    def __init__(self, points, num_users=16, num_iterations=5, d_model=128, num_heads=8, num_layers=4, edge_dim=48, ffn_dim=256, dropout=0.05, max_logit_correction=4.0, pairwise_mode="full"):
        super().__init__()

        if pairwise_mode not in {"full", "no_pairwise", "shuffled"}:
            raise ValueError(f"Unknown pairwise_mode: {pairwise_mode}")

        self.num_users = int(num_users)
        self.num_symbols = int(points.numel())
        self.num_iterations = int(num_iterations)
        self.max_logit_correction = float(max_logit_correction)
        self.pairwise_mode = pairwise_mode

        if self.num_symbols != 16:
            raise ValueError("Pairwise GT-EP v2 is currently fixed to 16-QAM.")

        self.register_buffer("points", points.detach().clone())

        node_input_dim = self.num_symbols + self.num_symbols + 6
        edge_input_dim = self.num_symbols + 5

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_dim),
            nn.GELU(),
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, edge_dim),
        )

        self.iteration_embedding = nn.Embedding(self.num_iterations, d_model)

        self.layers = nn.ModuleList([
            PairwiseGraphTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                edge_dim=edge_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.logit_head = nn.Linear(d_model, self.num_symbols)

        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

        mask = ~torch.eye(self.num_users, dtype=torch.bool)
        self.register_buffer("cross_mask", mask[None])

    def build_pair_energy(self, gram):
        points_i = self.points.view(1, 1, 1, self.num_symbols, 1)
        points_j = self.points.view(1, 1, 1, 1, self.num_symbols)
        gij = gram[:, :, :, None, None]

        pair_energy = -2.0 * (points_i.conj() * gij * points_j).real

        return pair_energy

    def build_pairwise_messages(self, gram, base_log_prob, pair_energy):
        log_q = torch.log_softmax(base_log_prob, dim=-1)

        log_q_j = log_q[:, None, :, None, :]
        raw_message = torch.logsumexp(log_q_j + pair_energy, dim=-1)

        centered = raw_message - raw_message.mean(dim=-1, keepdim=True)

        rms = torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + 1e-8)

        # Large exact energies are compressed, while weak couplings are not artificially amplified.
        scaled_message = centered / torch.sqrt(1.0 + rms.square())

        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
        denom = torch.sqrt(diag[:, :, None] * diag[:, None, :]).clamp_min(1e-8)
        rho = gram / denom

        strength = torch.log1p(rms)

        gram_abs = torch.log1p(gram.abs()).unsqueeze(-1)

        scalar_edge = torch.cat([
            strength,
            rho.real.unsqueeze(-1),
            rho.imag.unsqueeze(-1),
            rho.abs().unsqueeze(-1),
            gram_abs,
        ], dim=-1)

        mask = self.cross_mask[..., None].to(scaled_message.dtype)

        scaled_message = scaled_message * mask
        scalar_edge = scalar_edge * mask

        if self.pairwise_mode == "no_pairwise":
            scaled_message = torch.zeros_like(scaled_message)
            scalar_edge = torch.zeros_like(scalar_edge)

        elif self.pairwise_mode == "shuffled":
            perm = torch.roll(torch.arange(self.num_users, device=gram.device), shifts=1)
            scaled_message = scaled_message[:, perm][:, :, perm]
            scalar_edge = scalar_edge[:, perm][:, :, perm]

        pairwise_aggregate = scaled_message.sum(dim=2) / float(self.num_users - 1)

        edge_features = torch.cat([
            scaled_message,
            scalar_edge,
        ], dim=-1)

        edge_state = self.edge_encoder(edge_features)

        pairwise_rms = torch.sqrt(
            pairwise_aggregate.square().mean()
            + 1e-8
        )

        return edge_state, pairwise_aggregate, pairwise_rms

    def forward(self, z, gram, cavity_mean, cavity_precision, base_log_prob, iteration, pair_energy):
        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)

        mf = z / diag.to(z.dtype)

        base_log_post = torch.log_softmax(base_log_prob, dim=-1).clamp_min(-20.0)

        edge_state, pairwise_aggregate, pairwise_rms = self.build_pairwise_messages(
            gram=gram,
            base_log_prob=base_log_prob,
            pair_energy=pair_energy,
        )

        scalar_features = torch.stack([
            mf.real,
            mf.imag,
            cavity_mean.real,
            cavity_mean.imag,
            torch.log(cavity_precision.clamp_min(1e-8)),
            torch.log(diag),
        ], dim=-1)

        node_features = torch.cat([
            base_log_post,
            pairwise_aggregate,
            scalar_features,
        ], dim=-1)

        x = self.node_encoder(node_features)

        iteration_index = torch.full(
            (x.shape[0], self.num_users),
            int(iteration),
            dtype=torch.long,
            device=x.device,
        )

        x = x + self.iteration_embedding(iteration_index)

        for layer in self.layers:
            x = layer(x, edge_state, self.cross_mask)

        raw_delta = self.logit_head(self.final_norm(x))

        delta_log_prob = self.max_logit_correction * torch.tanh(raw_delta)
        delta_log_prob = delta_log_prob - delta_log_prob.mean(dim=-1, keepdim=True)

        return delta_log_prob, pairwise_rms


class PairwiseGraphTransformerEPDetector(nn.Module):
    def __init__(self, cfg, num_users=16, num_iterations=5, damping=0.5, min_variance=1e-6, min_site_precision=1e-6, d_model=128, num_heads=8, num_layers=4, edge_dim=48, ffn_dim=256, dropout=0.05, max_logit_correction=4.0, pairwise_mode="full"):
        super().__init__()

        self.num_users = int(num_users)
        self.num_iterations = int(num_iterations)
        self.damping = float(damping)
        self.min_variance = float(min_variance)
        self.min_site_precision = float(min_site_precision)

        self.bits_per_symbol = int(cfg["modulation"]["bits_per_symbol"])

        if self.bits_per_symbol != 4:
            raise ValueError("Current Pairwise GT-EP implementation is fixed to 16-QAM.")

        constellation = Constellation(
            "qam",
            self.bits_per_symbol,
            precision=cfg["general"]["precision"],
            device=cfg["general"]["device"],
        )

        points = constellation.points.detach().clone()

        if points.numel() != 16:
            raise ValueError("Expected 16-QAM constellation with 16 points.")

        self.register_buffer("points", points)

        indices = torch.arange(points.numel(), dtype=torch.long, device=points.device)
        shifts = torch.arange(self.bits_per_symbol - 1, -1, -1, dtype=torch.long, device=points.device)
        bit_labels = ((indices[:, None] >> shifts[None, :]) & 1).to(torch.bool)

        self.register_buffer("bit_labels", bit_labels)

        self.graph_refiner = PairwiseSymbolGraphRefiner(
            points=points,
            num_users=self.num_users,
            num_iterations=self.num_iterations,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            edge_dim=edge_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_logit_correction=max_logit_correction,
            pairwise_mode=pairwise_mode,
        )

    def _gaussian_marginals(self, z, gram, site_precision, site_natural):
        precision = gram + torch.diag_embed(site_precision.to(gram.dtype))
        precision = 0.5 * (precision + precision.mH)

        chol = torch.linalg.cholesky(precision)
        eta = z + site_natural

        mean = torch.cholesky_solve(eta.unsqueeze(-1), chol).squeeze(-1)
        covariance = torch.cholesky_inverse(chol)

        variance = torch.diagonal(covariance, dim1=-2, dim2=-1).real.clamp_min(self.min_variance)

        return mean, variance

    def _base_log_prob(self, cavity_mean, cavity_precision):
        diff = cavity_mean[..., None] - self.points
        return -cavity_precision[..., None] * diff.abs().square()

    def _posterior_from_log_prob(self, log_prob):
        prob = torch.softmax(log_prob, dim=-1)

        post_mean = (prob * self.points).sum(dim=-1)
        second_moment = (prob * self.points.abs().square()).sum(dim=-1)

        post_variance = (second_moment - post_mean.abs().square()).real.clamp_min(self.min_variance)

        return post_mean, post_variance

    def _llr_from_log_prob(self, log_prob):
        llrs = []

        for q in range(self.bits_per_symbol):
            mask1 = self.bit_labels[:, q]
            mask0 = ~mask1

            log_p1 = torch.logsumexp(log_prob[..., mask1], dim=-1)
            log_p0 = torch.logsumexp(log_prob[..., mask0], dim=-1)

            llrs.append(log_p1 - log_p0)

        return torch.stack(llrs, dim=-1)

    def forward(self, z, gram, return_iterations=(5,)):
        if z.ndim != 2:
            raise ValueError(f"Expected z [B,K], got {tuple(z.shape)}")

        if gram.ndim != 3:
            raise ValueError(f"Expected Gram [B,K,K], got {tuple(gram.shape)}")

        if z.shape[-1] != self.num_users:
            raise ValueError(f"Expected {self.num_users} users, got {z.shape[-1]}")

        if gram.shape[-2:] != (self.num_users, self.num_users):
            raise ValueError(f"Expected Gram [B,{self.num_users},{self.num_users}], got {tuple(gram.shape)}")

        gram = 0.5 * (gram + gram.mH)

        real_dtype = gram.real.dtype

        site_precision = torch.ones(z.shape, dtype=real_dtype, device=z.device)
        site_natural = torch.zeros_like(z)

        requested = set(int(i) for i in return_iterations)
        outputs = {}

        validity = []
        correction_rms = []
        pairwise_message_rms = []

        # G is fixed during the five unfolded EP iterations, so exact 16x16
        # symbol-pair energies are computed only once.
        pair_energy = self.graph_refiner.build_pair_energy(gram)

        for iteration in range(1, self.num_iterations + 1):
            mean, variance = self._gaussian_marginals(
                z,
                gram,
                site_precision,
                site_natural,
            )

            marginal_precision = 1.0 / variance

            cavity_precision = (
                marginal_precision
                - site_precision
            ).clamp_min(self.min_site_precision)

            cavity_natural = (
                mean / variance.to(mean.dtype)
                - site_natural
            )

            cavity_mean = (
                cavity_natural
                / cavity_precision.to(cavity_natural.dtype)
            )

            base_log_prob = self._base_log_prob(
                cavity_mean,
                cavity_precision,
            )

            delta_log_prob, pair_rms = self.graph_refiner(
                z=z,
                gram=gram,
                cavity_mean=cavity_mean,
                cavity_precision=cavity_precision,
                base_log_prob=base_log_prob,
                iteration=iteration - 1,
                pair_energy=pair_energy,
            )

            refined_log_prob = (
                base_log_prob
                + delta_log_prob
            )

            post_mean, post_variance = self._posterior_from_log_prob(
                refined_log_prob
            )

            llr = self._llr_from_log_prob(
                refined_log_prob
            )

            correction_rms.append(
                delta_log_prob.square().mean().sqrt().detach()
            )

            pairwise_message_rms.append(
                pair_rms.detach()
            )

            if iteration in requested:
                outputs[iteration] = {
                    "llr": llr,
                    "x_hat": post_mean,
                    "posterior_variance": post_variance,
                    "delta_log_prob": delta_log_prob,
                }

            candidate_precision = (
                1.0 / post_variance
                - cavity_precision
            )

            candidate_natural = (
                post_mean / post_variance.to(post_mean.dtype)
                - cavity_natural
            )

            valid = (
                torch.isfinite(candidate_precision)
                & torch.isfinite(candidate_natural.real)
                & torch.isfinite(candidate_natural.imag)
                & (candidate_precision > self.min_site_precision)
            )

            validity.append(
                valid.float().mean().detach()
            )

            candidate_precision = torch.where(
                valid,
                candidate_precision,
                site_precision,
            )

            candidate_natural = torch.where(
                valid,
                candidate_natural,
                site_natural,
            )

            site_precision = (
                (1.0 - self.damping) * site_precision
                + self.damping * candidate_precision
            ).clamp_min(self.min_site_precision)

            site_natural = (
                (1.0 - self.damping) * site_natural
                + self.damping * candidate_natural
            )

        if self.num_iterations not in outputs:
            outputs[self.num_iterations] = {
                "llr": llr,
                "x_hat": post_mean,
                "posterior_variance": post_variance,
                "delta_log_prob": delta_log_prob,
            }

        return {
            "llr": outputs[self.num_iterations]["llr"],
            "x_hat": outputs[self.num_iterations]["x_hat"],
            "posterior_variance": outputs[self.num_iterations]["posterior_variance"],
            "iterations": outputs,
            "valid_update_fraction": torch.stack(validity),
            "correction_rms": torch.stack(correction_rms),
            "pairwise_message_rms": torch.stack(pairwise_message_rms),
            "site_precision": site_precision,
            "site_natural": site_natural,
        }