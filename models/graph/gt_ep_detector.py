import math

import torch
import torch.nn as nn
from sionna.phy.mapping import Constellation


class InterferenceGraphTransformerLayer(nn.Module):
    def __init__(self, d_model=128, num_heads=8, edge_dim=32, ffn_dim=256, dropout=0.05, use_uncertainty_gate=False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_uncertainty_gate = bool(use_uncertainty_gate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_bias = nn.Sequential(nn.Linear(edge_dim, edge_dim), nn.GELU(), nn.Linear(edge_dim, num_heads))
        self.edge_value = nn.Sequential(nn.Linear(edge_dim, edge_dim), nn.GELU(), nn.Linear(edge_dim, d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        if self.use_uncertainty_gate:
            # beta = exp(log_beta), initialized at beta=1.
            self.log_gate_beta = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_gate_beta", None)

    def gate_beta(self):
        if not self.use_uncertainty_gate:
            return None
        return torch.exp(self.log_gate_beta.clamp(-3.0, 3.0))

    def forward(self, x, edge_state, cross_mask, edge_reliability=None):
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

        if self.use_uncertainty_gate:
            if edge_reliability is None:
                raise ValueError("edge_reliability is required when use_uncertainty_gate=True.")
            beta = self.gate_beta()
            gate = edge_reliability.clamp(1e-4, 1.0).pow(beta)
            messages = messages * gate[..., None, None].to(messages.dtype)

        weights = attn.permute(0, 2, 3, 1).unsqueeze(-1)
        aggregate = (weights * messages).sum(dim=2).reshape(b, k, self.d_model)

        x = x + self.dropout(self.out_proj(aggregate))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class InterferenceGraphLogitRefiner(nn.Module):
    def __init__(self, num_users=16, num_symbols=16, num_iterations=5, d_model=128, num_heads=8, num_layers=4, edge_dim=32, ffn_dim=256, dropout=0.05, max_logit_correction=4.0, edge_mode="full", message_mode="cross_user", use_ce_uncertainty=False):
        super().__init__()

        self.num_users = int(num_users)
        self.num_symbols = int(num_symbols)
        self.num_iterations = int(num_iterations)
        self.max_logit_correction = float(max_logit_correction)
        self.use_ce_uncertainty = bool(use_ce_uncertainty)

        if edge_mode not in {"full", "no_edge", "shuffled"}:
            raise ValueError(f"Unknown edge_mode: {edge_mode}")
        if message_mode not in {"cross_user", "self_only"}:
            raise ValueError(f"Unknown message_mode: {message_mode}")

        self.edge_mode = edge_mode
        self.message_mode = message_mode

        # UA-GT adds:
        # 1) global mean log uncertainty
        # 2) stream-relative log uncertainty
        node_input_dim = self.num_symbols + 6 + (2 if self.use_ce_uncertainty else 0)
        edge_input_dim = 6

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

        self.iteration_embedding = nn.Embedding(self.num_iterations, d_model)

        self.layers = nn.ModuleList([
            InterferenceGraphTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                edge_dim=edge_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_uncertainty_gate=self.use_ce_uncertainty,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.logit_head = nn.Linear(d_model, self.num_symbols)

        # Preserve exact EP5 initialization.
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

        mask = ~torch.eye(self.num_users, dtype=torch.bool)
        self.register_buffer("cross_mask", mask[None])

    def build_edge_features(self, gram):
        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
        denom = torch.sqrt(diag[:, :, None] * diag[:, None, :]).clamp_min(1e-8)
        rho = gram / denom

        directional = gram / diag[:, :, None].to(gram.dtype)

        edge_features = torch.stack([
            rho.real,
            rho.imag,
            rho.abs(),
            directional.real,
            directional.imag,
            torch.log1p(directional.abs()),
        ], dim=-1)

        edge_features = edge_features * self.cross_mask[..., None].to(edge_features.dtype)

        if self.edge_mode == "no_edge":
            edge_features = torch.zeros_like(edge_features)
        elif self.edge_mode == "shuffled":
            perm = torch.roll(torch.arange(self.num_users, device=gram.device), shifts=1)
            edge_features = edge_features[:, perm][:, :, perm]

        return self.edge_encoder(edge_features)

    def build_uncertainty_context(self, ce_uncertainty):
        if ce_uncertainty.ndim != 2 or ce_uncertainty.shape[-1] != self.num_users:
            raise ValueError(
                f"Expected ce_uncertainty [B,{self.num_users}], "
                f"got {tuple(ce_uncertainty.shape)}"
            )

        u = ce_uncertainty.real.clamp(1e-4, 1e2)
        log_u = torch.log(u)

        global_log_u = log_u.mean(dim=-1, keepdim=True)
        relative_log_u = log_u - global_log_u
        global_log_u = global_log_u.expand_as(log_u)

        stream_reliability = 1.0 / (1.0 + u)
        pair_reliability = torch.sqrt(
            stream_reliability[:, :, None] *
            stream_reliability[:, None, :]
        ).clamp(1e-4, 1.0)

        return global_log_u, relative_log_u, pair_reliability

    def gate_betas(self):
        if not self.use_ce_uncertainty:
            return torch.empty(0, device=self.logit_head.weight.device)
        return torch.stack([layer.gate_beta() for layer in self.layers])

    def forward(self, z, gram, cavity_mean, cavity_precision, base_log_prob, iteration, edge_state, ce_uncertainty=None):
        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
        mf = z / diag.to(z.dtype)

        base_log_post = torch.log_softmax(base_log_prob, dim=-1).clamp_min(-20.0)

        scalar_features = torch.stack([
            mf.real,
            mf.imag,
            cavity_mean.real,
            cavity_mean.imag,
            torch.log(cavity_precision.clamp_min(1e-8)),
            torch.log(diag),
        ], dim=-1)

        edge_reliability = None

        if self.use_ce_uncertainty:
            if ce_uncertainty is None:
                raise ValueError("ce_uncertainty is required for UA-GT-EP.")
            global_log_u, relative_log_u, edge_reliability = self.build_uncertainty_context(ce_uncertainty)
            scalar_features = torch.cat([
                scalar_features,
                global_log_u[..., None],
                relative_log_u[..., None],
            ], dim=-1)

        node_features = torch.cat([base_log_post, scalar_features], dim=-1)
        x = self.node_encoder(node_features)

        iteration_index = torch.full(
            (x.shape[0], self.num_users),
            int(iteration),
            dtype=torch.long,
            device=x.device,
        )
        x = x + self.iteration_embedding(iteration_index)

        if self.message_mode == "cross_user":
            attention_mask = self.cross_mask
        else:
            attention_mask = torch.eye(
                self.num_users,
                dtype=torch.bool,
                device=x.device,
            )[None]

        for layer in self.layers:
            x = layer(
                x,
                edge_state,
                attention_mask,
                edge_reliability=edge_reliability,
            )

        raw_delta = self.logit_head(self.final_norm(x))
        delta_log_prob = self.max_logit_correction * torch.tanh(raw_delta)
        delta_log_prob = delta_log_prob - delta_log_prob.mean(dim=-1, keepdim=True)

        return delta_log_prob


class GraphTransformerEPDetector(nn.Module):
    def __init__(self, cfg, num_users=16, num_iterations=5, damping=0.5, min_variance=1e-6, min_site_precision=1e-6, d_model=128, num_heads=8, num_layers=4, edge_dim=32, ffn_dim=256, dropout=0.05, max_logit_correction=4.0, edge_mode="full", message_mode="cross_user", use_ce_uncertainty=False):
        super().__init__()

        self.num_users = int(num_users)
        self.num_iterations = int(num_iterations)
        self.damping = float(damping)
        self.min_variance = float(min_variance)
        self.min_site_precision = float(min_site_precision)
        self.use_ce_uncertainty = bool(use_ce_uncertainty)

        self.bits_per_symbol = int(cfg["modulation"]["bits_per_symbol"])

        if self.bits_per_symbol not in {2, 4, 6}:
            raise ValueError("GT-EP currently supports QPSK, 16-QAM, and 64-QAM.")

        constellation = Constellation(
            "qam",
            self.bits_per_symbol,
            precision=cfg["general"]["precision"],
            device=cfg["general"]["device"],
        )

        points = constellation.points.detach().clone()
        expected_points = 1 << self.bits_per_symbol

        if points.numel() != expected_points:
            raise ValueError(
                f"Expected {expected_points} constellation points for "
                f"{self.bits_per_symbol} bits/symbol, got {points.numel()}."
            )

        self.register_buffer("points", points)

        indices = torch.arange(points.numel(), dtype=torch.long, device=points.device)
        shifts = torch.arange(self.bits_per_symbol - 1, -1, -1, dtype=torch.long, device=points.device)
        bit_labels = ((indices[:, None] >> shifts[None, :]) & 1).to(torch.bool)
        self.register_buffer("bit_labels", bit_labels)

        self.graph_refiner = InterferenceGraphLogitRefiner(
            num_users=self.num_users,
            num_symbols=points.numel(),
            num_iterations=self.num_iterations,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            edge_dim=edge_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_logit_correction=max_logit_correction,
            edge_mode=edge_mode,
            message_mode=message_mode,
            use_ce_uncertainty=self.use_ce_uncertainty,
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

    def forward(self, z, gram, return_iterations=(5,), ce_uncertainty=None):
        if z.ndim != 2:
            raise ValueError(f"Expected z [B,K], got {tuple(z.shape)}")

        if gram.ndim != 3:
            raise ValueError(f"Expected Gram [B,K,K], got {tuple(gram.shape)}")

        if z.shape[-1] != self.num_users:
            raise ValueError(f"Expected {self.num_users} users, got {z.shape[-1]}")

        if gram.shape[-2:] != (self.num_users, self.num_users):
            raise ValueError(
                f"Expected Gram [B,{self.num_users},{self.num_users}], "
                f"got {tuple(gram.shape)}"
            )

        if self.use_ce_uncertainty:
            if ce_uncertainty is None:
                raise ValueError("ce_uncertainty must be supplied to UA-GT-EP.")
            if ce_uncertainty.shape != z.shape:
                raise ValueError(
                    f"Expected ce_uncertainty {tuple(z.shape)}, "
                    f"got {tuple(ce_uncertainty.shape)}"
                )

        gram = 0.5 * (gram + gram.mH)
        real_dtype = gram.real.dtype

        site_precision = torch.ones(z.shape, dtype=real_dtype, device=z.device)
        site_natural = torch.zeros_like(z)

        requested = set(int(i) for i in return_iterations)
        outputs = {}

        validity = []
        correction_rms = []

        edge_state = self.graph_refiner.build_edge_features(gram)

        for iteration in range(1, self.num_iterations + 1):
            mean, variance = self._gaussian_marginals(
                z,
                gram,
                site_precision,
                site_natural,
            )

            marginal_precision = 1.0 / variance
            cavity_precision = (
                marginal_precision - site_precision
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

            delta_log_prob = self.graph_refiner(
                z=z,
                gram=gram,
                cavity_mean=cavity_mean,
                cavity_precision=cavity_precision,
                base_log_prob=base_log_prob,
                iteration=iteration - 1,
                edge_state=edge_state,
                ce_uncertainty=ce_uncertainty,
            )

            refined_log_prob = base_log_prob + delta_log_prob

            post_mean, post_variance = self._posterior_from_log_prob(
                refined_log_prob
            )
            llr = self._llr_from_log_prob(refined_log_prob)

            correction_rms.append(
                delta_log_prob.square().mean().sqrt().detach()
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

            validity.append(valid.float().mean().detach())

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

        result = {
            "llr": outputs[self.num_iterations]["llr"],
            "x_hat": outputs[self.num_iterations]["x_hat"],
            "posterior_variance": outputs[self.num_iterations]["posterior_variance"],
            "iterations": outputs,
            "valid_update_fraction": torch.stack(validity),
            "correction_rms": torch.stack(correction_rms),
            "site_precision": site_precision,
            "site_natural": site_natural,
        }

        if self.use_ce_uncertainty:
            result["uncertainty_gate_beta"] = self.graph_refiner.gate_betas()

        return result