import torch
import torch.nn as nn
from sionna.phy.mapping import Constellation


class DETRLogitRefiner(nn.Module):
    """DETR-style stream-query decoder for EP belief refinement.

    Each stream has one learned query. Queries self-attend and cross-attend to a
    set of stream memory tokens carrying the same EP residual/uncertainty state
    used by the residual Graph-Transformer branch. Stream identities are fixed,
    so no Hungarian matching is needed.
    """

    def __init__(self, num_users=16, num_symbols=16, num_iterations=5, d_model=128, num_heads=8, num_layers=3, ffn_dim=256, dropout=0.05, max_logit_correction=4.0):
        super().__init__()
        self.num_users = int(num_users)
        self.num_symbols = int(num_symbols)
        self.num_iterations = int(num_iterations)
        self.max_logit_correction = float(max_logit_correction)

        # Pair features per j: rho(Re,Im,abs), current contribution(Re,Im),
        # uncertainty-weighted coupling, residual alignment.
        self.pair_feature_dim = 7
        node_scalar_dim = 12
        memory_input_dim = self.num_symbols + node_scalar_dim + self.num_users * self.pair_feature_dim

        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )
        self.query_embed = nn.Parameter(torch.randn(1, self.num_users, d_model) * 0.02)
        self.stream_embedding = nn.Embedding(self.num_users, d_model)
        self.iteration_embedding = nn.Embedding(self.num_iterations, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.final_norm = nn.LayerNorm(d_model)
        self.logit_head = nn.Linear(d_model, self.num_symbols)

        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

        cross_mask = ~torch.eye(self.num_users, dtype=torch.bool)
        self.register_buffer("cross_mask", cross_mask[None])

    def forward(self, z, gram, cavity_mean, cavity_precision, base_log_prob, base_mean, base_variance, iteration):
        b, k = z.shape
        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
        sqrt_diag = torch.sqrt(diag).clamp_min(1e-8)
        denom = torch.sqrt(diag[:, :, None] * diag[:, None, :]).clamp_min(1e-8)
        rho = gram / denom
        mf = z / diag.to(z.dtype)
        base_log_post = torch.log_softmax(base_log_prob, dim=-1).clamp_min(-20.0)

        predicted_z = torch.einsum("bij,bj->bi", gram, base_mean)
        residual = z - predicted_z
        residual_norm = residual / sqrt_diag.to(residual.dtype)
        innovation = base_mean - cavity_mean

        node_scalars = torch.stack([
            mf.real,
            mf.imag,
            cavity_mean.real,
            cavity_mean.imag,
            torch.log(cavity_precision.clamp_min(1e-8)),
            torch.log(diag),
            residual_norm.real,
            residual_norm.imag,
            torch.log1p(residual_norm.abs().square()),
            torch.log(base_variance.clamp_min(1e-8)),
            innovation.real,
            innovation.imag,
        ], dim=-1)

        contribution = rho * base_mean[:, None, :]
        uncertainty = rho.abs() * torch.sqrt(base_variance.clamp_min(1e-8))[:, None, :]
        alignment = (residual_norm[:, :, None].conj() * contribution).real
        pair_features = torch.stack([
            rho.real,
            rho.imag,
            rho.abs(),
            contribution.real,
            contribution.imag,
            uncertainty,
            alignment,
        ], dim=-1)
        pair_features = pair_features * self.cross_mask[..., None].to(pair_features.dtype)
        pair_flat = pair_features.reshape(b, k, self.num_users * self.pair_feature_dim)

        memory_features = torch.cat([base_log_post, node_scalars, pair_flat], dim=-1)
        stream_index = torch.arange(self.num_users, device=z.device)
        stream_embed = self.stream_embedding(stream_index)[None]
        memory = self.memory_encoder(memory_features) + stream_embed

        iteration_index = torch.full((b, self.num_users), int(iteration), dtype=torch.long, device=z.device)
        queries = self.query_embed.expand(b, -1, -1) + stream_embed + self.iteration_embedding(iteration_index)
        decoded = self.decoder(tgt=queries, memory=memory)

        raw_delta = self.logit_head(self.final_norm(decoded))
        delta_log_prob = self.max_logit_correction * torch.tanh(raw_delta)
        delta_log_prob = delta_log_prob - delta_log_prob.mean(dim=-1, keepdim=True)

        residual_rms = residual_norm.abs().square().mean().sqrt().detach()
        return delta_log_prob, residual_rms


class DETREPDetector(nn.Module):
    def __init__(self, cfg, num_users=16, num_iterations=5, damping=0.5, min_variance=1e-6, min_site_precision=1e-6, d_model=128, num_heads=8, num_layers=3, ffn_dim=256, dropout=0.05, max_logit_correction=4.0):
        super().__init__()
        self.num_users = int(num_users)
        self.num_iterations = int(num_iterations)
        self.damping = float(damping)
        self.min_variance = float(min_variance)
        self.min_site_precision = float(min_site_precision)
        self.bits_per_symbol = int(cfg["modulation"]["bits_per_symbol"])

        if self.bits_per_symbol not in {2, 4, 6}:
            raise ValueError("DETR-EP supports QPSK, 16-QAM, and 64-QAM.")

        constellation = Constellation("qam", self.bits_per_symbol, precision=cfg["general"]["precision"], device=cfg["general"]["device"])
        points = constellation.points.detach().clone()
        expected_points = 1 << self.bits_per_symbol
        if points.numel() != expected_points:
            raise ValueError(f"Expected {expected_points} constellation points, got {points.numel()}.")
        self.register_buffer("points", points)

        indices = torch.arange(points.numel(), dtype=torch.long, device=points.device)
        shifts = torch.arange(self.bits_per_symbol - 1, -1, -1, dtype=torch.long, device=points.device)
        self.register_buffer("bit_labels", ((indices[:, None] >> shifts[None, :]) & 1).to(torch.bool))

        self.refiner = DETRLogitRefiner(
            num_users=self.num_users,
            num_symbols=points.numel(),
            num_iterations=self.num_iterations,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_logit_correction=max_logit_correction,
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
            llrs.append(torch.logsumexp(log_prob[..., mask1], dim=-1) - torch.logsumexp(log_prob[..., mask0], dim=-1))
        return torch.stack(llrs, dim=-1)

    def forward(self, z, gram, return_iterations=(5,)):
        if z.ndim != 2:
            raise ValueError(f"Expected z [B,K], got {tuple(z.shape)}")
        if gram.ndim != 3:
            raise ValueError(f"Expected Gram [B,K,K], got {tuple(gram.shape)}")
        if z.shape[-1] != self.num_users or gram.shape[-2:] != (self.num_users, self.num_users):
            raise ValueError(f"Expected K={self.num_users}, got z={tuple(z.shape)}, Gram={tuple(gram.shape)}")

        gram = 0.5 * (gram + gram.mH)
        real_dtype = gram.real.dtype
        site_precision = torch.ones(z.shape, dtype=real_dtype, device=z.device)
        site_natural = torch.zeros_like(z)
        requested = set(int(i) for i in return_iterations)
        outputs = {}
        validity = []
        correction_rms = []
        residual_rms = []

        for iteration in range(1, self.num_iterations + 1):
            mean, variance = self._gaussian_marginals(z, gram, site_precision, site_natural)
            marginal_precision = 1.0 / variance
            cavity_precision = (marginal_precision - site_precision).clamp_min(self.min_site_precision)
            cavity_natural = mean / variance.to(mean.dtype) - site_natural
            cavity_mean = cavity_natural / cavity_precision.to(cavity_natural.dtype)

            base_log_prob = self._base_log_prob(cavity_mean, cavity_precision)
            base_mean, base_variance = self._posterior_from_log_prob(base_log_prob)
            delta_log_prob, res_rms = self.refiner(
                z=z,
                gram=gram,
                cavity_mean=cavity_mean,
                cavity_precision=cavity_precision,
                base_log_prob=base_log_prob,
                base_mean=base_mean,
                base_variance=base_variance,
                iteration=iteration - 1,
            )

            refined_log_prob = base_log_prob + delta_log_prob
            post_mean, post_variance = self._posterior_from_log_prob(refined_log_prob)
            llr = self._llr_from_log_prob(refined_log_prob)
            correction_rms.append(delta_log_prob.square().mean().sqrt().detach())
            residual_rms.append(res_rms)

            if iteration in requested:
                outputs[iteration] = {
                    "llr": llr,
                    "x_hat": post_mean,
                    "posterior_variance": post_variance,
                    "delta_log_prob": delta_log_prob,
                }

            candidate_precision = 1.0 / post_variance - cavity_precision
            candidate_natural = post_mean / post_variance.to(post_mean.dtype) - cavity_natural
            valid = (
                torch.isfinite(candidate_precision)
                & torch.isfinite(candidate_natural.real)
                & torch.isfinite(candidate_natural.imag)
                & (candidate_precision > self.min_site_precision)
            )
            validity.append(valid.float().mean().detach())
            candidate_precision = torch.where(valid, candidate_precision, site_precision)
            candidate_natural = torch.where(valid, candidate_natural, site_natural)
            site_precision = ((1.0 - self.damping) * site_precision + self.damping * candidate_precision).clamp_min(self.min_site_precision)
            site_natural = (1.0 - self.damping) * site_natural + self.damping * candidate_natural

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
            "residual_rms": torch.stack(residual_rms),
            "site_precision": site_precision,
            "site_natural": site_natural,
        }
