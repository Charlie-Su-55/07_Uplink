import torch
from sionna.phy.mapping import Constellation


class ExpectationPropagationDetector:
    def __init__(self, cfg, num_iterations=5, damping=0.5,
                 min_variance=1e-6, min_site_precision=1e-6):
        self.num_iterations = int(num_iterations)
        self.damping = float(damping)
        self.min_variance = float(min_variance)
        self.min_site_precision = float(min_site_precision)

        self.bits_per_symbol = int(cfg["modulation"]["bits_per_symbol"])
        self.device = cfg["general"]["device"]
        self.precision = cfg["general"]["precision"]

        constellation = Constellation(
            "qam",
            self.bits_per_symbol,
            precision=self.precision,
            device=self.device,
        )
        self.points = constellation.points

        num_points = self.points.numel()
        indices = torch.arange(num_points, dtype=torch.long, device=self.device)
        shifts = torch.arange(
            self.bits_per_symbol - 1, -1, -1,
            dtype=torch.long, device=self.device
        )

        self.bit_labels = (
            (indices[:, None] >> shifts[None, :]) & 1
        ).to(torch.bool)

    def _gaussian_marginals(self, z, gram, site_precision, site_natural):
        precision = gram + torch.diag_embed(site_precision.to(gram.dtype))
        precision = 0.5 * (precision + precision.mH)

        chol = torch.linalg.cholesky(precision)
        eta = z + site_natural

        mean = torch.cholesky_solve(
            eta.unsqueeze(-1), chol
        ).squeeze(-1)

        covariance = torch.cholesky_inverse(chol)
        variance = torch.diagonal(
            covariance, dim1=-2, dim2=-1
        ).real.clamp_min(self.min_variance)

        return mean, variance

    def _discrete_posterior(self, cavity_mean, cavity_precision):
        points = self.points.to(cavity_mean.device)

        diff = cavity_mean[..., None] - points
        log_prob = -cavity_precision[..., None] * diff.abs().square()

        prob = torch.softmax(log_prob, dim=-1)
        post_mean = (prob * points).sum(dim=-1)

        second_moment = (
            prob * points.abs().square()
        ).sum(dim=-1)

        post_variance = (
            second_moment - post_mean.abs().square()
        ).real.clamp_min(self.min_variance)

        return log_prob, post_mean, post_variance

    def _llr_from_log_prob(self, log_prob):
        llrs = []

        for q in range(self.bits_per_symbol):
            mask1 = self.bit_labels[:, q]
            mask0 = ~mask1

            log_p1 = torch.logsumexp(
                log_prob[..., mask1], dim=-1
            )
            log_p0 = torch.logsumexp(
                log_prob[..., mask0], dim=-1
            )

            llrs.append(log_p1 - log_p0)

        return torch.stack(llrs, dim=-1)

    @torch.no_grad()
    def __call__(self, z, gram, return_iterations=(1, 3, 5)):
        if z.ndim < 2:
            raise ValueError(f"Expected z [...,K], got {tuple(z.shape)}")
        if gram.shape[:-2] != z.shape[:-1]:
            raise ValueError("Leading dimensions of z and Gram do not match.")
        if gram.shape[-1] != z.shape[-1] or gram.shape[-2] != z.shape[-1]:
            raise ValueError("Stream dimensions of z and Gram do not match.")

        gram = 0.5 * (gram + gram.mH)
        real_dtype = gram.real.dtype

        site_precision = torch.ones(
            z.shape,
            dtype=real_dtype,
            device=z.device,
        )

        site_natural = torch.zeros_like(z)

        requested = set(int(i) for i in return_iterations)
        outputs = {}
        update_validity = []

        for iteration in range(1, self.num_iterations + 1):
            mean, variance = self._gaussian_marginals(
                z, gram, site_precision, site_natural
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

            log_prob, post_mean, post_variance = (
                self._discrete_posterior(
                    cavity_mean,
                    cavity_precision,
                )
            )

            llr = self._llr_from_log_prob(log_prob)

            if iteration in requested:
                outputs[iteration] = {
                    "llr": llr.clone(),
                    "x_hat": post_mean.clone(),
                    "posterior_variance": post_variance.clone(),
                }

            candidate_precision = (
                1.0 / post_variance - cavity_precision
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

            update_validity.append(valid.float().mean().item())

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

        final_iteration = self.num_iterations

        if final_iteration not in outputs:
            outputs[final_iteration] = {
                "llr": llr,
                "x_hat": post_mean,
                "posterior_variance": post_variance,
            }

        return {
            "llr": outputs[final_iteration]["llr"],
            "x_hat": outputs[final_iteration]["x_hat"],
            "posterior_variance": outputs[final_iteration]["posterior_variance"],
            "iterations": outputs,
            "valid_update_fraction": update_validity,
            "site_precision": site_precision,
            "site_natural": site_natural,
        }