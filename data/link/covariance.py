import torch


class CovarianceEstimator:
    def __init__(self, cfg, resource_grid):
        ccfg = cfg["covariance_estimation"]
        self.observation_symbols = tuple(int(x) for x in ccfg["observation_symbols"])
        self.mode = ccfg["mode"]
        self.shrinkage_lambda = float(ccfg["shrinkage_lambda"])
        if not set(self.observation_symbols).issubset(set(resource_grid.omitted_symbols)):
            raise ValueError("Covariance observation symbols must be desired-user silent symbols.")
        if not 0.0 <= self.shrinkage_lambda <= 1.0:
            raise ValueError("shrinkage_lambda must be between 0 and 1.")

    def shrink(self, ruu_scm, shrinkage_lambda=None):
        lam = self.shrinkage_lambda if shrinkage_lambda is None else float(shrinkage_lambda)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("shrinkage_lambda must be between 0 and 1.")
        m = ruu_scm.shape[-1]
        mean_power = torch.diagonal(ruu_scm, dim1=-2, dim2=-1).real.mean(dim=-1)
        eye = torch.eye(m, dtype=ruu_scm.dtype, device=ruu_scm.device)
        isotropic = mean_power[:, None, None] * eye[None]
        out = (1.0 - lam) * ruu_scm + lam * isotropic
        return 0.5 * (out + out.mH)

    @staticmethod
    def scm(snapshots):
        n = snapshots.shape[1]
        out = torch.einsum("btm,btn->bmn", snapshots, snapshots.conj()) / n
        return 0.5 * (out + out.mH)

    def select_mode(self, ruu_scm):
        if self.mode == "scm":
            return ruu_scm
        if self.mode == "shrinkage":
            return self.shrink(ruu_scm)
        raise ValueError(f"Unsupported covariance estimation mode: {self.mode}")

    def __call__(self, y):
        if y.ndim != 4:
            raise ValueError(f"Expected y [B,S,F,M], got {tuple(y.shape)}")

        y_obs = y[:, list(self.observation_symbols)]
        b, s, f, m = y_obs.shape
        snapshots = y_obs.reshape(b, s * f, m)

        ruu_scm = self.scm(snapshots)
        ruu_shrinkage = self.shrink(ruu_scm)
        ruu_hat = self.select_mode(ruu_scm)

        view_scm, view_hat = [], []
        for symbol in self.observation_symbols:
            scm_i = self.scm(y[:, symbol])
            view_scm.append(scm_i)
            view_hat.append(self.select_mode(scm_i))

        return {
            "ruu_hat": ruu_hat,
            "ruu_scm": ruu_scm,
            "ruu_shrinkage": ruu_shrinkage,
            "ruu_views": torch.stack(view_hat, dim=1),
            "ruu_view_scm": torch.stack(view_scm, dim=1),
            "view_symbols": self.observation_symbols,
            "num_snapshots": snapshots.shape[1],
            "num_snapshots_per_view": f,
        }