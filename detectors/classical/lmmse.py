import torch
from sionna.phy.mapping import Demapper

from data.preprocessing.whitening import CovarianceAwareFrontEnd


class LMMSESoftDetector:
    def __init__(self, cfg):
        self.precision = cfg["general"]["precision"]
        self.device = cfg["general"]["device"]
        self.num_bits_per_symbol = int(cfg["modulation"]["bits_per_symbol"])
        self.frontend = CovarianceAwareFrontEnd(self.precision)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol,
                                 precision=self.precision, device=self.device)

    def __call__(self, y, h, ruu):
        stats = self.frontend(y, h, ruu)
        z, gram = stats["z"], stats["gram"]
        k = gram.shape[-1]

        eye = torch.eye(k, dtype=gram.dtype, device=gram.device)
        a = gram + eye
        chol, info = torch.linalg.cholesky_ex(a, check_errors=False)
        if (info != 0).any().item():
            raise RuntimeError("LMMSE system matrix is not positive definite.")

        gy = torch.cholesky_solve(z.unsqueeze(-1), chol).squeeze(-1)
        gh = torch.cholesky_solve(gram, chol)
        d = torch.diagonal(gh, dim1=-2, dim2=-1).real
        eps = 1e-7 if self.precision == "single" else 1e-12
        d = d.clamp(min=eps, max=1.0 - eps)

        x_hat = gy / d
        no_eff = 1.0 / d - 1.0

        llr = self.demapper(x_hat, no_eff)
        llr = llr.reshape(*x_hat.shape, self.num_bits_per_symbol)

        return {
            "x_hat": x_hat,
            "no_eff": no_eff,
            "llr": llr,
            "z": z,
            "gram": gram,
        }