import torch


class CovarianceAwareFrontEnd:
    def __init__(self, precision="single"):
        self.precision = precision

    def __call__(self, y, h, ruu):
        if y.ndim != 4:
            raise ValueError(f"Expected y [B,S,F,M], got {tuple(y.shape)}")
        if h.ndim != 5:
            raise ValueError(f"Expected h [B,S,F,M,K], got {tuple(h.shape)}")
        if ruu.ndim != 3:
            raise ValueError(f"Expected Ruu [B,M,M], got {tuple(ruu.shape)}")
        if y.shape[:-1] != h.shape[:-2] or y.shape[-1] != h.shape[-2]:
            raise ValueError(f"Incompatible y/h shapes: {tuple(y.shape)}, {tuple(h.shape)}")

        b, s, f, m = y.shape
        k = h.shape[-1]
        ruu = ruu.to(dtype=h.dtype, device=h.device)

        chol, info = torch.linalg.cholesky_ex(ruu, check_errors=False)
        if (info != 0).any().item():
            raise RuntimeError("Ruu is not positive definite.")

        y_rhs = y.permute(0, 3, 1, 2).reshape(b, m, s * f)
        h_rhs = h.permute(0, 3, 1, 2, 4).reshape(b, m, s * f * k)

        y_white = torch.linalg.solve_triangular(chol, y_rhs, upper=False)
        h_white = torch.linalg.solve_triangular(chol, h_rhs, upper=False)

        y_white = y_white.reshape(b, m, s, f).permute(0, 2, 3, 1).contiguous()
        h_white = h_white.reshape(b, m, s, f, k).permute(0, 2, 3, 1, 4).contiguous()

        z = torch.einsum("bsfmk,bsfm->bsfk", h_white.conj(), y_white)
        gram = torch.einsum("bsfmk,bsfml->bsfkl", h_white.conj(), h_white)
        gram = 0.5 * (gram + gram.mH)

        return {"z": z, "gram": gram}