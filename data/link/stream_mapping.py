import math
import torch


class RankOneStreamMapper:
    def __init__(self, num_tx_antennas=4, mode="equal_gain", device="cuda:0", precision="single"):
        self.num_tx_antennas = int(num_tx_antennas)
        self.mode = mode
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        if mode == "equal_gain":
            p = torch.ones(self.num_tx_antennas, dtype=cdtype, device=device) / math.sqrt(self.num_tx_antennas)
        elif mode == "first_antenna":
            p = torch.zeros(self.num_tx_antennas, dtype=cdtype, device=device)
            p[0] = 1.0
        else:
            raise ValueError(f"Unsupported stream mapping mode: {mode}")

        self.precoder = p

    def __call__(self, h_raw):
        if h_raw.ndim != 7:
            raise ValueError(f"Expected h_raw rank 7, got shape {tuple(h_raw.shape)}")
        if h_raw.shape[1] != 1:
            raise ValueError(f"Expected one BS receiver dimension, got {h_raw.shape[1]}")
        if h_raw.shape[4] != self.num_tx_antennas:
            raise ValueError(f"Expected {self.num_tx_antennas} UE antennas, got {h_raw.shape[4]}")

        h_eff = torch.einsum("braktsf,t->braksf", h_raw, self.precoder)
        return h_eff[:, 0].permute(0, 3, 4, 1, 2).contiguous()

    def map_symbols(self, x_grid):
        if x_grid.ndim != 4:
            raise ValueError(f"Expected x_grid [B,S,F,K], got {tuple(x_grid.shape)}")

        x = x_grid.permute(0, 3, 1, 2).unsqueeze(2)
        return x * self.precoder.view(1, 1, self.num_tx_antennas, 1, 1)