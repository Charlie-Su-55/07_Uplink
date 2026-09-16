import torch
from sionna.phy.channel import ApplyOFDMChannel, AWGN


class UplinkLink:
    def __init__(self, cfg, resource_grid, stream_mapper):
        self.cfg = cfg
        self.grid = resource_grid
        self.stream_mapper = stream_mapper
        self.device = cfg["general"]["device"]
        self.precision = cfg["general"]["precision"]
        self.num_rx_ant = int(cfg["bs_array"]["num_rows_per_panel"]) * int(cfg["bs_array"]["num_cols_per_panel"]) * 2
        self.noise_mode = cfg["link"]["noise_mode"]
        self.rx_snr_db = float(cfg["link"]["rx_snr_db"])
        self.apply_channel = ApplyOFDMChannel(precision=self.precision, device=self.device)
        self.awgn = AWGN(precision=self.precision, device=self.device)

    def clean_signal(self, h_raw, x_grid, tx_power):
        x_ant = self.stream_mapper.map_symbols(x_grid)
        x_ant = x_ant * torch.sqrt(tx_power).to(x_ant.dtype)[:, :, None, None, None]
        y_clean = self.apply_channel(x_ant, h_raw)
        y_clean = y_clean[:, 0].permute(0, 2, 3, 1).contiguous()
        return {"x_ant": x_ant, "y_clean": y_clean}

    def noise_variance(self, y_clean):
        if self.noise_mode != "rx_snr":
            raise ValueError(f"Unsupported noise mode: {self.noise_mode}")

        y_data = y_clean[:, list(self.grid.data_symbols)]
        signal_power = y_data.abs().square().mean(dim=(1, 2, 3))
        snr_linear = 10.0 ** (self.rx_snr_db / 10.0)
        return signal_power / snr_linear

    def finalize(self, y_clean, noise_var, interference=None, ruu_interference=None):
        y = self.awgn(y_clean, noise_var)
        if interference is not None:
            y = y + interference

        eye = torch.eye(self.num_rx_ant, dtype=y.dtype, device=self.device)
        ruu_noise = noise_var[:, None, None] * eye[None]
        ruu = ruu_noise if ruu_interference is None else ruu_noise + ruu_interference

        return {
            "y": y,
            "y_clean": y_clean,
            "noise_var": noise_var,
            "ruu_noise": ruu_noise,
            "ruu_true": ruu,
        }

    def __call__(self, h_raw, x_grid, tx_power):
        clean = self.clean_signal(h_raw, x_grid, tx_power)
        noise_var = self.noise_variance(clean["y_clean"])
        out = self.finalize(clean["y_clean"], noise_var)
        out["x_ant"] = clean["x_ant"]
        return out