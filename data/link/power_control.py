import torch


class FractionalPowerController:
    def __init__(self, cfg):
        pcfg = cfg["power_control"]
        self.mode = pcfg["mode"]
        self.alpha = float(pcfg["alpha"])
        self.max_boost_db = float(pcfg["max_boost_db"])
        self.max_attenuation_db = float(pcfg["max_attenuation_db"])

    def __call__(self, h):
        if h.ndim != 5:
            raise ValueError(f"Expected H [B,S,F,M,K], got {tuple(h.shape)}")

        gain = h.abs().square().mean(dim=(1, 2, 3)).real
        eps = torch.finfo(gain.dtype).tiny

        if self.mode == "none":
            power = torch.ones_like(gain)
        elif self.mode == "fractional":
            gain_db = 10.0 * torch.log10(gain.clamp_min(eps))
            gain_db_centered = gain_db - gain_db.mean(dim=-1, keepdim=True)
            power_db = -self.alpha * gain_db_centered
            power_db = power_db.clamp(min=-self.max_attenuation_db, max=self.max_boost_db)
            power = torch.pow(10.0, power_db / 10.0)
            power = power / power.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unsupported power-control mode: {self.mode}")

        sqrt_power = torch.sqrt(power)
        h_powered = h * sqrt_power[:, None, None, None, :]

        return {
            "h_powered": h_powered,
            "tx_power": power,
            "tx_power_db": 10.0 * torch.log10(power.clamp_min(eps)),
            "channel_gain": gain,
        }