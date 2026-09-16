import copy
import torch
from sionna.phy.channel import ApplyOFDMChannel

from data.channels.uma import UMAChannelProvider
from data.link.modulation import UplinkDataGenerator
from data.link.stream_mapping import RankOneStreamMapper


class UplinkInterferenceGenerator:
    def __init__(self, cfg, protect_dmrs=True):
        icfg = cfg["interference"]
        self.enabled = bool(icfg["enabled"])
        self.num_interferers = int(icfg["num_interferers"])
        self.iot_db = float(icfg["iot_db"])
        self.iot_definition = icfg["iot_definition"]
        self.device = cfg["general"]["device"]
        self.precision = cfg["general"]["precision"]
        self.protect_dmrs = bool(protect_dmrs)

        int_cfg = copy.deepcopy(cfg)
        int_cfg["general"]["num_ues"] = self.num_interferers
        int_cfg["general"]["streams_per_ue"] = 1

        self.channel = UMAChannelProvider(int_cfg)
        self.mapper = RankOneStreamMapper(
            num_tx_antennas=self.channel.ut_array.num_ant,
            mode=cfg["stream_mapping"]["mode"],
            device=self.device,
            precision=self.precision)
        self.data_generator = UplinkDataGenerator(int_cfg, self.channel.resource_grid)
        self.apply_channel = ApplyOFDMChannel(precision=self.precision, device=self.device)

        strength_db = torch.tensor(icfg["relative_strength_db"], dtype=torch.float32, device=self.device)
        if strength_db.numel() != self.num_interferers:
            raise ValueError("relative_strength_db must have one value per interferer.")

        strength = torch.pow(10.0, strength_db / 10.0)
        self.relative_weights = strength / strength.sum()

    def _total_inr_linear(self):
        value = 10.0 ** (self.iot_db / 10.0)
        if self.iot_definition == "i_over_n":
            return value
        if self.iot_definition == "total_over_thermal":
            return value - 1.0
        raise ValueError(f"Unsupported IoT definition: {self.iot_definition}")

    def sample(self, batch_size, noise_var):
        h_raw, topology = self.channel.sample(batch_size)
        h_eff = self.mapper(h_raw)
        shape = [batch_size, self.channel.resource_grid.num_ofdm_symbols,
                 self.channel.resource_grid.num_subcarriers, self.num_interferers]
        x_grid, _ = self.data_generator.sample_symbols(shape)

        if self.protect_dmrs:
            x_grid[:, list(self.channel.resource_grid.dmrs_symbols)] = 0

        data_idx = list(self.channel.resource_grid.data_symbols)
        h_data = h_eff[:, data_idx]
        channel_power = h_data.abs().square().mean(dim=(1, 2, 3))

        total_inr = self._total_inr_linear()
        target_inr = total_inr * self.relative_weights
        target_rx_power = noise_var[:, None] * target_inr[None]

        eps = torch.finfo(channel_power.dtype).tiny
        tx_power = target_rx_power / channel_power.clamp_min(eps)

        x_ant = self.mapper.map_symbols(x_grid)
        x_ant = x_ant * torch.sqrt(tx_power).to(x_ant.dtype)[:, :, None, None, None]

        interference = self.apply_channel(x_ant, h_raw)
        interference = interference[:, 0].permute(0, 2, 3, 1).contiguous()

        h_scaled = h_eff * torch.sqrt(tx_power).to(h_eff.dtype)[:, None, None, None, :]
        h_cov = h_scaled[:, data_idx]
        num_re = len(data_idx) * self.channel.resource_grid.num_subcarriers

        r_int = torch.einsum("bsfml,bsfnl->bmn", h_cov, h_cov.conj()) / num_re
        r_int = 0.5 * (r_int + r_int.mH)

        return {
            "interference": interference,
            "ruu_interference": r_int,
            "h_interference": h_scaled,
            "tx_power": tx_power,
            "target_inr": target_inr,
            "topology": topology,
        }