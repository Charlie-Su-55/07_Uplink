"""Independent coded, normalized UMa / perfect-CSI / native LMMSE reference.

Only the UMa provider and unit-norm rank-one mapper are shared with legacy.
No dataset, waveform replacement, CE cache, estimated Ruu, or power control.
"""

import copy
import json
import math
from pathlib import Path

from link_level.nr_codec import NRTransportBlockCodec
from link_level.sionna_ebno import ebno_to_noise, require_sionna2


DEFAULT_CONFIG = "configs/system/uma_16ue_256rx_sionna_ebno.yaml"


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config needs PyYAML in the existing server environment.") from exc
    return yaml.safe_load(text)


def antenna_count(cfg):
    if cfg["polarization"] not in ("single", "dual"):
        raise ValueError("Unsupported antenna polarization.")
    return (int(cfg["num_rows_per_panel"]) * int(cfg["num_cols_per_panel"])
            * (2 if cfg["polarization"] == "dual" else 1))


def validate_reference_config(cfg):
    """Fail closed if a legacy assumption would be silently ignored."""
    required = {
        ("channel", "normalize_channel"): True,
        ("general", "streams_per_ue"): 1,
        ("power_control", "mode"): "none",
        ("interference", "enabled"): False,
        ("channel_estimation", "mode"): "perfect",
        ("link", "noise_mode"): "sionna_ebno",
        ("link", "energy_convention"): "unit_energy_per_ue",
        ("link", "coderate_convention"): "payload_bits_over_coded_bits",
        ("stream_mapping", "mode"): "equal_gain",
        ("topology", "scenario"): "uma",
    }
    for (section, field), value in required.items():
        actual = cfg.get(section, {}).get(field)
        if actual != value or isinstance(value, bool) and actual is not value:
            raise ValueError(f"Reference requires {section}.{field}={value!r}.")
    if "rx_snr_db" in cfg["link"] or "snr_db" in cfg["link"]:
        raise ValueError("Remove legacy SNR fields from the reference config.")
    ofdm = cfg["ofdm"]
    t, f = int(ofdm["num_ofdm_symbols"]), int(ofdm["fft_size"])
    cp = int(ofdm["cyclic_prefix_length"])
    if (t <= 0 or f <= 0 or not 0 <= cp <= f
            or ofdm["num_subcarriers"] != f
            or not math.isfinite(float(ofdm["subcarrier_spacing_hz"]))
            or float(ofdm["subcarrier_spacing_hz"]) <= 0):
        raise ValueError("Invalid OFDM dimensions, CP, or spacing.")
    if (ofdm["omitted_symbols"] or ofdm["dmrs_symbols"]
            or ofdm["data_symbols"] != list(range(t))
            or any(ofdm.get("num_guard_carriers", (0, 0)))
            or ofdm.get("dc_null", False)
            or ofdm.get("pilot_pattern", "empty") != "empty"):
        raise ValueError("First reference requires all REs to carry data, with no pilots/guards/DC null.")
    if "modulation" in cfg:
        raise ValueError("Reference modulation comes from nr.mcs_table/index; remove modulation overrides.")
    k, m, a = int(cfg["general"]["num_ues"]), antenna_count(cfg["bs_array"]), antenna_count(cfg["ut_array"])
    if not 1 <= k <= m or a <= 0:
        raise ValueError("Reference needs positive antenna counts and num_rx >= num_ues >= 1.")
    if cfg["general"]["precision"] not in ("single", "double"):
        raise ValueError("Precision must be single or double.")
    if cfg["nr"]["mcs_table"] not in (1, 2) or int(cfg["nr"]["num_bp_iter"]) <= 0:
        raise ValueError("Invalid NR MCS table or BP iteration count.")
    if not str(cfg["general"]["device"]).startswith("cuda") and m > 32:
        raise ValueError("Large reference simulations require the GPU server; CPU is limited to <=32 Rx.")
    return dict(num_ues=k, num_rx=m, num_tx_antennas=a,
                num_ofdm_symbols=t, fft_size=f, num_data_symbols=t * f,
                cyclic_prefix_length=cp, num_tx=k, num_streams_per_tx=1,
                pilot_pattern="empty", num_guard_carriers=[0, 0], dc_null=False,
                cp_energy_factor=1 + cp / f)


class SionnaReferenceLink:
    def __init__(self, cfg, *, channel_provider=None):
        self.cfg = copy.deepcopy(cfg)
        self.grid_metadata = validate_reference_config(self.cfg)
        self.sionna_version = require_sionna2()
        import numpy as np
        import torch
        from sionna.phy import config as sionna_config
        from sionna.phy.channel import ApplyOFDMChannel, AWGN
        from sionna.phy.mapping import Demapper
        from sionna.phy.mimo import StreamManagement
        from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer
        from data.link.stream_mapping import RankOneStreamMapper

        g, ofdm, nr = self.cfg["general"], self.cfg["ofdm"], self.cfg["nr"]
        self.device, self.precision = g["device"], g["precision"]
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run the 256Rx reference on the GPU server.")
        self.num_ues, self.num_rx = self.grid_metadata["num_ues"], self.grid_metadata["num_rx"]
        self.dtype = torch.float32 if self.precision == "single" else torch.float64
        self.options = dict(precision=self.precision, device=self.device)
        sionna_config.seed = int(g["seed"])
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=ofdm["num_ofdm_symbols"], fft_size=ofdm["fft_size"],
            subcarrier_spacing=ofdm["subcarrier_spacing_hz"], num_tx=self.num_ues,
            num_streams_per_tx=1, cyclic_prefix_length=ofdm["cyclic_prefix_length"],
            num_guard_carriers=(0, 0), dc_null=False, pilot_pattern="empty", **self.options)
        self.codec = NRTransportBlockCodec(
            int(self.resource_grid.num_data_symbols), self.num_ues,
            table=nr["mcs_table"], index=nr["mcs_index"],
            num_bp_iter=nr["num_bp_iter"], **self.options)
        self.grid_mapper = ResourceGridMapper(self.resource_grid, **self.options)
        self.stream_mapper = RankOneStreamMapper(
            self.grid_metadata["num_tx_antennas"], mode="equal_gain", **self.options)
        if channel_provider is None:
            from data.channels.uma import UMAChannelProvider
            channel_provider = UMAChannelProvider(self.cfg)
        self.channel_provider = channel_provider
        self.apply_channel = ApplyOFDMChannel(**self.options)
        self.awgn = AWGN(**self.options)
        stream_management = StreamManagement(np.ones((1, self.num_ues), dtype=int), 1)
        self.equalizer = LMMSEEqualizer(self.resource_grid, stream_management, **self.options)
        self.demapper = Demapper("app", "qam", self.codec.qm, **self.options)

    def noise_variance(self, ebno_db):
        return ebno_to_noise(ebno_db, self.codec.qm, self.codec.info_bits,
                             self.codec.coded_bits, self.resource_grid, **self.options)

    def transmit(self, batch_size, ebno_db, seed):
        import torch
        from sionna.phy import config as sionna_config

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        sionna_config.seed = int(seed)
        rng = sionna_config.torch_rng(self.device)
        with torch.no_grad():
            no = self.noise_variance(ebno_db)
            info = torch.randint(0, 2, (batch_size, self.num_ues, self.codec.info_bits),
                                 dtype=self.dtype, device=self.device, generator=rng)
            coded, symbols = self.codec.encode(info)
            x = self.grid_mapper(symbols)  # [B,K,1,T,F], actual coded symbols
            # Rebuild stochastic topology state for reproducible paired Eb/N0 points.
            if hasattr(self.channel_provider, "channel_model"):
                self.channel_provider.channel_model.reset_topology()
            h_raw, _ = self.channel_provider.sample(batch_size)
            expected = (batch_size, 1, self.num_rx, self.num_ues,
                        self.grid_metadata["num_tx_antennas"],
                        self.grid_metadata["num_ofdm_symbols"], self.grid_metadata["fft_size"])
            if tuple(h_raw.shape) != expected:
                raise RuntimeError(f"Expected raw channel {expected}, got {tuple(h_raw.shape)}.")
            h_eff = self.stream_mapper(h_raw)  # [B,T,F,M,K]; no second normalization
            h = h_eff.permute(0, 3, 4, 1, 2).unsqueeze(1).unsqueeze(4)
            x_ant = self.stream_mapper.map_symbols(x[:, :, 0].permute(0, 2, 3, 1))
            y_clean = self.apply_channel(x_ant, h_raw)
            y = self.awgn(y_clean, no)  # noise added exactly once
            if not torch.isfinite(y).all() or not torch.isfinite(h).all():
                raise RuntimeError("Non-finite channel/received samples.")
            return dict(info=info, coded=coded, x=x, x_ant=x_ant, h_raw=h_raw,
                        h=h, h_eff=h_eff, y_clean=y_clean, y=y, no=no,
                        ebno_db=float(ebno_db), seed=int(seed))

    def receive(self, sample):
        import torch

        with torch.no_grad():
            x_hat, no_eff = self.equalizer(sample["y"], sample["h"], 0.0, sample["no"])
            if (not torch.isfinite(x_hat).all() or not torch.isfinite(no_eff).all()
                    or not (no_eff > 0).all()):
                raise RuntimeError("Invalid LMMSE output/variance; inspect precision and channel conditioning.")
            llr = self.demapper(x_hat, no_eff)
            if not torch.isfinite(llr).all():
                raise RuntimeError("Non-finite APP LLRs.")
            decoded, crc_ok = self.codec.decode(llr)
            return dict(x_hat=x_hat, no_eff=no_eff, llr=llr, decoded=decoded, crc_ok=crc_ok)

    def check_codec_identity(self, sample):
        """Validate the actual run's MCS/TB/scrambling configuration before BLER."""
        import torch

        with torch.no_grad():
            symbols = self.codec.mapper(sample["coded"]).unsqueeze(2)
            llr = self.demapper(symbols, torch.tensor(0.001, dtype=self.dtype, device=self.device))
            decoded, crc = self.codec.decode(llr)
            if not torch.equal(decoded, sample["info"]) or not crc.all():
                raise RuntimeError("Noiseless mapper/APP/TB identity failed; stop PHY evaluation.")
            return {"status": "pass", "blocks": int(crc.numel()), "demapper_n0": 0.001}

    def metadata(self):
        return dict(config=self.cfg, grid=self.grid_metadata, codec=self.codec.metadata(),
                    sionna_version=self.sionna_version, receiver="Sionna LMMSEEqualizer + APP + TBDecoder",
                    noise_convention="complex AWGN: E[|n|^2]=N0, real/imag variance=N0/2",
                    ebno_convention="per-UE payload Eb/N0, actual k/n, CP included by ebnodb2no",
                    channel_normalization="raw link over Rx/Tx antennas, time and frequency; no post-projection renormalization",
                    tx_energy="E[|x_UE|^2]=1, aggregate=K; unit-norm rank-one antenna mapping",
                    channel_estimation_error_variance=0.0, external_interference=False)


def power_diagnostics(sample):
    """Report measured powers separately from the configured Eb/N0 and N0."""
    import torch

    with torch.no_grad():
        x, h_eff = sample["x"], sample["h_eff"]
        clean = sample["y_clean"][:, 0].permute(0, 2, 3, 1)
        reconstructed = torch.einsum("btfmk,btfk->btfm", h_eff, x[:, :, 0].permute(0, 2, 3, 1))
        noise = sample["y"] - sample["y_clean"]
        no = float(sample["no"].item())
        signal_power = sample["y_clean"].abs().square().mean(dim=(1, 2, 3, 4))
        noise_power = noise.abs().square().mean(dim=(1, 2, 3, 4))
        tx_power = x.abs().square().mean(dim=(2, 3, 4))
        # At most 32 REs per channel: diagnostic only, not a full-grid SVD.
        flat_h = h_eff.reshape(h_eff.shape[0], -1, h_eff.shape[-2], h_eff.shape[-1])
        indices = torch.linspace(0, flat_h.shape[1] - 1, min(32, flat_h.shape[1]),
                                 device=h_eff.device).long()
        singular = torch.linalg.svdvals(flat_h[:, indices])
        condition = singular[..., 0] / singular[..., -1]

        def values(tensor):
            return tensor.detach().cpu().tolist()

        max_condition = float(condition.max().item())
        return dict(
            seed=sample["seed"], ebno_db=sample["ebno_db"], n0=no,
            tx_power_per_ue=values(tx_power), aggregate_tx_power=values(tx_power.sum(-1)),
            tx_antenna_sum_power_per_ue=values(sample["x_ant"].abs().square().sum(2).mean((2, 3))),
            raw_channel_power_per_link=values(sample["h_raw"].abs().square().mean((2, 4, 5, 6))[:, 0]),
            effective_channel_power_per_ue=values(h_eff.abs().square().mean((1, 2, 3))),
            received_signal_power=values(signal_power), measured_noise_power=values(noise_power),
            measured_noise_over_n0=values(noise_power / no),
            measured_rx_snr_db=values(10 * torch.log10(signal_power / noise_power)),
            reconstruction_relative_rms=float(((reconstructed - clean).abs().square().mean()
                                                / clean.abs().square().mean().clamp_min(1e-30)).sqrt().item()),
            sampled_re_count=int(indices.numel()),
            sampled_min_singular_value=float(singular[..., -1].min().item()),
            sampled_max_condition_number=max_condition if math.isfinite(max_condition) else None,
            singular_channel_seen=not math.isfinite(max_condition))


def compare_classical_lmmse(link, sample, received, num_re=16):
    """Small diagnostic of the existing mathematical LMMSE core against Sionna."""
    import torch
    from detectors.classical.lmmse import LMMSESoftDetector

    if num_re < 1:
        raise ValueError("num_re must be positive.")
    cfg = copy.deepcopy(link.cfg)
    cfg["modulation"] = {"bits_per_symbol": link.codec.qm}
    detector = LMMSESoftDetector(cfg)
    b = sample["y"].shape[0]
    m, k = link.num_rx, link.num_ues
    count = min(num_re, link.grid_metadata["num_data_symbols"])
    y = sample["y"][:, 0].permute(0, 2, 3, 1).reshape(b, 1, -1, m)[:, :, :count]
    h = sample["h_eff"].reshape(b, 1, -1, m, k)[:, :, :count]
    ruu = sample["no"] * torch.eye(m, dtype=h.dtype, device=h.device).expand(b, m, m)
    with torch.no_grad():
        actual = detector(y, h, ruu)
        expected = {
            "x_hat": received["x_hat"][:, :, 0, :count].permute(0, 2, 1).unsqueeze(1),
            "no_eff": received["no_eff"][:, :, 0, :count].permute(0, 2, 1).unsqueeze(1),
            "llr": received["llr"].reshape(b, k, -1, link.codec.qm)[:, :, :count].permute(0, 2, 1, 3).unsqueeze(1),
        }
        errors = {}
        for name, target in expected.items():
            delta = actual[name] - target
            errors[name + "_relative_rms"] = float((delta.abs().square().mean()
                / target.abs().square().mean().clamp_min(1e-30)).sqrt().item())
        return dict(num_re=count, **errors)
