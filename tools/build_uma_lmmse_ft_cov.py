#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.channels.uma import UMAChannelProvider
from data.link.stream_mapping import RankOneStreamMapper
from data.link.power_control import FractionalPowerController


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_hermitian_psd(r):
    r = 0.5 * (r + r.mH)
    eigval, eigvec = torch.linalg.eigh(r)
    floor = torch.finfo(r.real.dtype).eps * eigval.abs().max().clamp_min(1e-30)
    eigval = eigval.clamp_min(floor)
    r = (eigvec * eigval.unsqueeze(0)) @ eigvec.mH
    return 0.5 * (r + r.mH)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--output", default="data/cache/uma_lmmse_ft_cov.pt")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])

    device = cfg["general"]["device"]
    precision = cfg["general"]["precision"]

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = int(args.seed)
    torch.manual_seed(int(args.seed))

    channel = UMAChannelProvider(cfg)
    mapper = RankOneStreamMapper(
        num_tx_antennas=channel.ut_array.num_ant,
        mode=cfg["stream_mapping"]["mode"],
        device=device,
        precision=precision,
    )
    power_controller = FractionalPowerController(cfg)

    rf_sum = None
    rt_sum = None
    rf_count = 0
    rt_count = 0

    for n in range(args.channels):
        h_raw, _ = channel.sample(1)
        h_propagation = mapper(h_raw)
        pc = power_controller(h_propagation)
        h = pc["h_powered"]

        if h.ndim != 5:
            raise RuntimeError(f"Expected powered effective H [B,S,F,M,K], got {tuple(h.shape)}")

        b, s, f, m, k = h.shape

        # Frequency covariance: each (batch, time, Rx antenna, stream) is one sample.
        xf = h.permute(0, 1, 3, 4, 2).contiguous().reshape(-1, f)
        rf_batch = xf.transpose(0, 1) @ xf.conj()
        rf_sum = rf_batch if rf_sum is None else rf_sum + rf_batch
        rf_count += xf.shape[0]

        # Time covariance: each (batch, subcarrier, Rx antenna, stream) is one sample.
        xt = h.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, s)
        rt_batch = xt.transpose(0, 1) @ xt.conj()
        rt_sum = rt_batch if rt_sum is None else rt_sum + rt_batch
        rt_count += xt.shape[0]

        if (n + 1) % 8 == 0 or n + 1 == args.channels:
            print(f"channel {n + 1}/{args.channels}")

    rf = make_hermitian_psd(rf_sum / float(rf_count))
    rt = make_hermitian_psd(rt_sum / float(rt_count))

    cdtype = torch.complex64 if precision == "single" else torch.complex128
    rf = rf.to(cdtype).cpu()
    rt = rt.to(cdtype).cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "freq_cov": rf,
            "time_cov": rt,
            "num_channels": int(args.channels),
            "seed": int(args.seed),
            "num_symbols": int(rt.shape[0]),
            "num_subcarriers": int(rf.shape[0]),
            "freq_avg_power": float(torch.diagonal(rf).real.mean()),
            "time_avg_power": float(torch.diagonal(rt).real.mean()),
            "source": "Independent UMa powered effective channels after RankOneStreamMapper and FractionalPowerController",
        },
        output,
    )

    print(f"Saved: {output}")
    print(f"freq_cov: {tuple(rf.shape)}")
    print(f"time_cov: {tuple(rt.shape)}")
    print(f"frequency diag mean: {torch.diagonal(rf).real.mean().item():.6e}")
    print(f"time diag mean     : {torch.diagonal(rt).real.mean().item():.6e}")
    print(f"max Hermitian error freq: {(rf-rf.mH).abs().max().item():.3e}")
    print(f"max Hermitian error time: {(rt-rt.mH).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
