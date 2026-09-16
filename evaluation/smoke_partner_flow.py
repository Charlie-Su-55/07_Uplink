#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_platform_constellation(batch, qm):
    """
    Infer the exact platform constellation point/bit mapping from x_data + bits.

    Returns:
        points: [2**Qm] complex64
        bit_table: [2**Qm, Qm] float32
    """
    x = batch["x_data"].reshape(-1).to(torch.complex64)
    bits = batch["bits"].reshape(-1, qm).round().long()

    if x.shape[0] != bits.shape[0]:
        raise RuntimeError(
            f"x_data / bits mismatch: {x.shape[0]} vs {bits.shape[0]}"
        )

    weights = (
        2 ** torch.arange(qm - 1, -1, -1, device=bits.device)
    ).long()

    labels = (bits * weights).sum(-1)

    points = []
    bit_table = []

    for index in range(2 ** qm):
        mask = labels == index

        if not bool(mask.any()):
            raise RuntimeError(
                f"Constellation label {index} did not appear in this batch. "
                "Generate another batch and retry."
            )

        symbols = x[mask]
        point = symbols[0]

        max_error = (symbols - point).abs().max().item()

        if max_error > 1e-6:
            raise RuntimeError(
                f"Bit label {index} maps to multiple physical points, "
                f"max deviation={max_error:.3e}"
            )

        points.append(point)
        bit_table.append(bits[mask][0])

    points = torch.stack(points).to(torch.complex64)
    bit_table = torch.stack(bit_table).to(torch.float32)

    print("Platform constellation recovered:")
    print(f"  Qm             : {qm}")
    print(f"  points         : {tuple(points.shape)}")
    print(f"  bits           : {tuple(bit_table.shape)}")
    print(f"  average power  : {points.abs().square().mean().item():.6f}")

    return points, bit_table


def hard_ber(llr, bits):
    hard = llr > 0
    truth = bits > 0.5
    return (hard != truth).float().mean().item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--bundle",
        default="third_party/flow_partner",
    )

    parser.add_argument(
        "--qm",
        type=int,
        choices=[2, 4, 6],
        default=4,
    )

    parser.add_argument(
        "--snr-db",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()

    if not bundle.is_dir():
        raise FileNotFoundError(bundle)

    sys.path.insert(
        0,
        str(bundle),
    )

    from example_minimal_integration import PartnerFlowDetector

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])

    cfg["modulation"]["bits_per_symbol"] = int(args.qm)
    cfg["link"]["rx_snr_db"] = float(args.snr_db)
    cfg["general"]["seed"] = int(args.seed)
    cfg["general"]["device"] = args.device

    sionna_config.device = args.device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(args.seed)

    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    dataset = UplinkUMADataset(cfg)

    print("=" * 120)
    print("PARTNER FLOW INTEGRATION SMOKE TEST")
    print("=" * 120)
    print(f"Qm          : {args.qm}")
    print(f"SNR         : {args.snr_db:.1f} dB")
    print(f"Device      : {args.device}")
    print(f"Bundle      : {bundle}")
    print("=" * 120)

    batch = dataset.sample(1)

    data_idx = list(
        dataset.channel.resource_grid.data_symbols
    )

    y = (
        batch["y"][:, data_idx]
        .to(device=args.device, dtype=torch.complex64)
        .contiguous()
    )

    h = (
        batch["h_hat_lmmse"][:, data_idx]
        .to(device=args.device, dtype=torch.complex64)
        .contiguous()
    )

    ruu = (
        batch["ruu_hat"]
        .to(device=args.device)
        .contiguous()
    )

    bits = (
        batch["bits"]
        .to(device=args.device, dtype=torch.float32)
        .contiguous()
    )

    platform_points, platform_bits = (
        infer_platform_constellation(
            batch,
            args.qm,
        )
    )

    receiver = PartnerFlowDetector(
        Qm=args.qm,
        bundle_dir=bundle,
        device=args.device,
        llr_convention="log_p1_over_p0",
        seed=args.seed,
        constellation_points=platform_points,
        constellation_bits=platform_bits,
    )

    print()
    print("Input shapes:")
    print(f"  y    : {tuple(y.shape)} {y.dtype}")
    print(f"  h    : {tuple(h.shape)} {h.dtype}")
    print(f"  ruu  : {tuple(ruu.shape)} {ruu.dtype}")
    print(f"  bits : {tuple(bits.shape)}")

    expected_y = (1, 10, 192, 256)
    expected_h = (1, 10, 192, 256, 16)
    expected_bits = (
        1,
        10,
        192,
        16,
        args.qm,
    )

    if tuple(y.shape) != expected_y:
        raise RuntimeError(
            f"Unexpected y shape: {tuple(y.shape)}"
        )

    if tuple(h.shape) != expected_h:
        raise RuntimeError(
            f"Unexpected h shape: {tuple(h.shape)}"
        )

    if tuple(bits.shape) != expected_bits:
        raise RuntimeError(
            f"Unexpected bits shape: {tuple(bits.shape)}"
        )

    with torch.inference_mode():
        llr = receiver(
            y,
            h,
            ruu,
        )

    expected_llr = (
        1,
        10,
        192,
        16,
        args.qm,
    )

    if tuple(llr.shape) != expected_llr:
        raise RuntimeError(
            f"Unexpected LLR shape: {tuple(llr.shape)}"
        )

    if not bool(torch.isfinite(llr).all()):
        raise RuntimeError(
            "Flow output contains NaN/Inf"
        )

    ber = hard_ber(
        llr,
        bits,
    )

    print()
    print("=" * 120)
    print("SMOKE TEST RESULT")
    print("=" * 120)
    print(f"LLR shape       : {tuple(llr.shape)}")
    print(f"LLR dtype       : {llr.dtype}")
    print(f"LLR range       : [{llr.min().item():.4f}, {llr.max().item():.4f}]")
    print(f"Hard BER        : {ber:.8e}")
    print("LLR convention  : log P(bit=1) / P(bit=0)")
    print("Constellation   : platform mapping verified")
    print("=" * 120)


if __name__ == "__main__":
    main()