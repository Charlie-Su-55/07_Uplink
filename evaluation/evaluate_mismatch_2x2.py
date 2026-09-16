#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector


CASES = {
    "A_trueH_trueR": ("true", "true"),
    "B_trueH_hatR": ("true", "hat"),
    "C_lmmseH_trueR": ("lmmse", "true"),
    "D_lmmseH_hatR": ("lmmse", "hat"),
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def whiten(y, h, ruu):
    chol = torch.linalg.cholesky(ruu)

    if chol.shape[0] == 1 and y.shape[0] > 1:
        chol = chol.expand(y.shape[0], -1, -1)

    y_white = torch.linalg.solve_triangular(
        chol,
        y.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)

    h_white = torch.linalg.solve_triangular(
        chol,
        h,
        upper=False,
    )

    return y_white, h_white


def sufficient_statistics(y_white, h_white):
    z = torch.einsum(
        "bmk,bm->bk",
        h_white.conj(),
        y_white,
    )

    gram = torch.einsum(
        "bmk,bml->bkl",
        h_white.conj(),
        h_white,
    )

    return z, gram


def hard_errors(llr, bits):
    return (
        (llr > 0)
        != (bits > 0.5)
    ).sum().item()


@torch.no_grad()
def evaluate_case(
    ep5,
    y,
    h,
    ruu,
    bits,
    chunk_size,
):
    total_errors = 0
    total_bits = 0

    for start in range(
        0,
        y.shape[0],
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            y.shape[0],
        )

        yc = y[start:end]
        hc = h[start:end]

        if ruu.shape[0] == 1:
            rc = ruu
        else:
            rc = ruu[start:end]

        y_white, h_white = whiten(
            yc,
            hc,
            rc,
        )

        z, gram = sufficient_statistics(
            y_white,
            h_white,
        )

        out = ep5(
            z,
            gram,
            return_iterations=(5,),
        )

        bc = bits[start:end]

        total_errors += hard_errors(
            out["llr"].float(),
            bc,
        )

        total_bits += bc.numel()

    return total_errors, total_bits


@torch.no_grad()
def run_point(
    train_cfg,
    bits_per_symbol,
    snr_db,
    channels,
    re_per_channel,
    chunk_size,
    seed,
):
    cfg = load_yaml(
        train_cfg["system_config"]
    )

    cfg["modulation"]["bits_per_symbol"] = (
        int(bits_per_symbol)
    )

    cfg["link"]["rx_snr_db"] = (
        float(snr_db)
    )

    cfg["general"]["seed"] = int(seed)

    device = cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = (
        cfg["general"]["precision"]
    )

    sionna_config.seed = int(seed)
    torch.manual_seed(int(seed))

    dataset = UplinkUMADataset(cfg)

    ep5 = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    errors = {
        key: 0
        for key in CASES
    }

    total_bits = 0

    channel_rows = []

    h_nmse_num = 0.0
    h_nmse_den = 0.0

    r_nmse_num = 0.0
    r_nmse_den = 0.0

    mod_name = {
        2: "QPSK",
        4: "16QAM",
        6: "64QAM",
    }[int(bits_per_symbol)]

    print()
    print("=" * 132)
    print(
        f"{mod_name} @ {snr_db:.1f} dB"
    )
    print("=" * 132)

    for ch in range(channels):
        batch = dataset.sample(1)

        data_idx = list(
            dataset.channel
            .resource_grid
            .data_symbols
        )

        num_streams = int(
            batch["metadata"]["num_streams"]
        )

        qm = int(
            batch["metadata"]["bits_per_symbol"]
        )

        y_raw = batch["y"][:, data_idx]
        h_true_raw = (
            batch["h_true"][:, data_idx]
        )

        h_lmmse_raw = (
            batch["h_hat_lmmse"][:, data_idx]
        )

        bits_raw = batch["bits"]

        num_rx = int(
            y_raw.shape[-1]
        )

        y = y_raw.reshape(
            -1,
            num_rx,
        )

        h_true = h_true_raw.reshape(
            -1,
            num_rx,
            num_streams,
        )

        h_lmmse = h_lmmse_raw.reshape(
            -1,
            num_rx,
            num_streams,
        )

        bits = bits_raw.reshape(
            -1,
            num_streams,
            qm,
        ).float()

        count = min(
            int(re_per_channel),
            y.shape[0],
        )

        # Same selected REs for all four cases.
        idx = torch.randperm(
            y.shape[0],
            device=y.device,
        )[:count]

        y = y[idx]
        h_true = h_true[idx]
        h_lmmse = h_lmmse[idx]
        bits = bits[idx]

        ruu_true = batch["ruu_true"]
        ruu_hat = batch["ruu_hat"]

        # Diagnostics.
        h_nmse_num += (
            h_lmmse - h_true
        ).abs().square().sum().item()

        h_nmse_den += (
            h_true
        ).abs().square().sum().item()

        r_nmse_num += (
            ruu_hat - ruu_true
        ).abs().square().sum().item()

        r_nmse_den += (
            ruu_true
        ).abs().square().sum().item()

        channel_result = {
            "channel": ch,
        }

        for case_name, (
            h_mode,
            r_mode,
        ) in CASES.items():

            h = (
                h_true
                if h_mode == "true"
                else h_lmmse
            )

            ruu = (
                ruu_true
                if r_mode == "true"
                else ruu_hat
            )

            e, n = evaluate_case(
                ep5,
                y,
                h,
                ruu,
                bits,
                chunk_size,
            )

            errors[case_name] += e

            channel_result[
                case_name + "_ber"
            ] = e / n

        total_bits += bits.numel()
        channel_rows.append(
            channel_result
        )

        if (
            (ch + 1) % 8 == 0
            or ch + 1 == channels
        ):
            print(
                f"  channel "
                f"{ch + 1}/{channels}"
            )

    ber = {
        key: errors[key] / total_bits
        for key in CASES
    }

    h_nmse = (
        h_nmse_num
        / max(h_nmse_den, 1e-30)
    )

    r_nmse = (
        r_nmse_num
        / max(r_nmse_den, 1e-30)
    )

    a = ber["A_trueH_trueR"]

    print()
    print(
        " " * 24
        + "True Ruu"
        + " " * 13
        + "Estimated Ruu"
    )

    print(
        f"{'True H':<20}"
        f"{ber['A_trueH_trueR']:>14.6e}"
        f"{ber['B_trueH_hatR']:>22.6e}"
    )

    print(
        f"{'LMMSE H':<20}"
        f"{ber['C_lmmseH_trueR']:>14.6e}"
        f"{ber['D_lmmseH_hatR']:>22.6e}"
    )

    print()
    print(
        f"H LMMSE NMSE          : "
        f"{10.0 * torch.log10(torch.tensor(h_nmse)).item():+.3f} dB"
    )

    print(
        f"Ruu estimate NMSE     : "
        f"{10.0 * torch.log10(torch.tensor(r_nmse)).item():+.3f} dB"
    )

    print()
    print("Penalty relative to full oracle:")

    for key in CASES:
        delta = ber[key] - a

        relative = (
            0.0
            if a <= 0
            else 100.0 * delta / a
        )

        print(
            f"  {key:<20}: "
            f"BER={ber[key]:.6e} | "
            f"Δ={delta:+.6e} | "
            f"relative={relative:+.2f}%"
        )

    return {
        "modulation": mod_name,
        "bits_per_symbol": bits_per_symbol,
        "snr_db": snr_db,
        "channels": channels,
        "re_per_channel": re_per_channel,
        "total_bits": total_bits,
        "errors": errors,
        "ber": ber,
        "h_lmmse_nmse": h_nmse,
        "ruu_hat_nmse": r_nmse,
        "channel_rows": channel_rows,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--re-per-channel",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        default="results/mismatch_2x2",
    )

    args = parser.parse_args()

    train_cfg = load_yaml(args.config)

    torch.set_float32_matmul_precision(
        "high"
    )

    print("=" * 132)
    print(
        "2x2 RECEIVER-MISMATCH "
        "DECOMPOSITION | EP5"
    )
    print("=" * 132)

    print(
        "A : True H  + True Ruu"
    )

    print(
        "B : True H  + Estimated Ruu"
    )

    print(
        "C : LMMSE H + True Ruu"
    )

    print(
        "D : LMMSE H + Estimated Ruu"
    )

    print(
        f"Channels / point     : "
        f"{args.channels}"
    )

    print(
        f"RE / channel         : "
        f"{args.re_per_channel}"
    )

    print(
        "Paired comparison    : "
        "same y / bits / RE indices "
        "for all four cases"
    )

    print("=" * 132)

    # Point 1:
    # deployment region where neural refinement
    # previously gave ~7% BER gain.
    result_16 = run_point(
        train_cfg=train_cfg,
        bits_per_symbol=4,
        snr_db=8.0,
        channels=args.channels,
        re_per_channel=args.re_per_channel,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    # Point 2:
    # high-order-modulation deployment region.
    result_64 = run_point(
        train_cfg=train_cfg,
        bits_per_symbol=6,
        snr_db=17.0,
        channels=args.channels,
        re_per_channel=args.re_per_channel,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        out_dir
        / "mismatch_2x2.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "args": vars(args),
                "results": [
                    result_16,
                    result_64,
                ],
            },
            f,
            indent=2,
        )

    csv_path = (
        out_dir
        / "mismatch_2x2_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "modulation",
            "snr_db",
            "case",
            "ber",
            "errors",
            "total_bits",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in [
            result_16,
            result_64,
        ]:
            for case in CASES:
                writer.writerow({
                    "modulation": (
                        result["modulation"]
                    ),
                    "snr_db": (
                        result["snr_db"]
                    ),
                    "case": case,
                    "ber": (
                        result["ber"][case]
                    ),
                    "errors": (
                        result["errors"][case]
                    ),
                    "total_bits": (
                        result["total_bits"]
                    ),
                })

    print()
    print("=" * 132)
    print("2x2 DECOMPOSITION FINISHED")
    print("=" * 132)
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")
    print("=" * 132)


if __name__ == "__main__":
    main()
