#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from detectors.classical.lmmse import LMMSESoftDetector


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def whiten(y, h, ruu):
    chol = torch.linalg.cholesky(ruu)
    if chol.shape[0] == 1 and y.shape[0] > 1:
        chol = chol.expand(y.shape[0], -1, -1)

    y_white = torch.linalg.solve_triangular(chol, y.unsqueeze(-1), upper=False).squeeze(-1)
    h_white = torch.linalg.solve_triangular(chol, h, upper=False)
    return y_white, h_white


def sufficient_statistics(y_white, h_white):
    z = torch.einsum("bmk,bm->bk", h_white.conj(), y_white)
    gram = torch.einsum("bmk,bml->bkl", h_white.conj(), h_white)
    return z, gram


def hard_errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


def to_db(x):
    return 10.0 * math.log10(max(float(x), 1e-30))


def run_lmmse_detector(detector, y, h, ruu):
    """
    LMMSESoftDetector's CovarianceAwareFrontEnd expects grid-shaped inputs:
        y   [B,S,F,M]
        h   [B,S,F,M,K]
        ruu [B,M,M] or compatible

    Here every selected data RE is treated as an independent batch item with
    S=F=1, then the singleton grid dimensions are removed from the output.
    """
    y_grid = y[:, None, None, :]
    h_grid = h[:, None, None, :, :]
    out = detector(y_grid, h_grid, ruu)

    return {
        "llr": out["llr"][:, 0, 0],
        "x_hat": out["x_hat"][:, 0, 0],
        "no_eff": out["no_eff"][:, 0, 0],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--re-per-channel", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])
    cfg["link"]["rx_snr_db"] = float(args.snr_db)
    cfg["general"]["seed"] = int(args.seed)

    device = cfg["general"]["device"]
    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(args.seed)
    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    dataset = UplinkUMADataset(cfg)
    lmmse = LMMSESoftDetector(cfg)
    ep5 = ExpectationPropagationDetector(cfg, num_iterations=5, damping=0.5)

    nmse_num_ls = 0.0
    nmse_num_lmmse = 0.0
    nmse_den = 0.0

    total_bits = 0
    errors = {
        "lmmse_eq_lsH": 0,
        "lmmse_eq_lmmseH": 0,
        "lmmse_eq_trueH": 0,
        "ep5_lsH": 0,
        "ep5_lmmseH": 0,
        "ep5_trueH": 0,
    }

    print("=" * 112)
    print("CHANNEL-ESTIMATION SANITY | LS-linear vs 2D-LMMSE vs True-H")
    print("=" * 112)
    print("Detector covariance : estimated Ruu")
    print(f"SNR                 : {args.snr_db:.1f} dB")
    print(f"Independent set     : {args.channels} channels × {args.re_per_channel} data RE")
    print("=" * 112)

    for ch in range(args.channels):
        batch = dataset.sample(1)
        data_idx = list(dataset.channel.resource_grid.data_symbols)

        y_raw = batch["y"][:, data_idx]
        h_ls_raw = batch["h_hat_ls"][:, data_idx]
        h_lmmse_raw = batch["h_hat_lmmse"][:, data_idx]
        h_true_raw = batch["h_true"][:, data_idx]
        bits_raw = batch["bits"]

        num_rx = int(y_raw.shape[-1])
        num_streams = int(batch["metadata"]["num_streams"])
        bits_per_symbol = int(batch["metadata"]["bits_per_symbol"])

        y = y_raw.reshape(-1, num_rx)
        h_ls = h_ls_raw.reshape(-1, num_rx, num_streams)
        h_lmmse = h_lmmse_raw.reshape(-1, num_rx, num_streams)
        h_true = h_true_raw.reshape(-1, num_rx, num_streams)
        bits = bits_raw.reshape(-1, num_streams, bits_per_symbol).float()

        if not (y.shape[0] == h_ls.shape[0] == h_lmmse.shape[0] == h_true.shape[0] == bits.shape[0]):
            raise RuntimeError(
                f"RE count mismatch: y={y.shape[0]}, h_ls={h_ls.shape[0]}, "
                f"h_lmmse={h_lmmse.shape[0]}, h_true={h_true.shape[0]}, bits={bits.shape[0]}"
            )

        count = min(int(args.re_per_channel), y.shape[0])
        idx = torch.randperm(y.shape[0], device=y.device)[:count]

        y = y[idx]
        h_ls = h_ls[idx]
        h_lmmse = h_lmmse[idx]
        h_true = h_true[idx]
        bits = bits[idx]

        nmse_num_ls += (h_ls - h_true).abs().square().sum().item()
        nmse_num_lmmse += (h_lmmse - h_true).abs().square().sum().item()
        nmse_den += h_true.abs().square().sum().item()

        ruu = batch["ruu_hat"]
        if ruu.shape[0] == 1 and y.shape[0] > 1:
            ruu_eval = ruu.expand(y.shape[0], -1, -1)
        else:
            ruu_eval = ruu

        out_ls = run_lmmse_detector(lmmse, y, h_ls, ruu_eval)
        out_lmmse = run_lmmse_detector(lmmse, y, h_lmmse, ruu_eval)
        out_true = run_lmmse_detector(lmmse, y, h_true, ruu_eval)

        errors["lmmse_eq_lsH"] += hard_errors(out_ls["llr"], bits)
        errors["lmmse_eq_lmmseH"] += hard_errors(out_lmmse["llr"], bits)
        errors["lmmse_eq_trueH"] += hard_errors(out_true["llr"], bits)

        y_white, h_ls_white = whiten(y, h_ls, ruu_eval)
        _, h_lmmse_white = whiten(y, h_lmmse, ruu_eval)
        _, h_true_white = whiten(y, h_true, ruu_eval)

        z_ls, gram_ls = sufficient_statistics(y_white, h_ls_white)
        z_lmmse, gram_lmmse = sufficient_statistics(y_white, h_lmmse_white)
        z_true, gram_true = sufficient_statistics(y_white, h_true_white)

        errors["ep5_lsH"] += hard_errors(ep5(z_ls, gram_ls, return_iterations=(5,))["llr"], bits)
        errors["ep5_lmmseH"] += hard_errors(ep5(z_lmmse, gram_lmmse, return_iterations=(5,))["llr"], bits)
        errors["ep5_trueH"] += hard_errors(ep5(z_true, gram_true, return_iterations=(5,))["llr"], bits)

        total_bits += bits.numel()

        if (ch + 1) % 4 == 0 or ch + 1 == args.channels:
            print(
                f"channel={ch + 1:3d}/{args.channels} | "
                f"NMSE LS={to_db(nmse_num_ls / nmse_den):+.2f} dB | "
                f"LMMSE={to_db(nmse_num_lmmse / nmse_den):+.2f} dB | "
                f"EP5 LS={errors['ep5_lsH'] / total_bits:.5f} | "
                f"LMMSE-H={errors['ep5_lmmseH'] / total_bits:.5f} | "
                f"True-H={errors['ep5_trueH'] / total_bits:.5f}"
            )

    nmse_ls = nmse_num_ls / nmse_den
    nmse_lmmse = nmse_num_lmmse / nmse_den

    print()
    print("=" * 112)
    print("FINAL SANITY RESULT")
    print("=" * 112)
    print(f"LS-linear CSI NMSE          : {to_db(nmse_ls):+.4f} dB")
    print(f"2D-LMMSE CSI NMSE           : {to_db(nmse_lmmse):+.4f} dB")
    print(f"NMSE improvement            : {to_db(nmse_ls) - to_db(nmse_lmmse):+.4f} dB")
    print("-" * 112)
    print(f"LMMSE Eq | LS-H             : {errors['lmmse_eq_lsH'] / total_bits:.8e}")
    print(f"LMMSE Eq | LMMSE-H          : {errors['lmmse_eq_lmmseH'] / total_bits:.8e}")
    print(f"LMMSE Eq | True-H           : {errors['lmmse_eq_trueH'] / total_bits:.8e}")
    print("-" * 112)
    print(f"EP5      | LS-H             : {errors['ep5_lsH'] / total_bits:.8e}")
    print(f"EP5      | LMMSE-H          : {errors['ep5_lmmseH'] / total_bits:.8e}")
    print(f"EP5      | True-H           : {errors['ep5_trueH'] / total_bits:.8e}")
    print("-" * 112)

    nmse_ok = nmse_lmmse < nmse_ls
    lmmse_eq_ok = errors["lmmse_eq_lmmseH"] < errors["lmmse_eq_lsH"]
    ep_ok = errors["ep5_lmmseH"] < errors["ep5_lsH"]

    print(f"CHECK NMSE improvement      : {'PASS' if nmse_ok else 'FAIL'}")
    print(f"CHECK LMMSE-Eq BER improves : {'PASS' if lmmse_eq_ok else 'FAIL'}")
    print(f"CHECK EP5 BER improves      : {'PASS' if ep_ok else 'FAIL'}")
    print("=" * 112)

    if not (nmse_ok and lmmse_eq_ok and ep_ok):
        raise RuntimeError("LMMSE channel-estimation sanity check failed. Do not retrain neural detectors yet.")


if __name__ == "__main__":
    main()
