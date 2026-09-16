#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from detectors.classical.lmmse import LMMSESoftDetector
from models.graph.csi_robust_gt import CSIRobustGraphTransformer
from models.graph.gt_ep_detector import GraphTransformerEPDetector


METHODS = [
    "lmmse",
    "ep5",
    "crgt_ep5",
    "gt_ep",
    "trueH_estR_ep5",
    "trueH_trueR_ep5",
]


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


def fixed_broken(reference_llr, method_llr, bits):
    truth = bits > 0.5

    reference_wrong = (reference_llr > 0) != truth
    method_wrong = (method_llr > 0) != truth

    fixed = (reference_wrong & ~method_wrong).sum().item()
    broken = (~reference_wrong & method_wrong).sum().item()

    return fixed, broken


def paired_bootstrap(rows, baseline_key, method_key, num_bootstrap=10000, seed=12345):
    baseline = torch.tensor([row[baseline_key] for row in rows], dtype=torch.float64)
    method = torch.tensor([row[method_key] for row in rows], dtype=torch.float64)

    diff = baseline - method
    mean_diff = diff.mean().item()

    if len(rows) < 2:
        return {
            "delta_ber": mean_diff,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    generator = torch.Generator().manual_seed(seed)

    n = len(rows)
    samples = []

    remaining = int(num_bootstrap)

    while remaining > 0:
        current = min(1000, remaining)
        idx = torch.randint(0, n, (current, n), generator=generator)
        samples.append(diff[idx].mean(dim=1))
        remaining -= current

    samples = torch.cat(samples)

    return {
        "delta_ber": mean_diff,
        "ci95_low": torch.quantile(samples, 0.025).item(),
        "ci95_high": torch.quantile(samples, 0.975).item(),
    }


def nmse_db(error_power, reference_power):
    value = error_power / max(reference_power, 1e-30)
    return 10.0 * math.log10(max(value, 1e-30))


def build_models(cfg, gt_ep_checkpoint, crgt_checkpoint):
    device = cfg["general"]["device"]

    lmmse = LMMSESoftDetector(cfg)

    ep = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    gt_ep = GraphTransformerEPDetector(
        cfg=cfg,
        num_users=16,
        num_iterations=5,
        damping=0.5,
        d_model=128,
        num_heads=8,
        num_layers=4,
        edge_dim=32,
        ffn_dim=256,
        dropout=0.05,
        max_logit_correction=4.0,
    ).to(device)

    gt_ep_ckpt = torch.load(
        gt_ep_checkpoint,
        map_location=device,
        weights_only=False,
    )

    gt_ep.load_state_dict(gt_ep_ckpt["model_state"])
    gt_ep.eval()

    crgt = CSIRobustGraphTransformer(
        num_rx=256,
        num_users=16,
        d_model=128,
        num_heads=8,
        num_layers=8,
        edge_dim=32,
        ffn_dim=256,
        dropout=0.05,
        correction_scale=1.0,
    ).to(device)

    crgt_ckpt = torch.load(
        crgt_checkpoint,
        map_location=device,
        weights_only=False,
    )

    crgt.load_state_dict(crgt_ckpt["model_state"])
    crgt.eval()

    return {
        "lmmse": lmmse,
        "ep": ep,
        "gt_ep": gt_ep,
        "crgt": crgt,
        "gt_ep_ckpt": gt_ep_ckpt,
        "crgt_ckpt": crgt_ckpt,
    }


@torch.no_grad()
def evaluate_snr(
    base_cfg,
    snr_db,
    models,
    channels,
    re_per_channel,
    eval_seed,
    bootstrap_samples,
    progress_every,
):
    cfg = base_cfg.copy()
    cfg["link"] = dict(base_cfg["link"])
    cfg["link"]["rx_snr_db"] = float(snr_db)

    device = cfg["general"]["device"]

    # Reset to the same seed at every SNR so that propagation/topology/data
    # realizations are matched as closely as possible across the SNR sweep.
    sionna_config.seed = int(eval_seed)
    torch.manual_seed(int(eval_seed))

    dataset = UplinkUMADataset(cfg)

    lmmse = models["lmmse"]
    ep = models["ep"]
    gt_ep = models["gt_ep"]
    crgt = models["crgt"]

    total_errors = {name: 0 for name in METHODS}
    total_bits = 0

    gt_fixed_ep5 = 0
    gt_broken_ep5 = 0

    gt_fixed_crgt = 0
    gt_broken_crgt = 0

    ep_valid_sum = torch.zeros(5, dtype=torch.float64)
    gt_valid_sum = torch.zeros(5, dtype=torch.float64)
    gt_corr_sum = torch.zeros(5, dtype=torch.float64)

    ls_error_power = 0.0
    true_power = 0.0

    rows = []

    start_time = time.time()

    for channel_idx in range(1, channels + 1):
        batch = dataset.sample(1)

        data_idx = list(dataset.channel.resource_grid.data_symbols)

        y_full = batch["y"][:, data_idx]
        h_ls_full = batch["h_hat_ls"][:, data_idx]
        h_true_full = batch["h_true"][:, data_idx]

        bits_full = batch["bits"].reshape(-1, 16, 4).float()

        num_re_total = bits_full.shape[0]
        count = min(int(re_per_channel), num_re_total)

        idx = torch.randperm(
            num_re_total,
            device=bits_full.device,
        )[:count]

        bits = bits_full[idx]

        y = y_full.reshape(-1, 256)[idx]
        h_ls = h_ls_full.reshape(-1, 256, 16)[idx]
        h_true = h_true_full.reshape(-1, 256, 16)[idx]

        ruu_hat = batch["ruu_hat"]
        ruu_true = batch["ruu_true"]

        # LMMSE baseline
        llr_lmmse = lmmse(
            y_full,
            h_ls_full,
            ruu_hat,
        )["llr"].reshape(-1, 16, 4)[idx].float()

        # Estimated-R whitening
        y_est_white, h_ls_est_white = whiten(
            y,
            h_ls,
            ruu_hat,
        )

        _, h_true_est_white = whiten(
            y,
            h_true,
            ruu_hat,
        )

        # True-R oracle whitening
        y_true_white, h_true_true_white = whiten(
            y,
            h_true,
            ruu_true,
        )

        # LS-H + Est-R statistics
        z_ls, gram_ls = sufficient_statistics(
            y_est_white,
            h_ls_est_white,
        )

        # True-H + Est-R statistics
        z_true_est, gram_true_est = sufficient_statistics(
            y_est_white,
            h_true_est_white,
        )

        # True-H + True-R statistics
        z_true_true, gram_true_true = sufficient_statistics(
            y_true_white,
            h_true_true_white,
        )

        # Classical EP5
        ep_out = ep(
            z_ls,
            gram_ls,
            return_iterations=(5,),
        )

        llr_ep5 = ep_out["llr"].float()

        # CSI-refinement Graph Transformer + EP5
        crgt_out = crgt(
            y_est_white,
            h_ls_est_white,
        )

        crgt_ep_out = ep(
            crgt_out["z_refined"],
            crgt_out["gram_refined"],
            return_iterations=(5,),
        )

        llr_crgt = crgt_ep_out["llr"].float()

        # Detection-native GT-EP
        gt_out = gt_ep(
            z_ls,
            gram_ls,
            return_iterations=(5,),
        )

        llr_gt = gt_out["llr"].float()

        # Oracle-H + Est-R EP5
        true_est_out = ep(
            z_true_est,
            gram_true_est,
            return_iterations=(5,),
        )

        llr_true_est = true_est_out["llr"].float()

        # Oracle-H + Oracle-R EP5
        true_true_out = ep(
            z_true_true,
            gram_true_true,
            return_iterations=(5,),
        )

        llr_true_true = true_true_out["llr"].float()

        outputs = {
            "lmmse": llr_lmmse,
            "ep5": llr_ep5,
            "crgt_ep5": llr_crgt,
            "gt_ep": llr_gt,
            "trueH_estR_ep5": llr_true_est,
            "trueH_trueR_ep5": llr_true_true,
        }

        num_bits = bits.numel()
        total_bits += num_bits

        row = {
            "channel": channel_idx,
            "num_bits": num_bits,
        }

        for name, llr in outputs.items():
            err = hard_errors(
                llr,
                bits,
            )

            total_errors[name] += err

            row[name] = err / num_bits
            row[f"{name}_errors"] = err

        fixed, broken = fixed_broken(
            llr_ep5,
            llr_gt,
            bits,
        )

        gt_fixed_ep5 += fixed
        gt_broken_ep5 += broken

        fixed, broken = fixed_broken(
            llr_crgt,
            llr_gt,
            bits,
        )

        gt_fixed_crgt += fixed
        gt_broken_crgt += broken

        ep_valid_sum += torch.tensor(
            ep_out["valid_update_fraction"],
            dtype=torch.float64,
        )

        gt_valid_sum += gt_out[
            "valid_update_fraction"
        ].double().cpu()

        gt_corr_sum += gt_out[
            "correction_rms"
        ].double().cpu()

        ls_error_power += (
            h_ls_est_white - h_true_est_white
        ).abs().square().sum().item()

        true_power += (
            h_true_est_white
        ).abs().square().sum().item()

        row["gt_vs_ep5_delta_ber"] = (
            row["ep5"] - row["gt_ep"]
        )

        row["gt_vs_crgt_delta_ber"] = (
            row["crgt_ep5"] - row["gt_ep"]
        )

        rows.append(row)

        if (
            channel_idx == 1
            or channel_idx % progress_every == 0
            or channel_idx == channels
        ):
            current_ber = {
                name: total_errors[name] / total_bits
                for name in METHODS
            }

            elapsed = time.time() - start_time
            rate = channel_idx / max(elapsed, 1e-9)
            eta = (
                channels - channel_idx
            ) / max(rate, 1e-9)

            print(
                f"SNR={snr_db:>4.1f} dB | "
                f"{channel_idx:>4d}/{channels} | "
                f"LMMSE={current_ber['lmmse']:.6e} | "
                f"EP5={current_ber['ep5']:.6e} | "
                f"CRGT={current_ber['crgt_ep5']:.6e} | "
                f"GT-EP={current_ber['gt_ep']:.6e} | "
                f"TrueH={current_ber['trueH_estR_ep5']:.6e} | "
                f"ETA={eta / 60.0:.1f}m"
            )

    ber = {
        name: total_errors[name] / total_bits
        for name in METHODS
    }

    gain_gt_vs_lmmse = (
        ber["lmmse"] - ber["gt_ep"]
    ) / max(ber["lmmse"], 1e-30)

    gain_gt_vs_ep5 = (
        ber["ep5"] - ber["gt_ep"]
    ) / max(ber["ep5"], 1e-30)

    gain_gt_vs_crgt = (
        ber["crgt_ep5"] - ber["gt_ep"]
    ) / max(ber["crgt_ep5"], 1e-30)

    gain_crgt_vs_ep5 = (
        ber["ep5"] - ber["crgt_ep5"]
    ) / max(ber["ep5"], 1e-30)

    gain_ep5_vs_lmmse = (
        ber["lmmse"] - ber["ep5"]
    ) / max(ber["lmmse"], 1e-30)

    oracle_h_gap = (
        ber["ep5"]
        - ber["trueH_estR_ep5"]
    )

    oracle_total_gap = (
        ber["ep5"]
        - ber["trueH_trueR_ep5"]
    )

    gt_h_gap_recovered = (
        ber["ep5"] - ber["gt_ep"]
    ) / max(oracle_h_gap, 1e-30)

    gt_total_gap_recovered = (
        ber["ep5"] - ber["gt_ep"]
    ) / max(oracle_total_gap, 1e-30)

    bootstrap = {
        "gt_ep_vs_ep5": paired_bootstrap(
            rows,
            "ep5",
            "gt_ep",
            num_bootstrap=bootstrap_samples,
            seed=eval_seed + 101,
        ),
        "gt_ep_vs_crgt": paired_bootstrap(
            rows,
            "crgt_ep5",
            "gt_ep",
            num_bootstrap=bootstrap_samples,
            seed=eval_seed + 102,
        ),
        "gt_ep_vs_lmmse": paired_bootstrap(
            rows,
            "lmmse",
            "gt_ep",
            num_bootstrap=bootstrap_samples,
            seed=eval_seed + 103,
        ),
        "crgt_vs_ep5": paired_bootstrap(
            rows,
            "ep5",
            "crgt_ep5",
            num_bootstrap=bootstrap_samples,
            seed=eval_seed + 104,
        ),
    }

    ep_valid = (
        ep_valid_sum / channels
    ).tolist()

    gt_valid = (
        gt_valid_sum / channels
    ).tolist()

    gt_corr = (
        gt_corr_sum / channels
    ).tolist()

    elapsed = time.time() - start_time

    return {
        "snr_db": float(snr_db),
        "channels": channels,
        "re_per_channel": re_per_channel,
        "total_bits": total_bits,
        "ber": ber,
        "relative_gain_ep5_vs_lmmse": gain_ep5_vs_lmmse,
        "relative_gain_crgt_vs_ep5": gain_crgt_vs_ep5,
        "relative_gain_gt_ep_vs_lmmse": gain_gt_vs_lmmse,
        "relative_gain_gt_ep_vs_ep5": gain_gt_vs_ep5,
        "relative_gain_gt_ep_vs_crgt": gain_gt_vs_crgt,
        "oracle_H_gap_recovered_by_gt_ep": gt_h_gap_recovered,
        "oracle_total_gap_recovered_by_gt_ep": gt_total_gap_recovered,
        "gt_ep_fixed_ep5_bits": gt_fixed_ep5,
        "gt_ep_broken_ep5_bits": gt_broken_ep5,
        "gt_ep_net_fixed_ep5_bits": gt_fixed_ep5 - gt_broken_ep5,
        "gt_ep_fixed_crgt_bits": gt_fixed_crgt,
        "gt_ep_broken_crgt_bits": gt_broken_crgt,
        "gt_ep_net_fixed_crgt_bits": gt_fixed_crgt - gt_broken_crgt,
        "ls_whitened_nmse_db": nmse_db(
            ls_error_power,
            true_power,
        ),
        "ep5_valid_update_fraction": ep_valid,
        "gt_ep_valid_update_fraction": gt_valid,
        "gt_ep_correction_rms": gt_corr,
        "paired_channel_bootstrap": bootstrap,
        "runtime_minutes": elapsed / 60.0,
        "per_channel": rows,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--gt-ep-checkpoint",
        default="ckp/gt_ep_256rx_16ue_lsH_estR_5db/best.pth",
    )

    parser.add_argument(
        "--crgt-checkpoint",
        default="ckp/csi_robust_gt_256rx_16ue_lsH_estR_5db/best.pth",
    )

    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[0, 2, 4, 5, 6, 8, 10],
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--re-per-channel",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=20260910,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--output-dir",
        default="results/gt_ep_snr_sweep",
    )

    args = parser.parse_args()

    train_cfg = load_yaml(
        args.config
    )

    base_cfg = load_yaml(
        train_cfg["system_config"]
    )

    device = base_cfg[
        "general"
    ]["device"]

    precision = base_cfg[
        "general"
    ]["precision"]

    if (
        device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is configured but unavailable."
        )

    sionna_config.device = device
    sionna_config.precision = precision

    torch.set_float32_matmul_precision(
        "high"
    )

    # Build models once. SNR changes only affect dataset generation.
    models = build_models(
        base_cfg,
        args.gt_ep_checkpoint,
        args.crgt_checkpoint,
    )

    print("=" * 150)
    print("GT-EP SNR Generalization Sweep")
    print("=" * 150)
    print("System               : 256 Rx / 16 users / 16-QAM")
    print(f"GT-EP checkpoint     : {args.gt_ep_checkpoint}")
    print(f"GT-EP checkpoint step: {models['gt_ep_ckpt'].get('step', -1)}")
    print(f"CR-GT checkpoint     : {args.crgt_checkpoint}")
    print(f"CR-GT checkpoint step: {models['crgt_ckpt'].get('step', -1)}")
    print(f"Evaluation SNRs      : {args.snrs}")
    print(f"Channels / SNR       : {args.channels}")
    print(f"RE / channel         : {args.re_per_channel}")
    print(f"Bits / SNR           : {args.channels * args.re_per_channel * 16 * 4}")
    print(f"Matched eval seed    : {args.eval_seed}")
    print("=" * 150)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for snr_db in args.snrs:
        print()
        print("#" * 150)
        print(f"Evaluating SNR = {snr_db:.1f} dB")
        print("#" * 150)

        result = evaluate_snr(
            base_cfg=base_cfg,
            snr_db=snr_db,
            models=models,
            channels=args.channels,
            re_per_channel=args.re_per_channel,
            eval_seed=args.eval_seed,
            bootstrap_samples=args.bootstrap,
            progress_every=args.progress_every,
        )

        results.append(
            result
        )

        ber = result["ber"]
        ci = result[
            "paired_channel_bootstrap"
        ]["gt_ep_vs_ep5"]

        print()
        print(f"SNR {snr_db:.1f} dB summary")
        print("-" * 120)
        print(f"LMMSE               : {ber['lmmse']:.8e}")
        print(f"EP5                 : {ber['ep5']:.8e}")
        print(f"CR-GT + EP5         : {ber['crgt_ep5']:.8e}")
        print(f"GT-EP               : {ber['gt_ep']:.8e}")
        print(f"TrueH + EstR + EP5  : {ber['trueH_estR_ep5']:.8e}")
        print(f"TrueH + TrueR + EP5 : {ber['trueH_trueR_ep5']:.8e}")
        print()
        print(f"GT-EP vs LMMSE      : {100.0 * result['relative_gain_gt_ep_vs_lmmse']:+.3f}%")
        print(f"GT-EP vs EP5        : {100.0 * result['relative_gain_gt_ep_vs_ep5']:+.3f}%")
        print(f"GT-EP vs CR-GT      : {100.0 * result['relative_gain_gt_ep_vs_crgt']:+.3f}%")
        print(f"CR-GT vs EP5        : {100.0 * result['relative_gain_crgt_vs_ep5']:+.3f}%")
        print(f"Oracle-H recovered  : {100.0 * result['oracle_H_gap_recovered_by_gt_ep']:+.3f}%")
        print()
        print(
            "GT-EP vs EP5 CI     : "
            f"ΔBER={ci['delta_ber']:+.6e} "
            f"95%CI=[{ci['ci95_low']:+.6e}, {ci['ci95_high']:+.6e}]"
        )
        print(f"LS whitened NMSE    : {result['ls_whitened_nmse_db']:+.3f} dB")

        snr_tag = str(snr_db).replace(
            ".",
            "p",
        )

        per_snr_path = output_dir / (
            f"gt_ep_snr_{snr_tag}db.json"
        )

        with open(
            per_snr_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

    summary_rows = []

    for result in results:
        ber = result["ber"]
        ci = result[
            "paired_channel_bootstrap"
        ]["gt_ep_vs_ep5"]

        summary_rows.append({
            "snr_db": result["snr_db"],
            "lmmse": ber["lmmse"],
            "ep5": ber["ep5"],
            "crgt_ep5": ber["crgt_ep5"],
            "gt_ep": ber["gt_ep"],
            "trueH_estR_ep5": ber["trueH_estR_ep5"],
            "trueH_trueR_ep5": ber["trueH_trueR_ep5"],
            "gt_ep_vs_lmmse_percent": 100.0 * result["relative_gain_gt_ep_vs_lmmse"],
            "gt_ep_vs_ep5_percent": 100.0 * result["relative_gain_gt_ep_vs_ep5"],
            "gt_ep_vs_crgt_percent": 100.0 * result["relative_gain_gt_ep_vs_crgt"],
            "crgt_vs_ep5_percent": 100.0 * result["relative_gain_crgt_vs_ep5"],
            "oracle_H_gap_recovered_percent": 100.0 * result["oracle_H_gap_recovered_by_gt_ep"],
            "ls_whitened_nmse_db": result["ls_whitened_nmse_db"],
            "gt_ep_vs_ep5_delta_ber": ci["delta_ber"],
            "gt_ep_vs_ep5_ci95_low": ci["ci95_low"],
            "gt_ep_vs_ep5_ci95_high": ci["ci95_high"],
        })

    summary_json = output_dir / (
        "gt_ep_snr_sweep_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "config": args.config,
                "gt_ep_checkpoint": args.gt_ep_checkpoint,
                "gt_ep_checkpoint_step": models["gt_ep_ckpt"].get("step", -1),
                "crgt_checkpoint": args.crgt_checkpoint,
                "crgt_checkpoint_step": models["crgt_ckpt"].get("step", -1),
                "snrs": args.snrs,
                "channels_per_snr": args.channels,
                "re_per_channel": args.re_per_channel,
                "eval_seed": args.eval_seed,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )

    summary_csv = output_dir / (
        "gt_ep_snr_sweep_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 150)
    print("FINAL SNR SWEEP SUMMARY")
    print("=" * 150)
    print(
        f"{'SNR':>5s} | "
        f"{'LMMSE':>10s} | "
        f"{'EP5':>10s} | "
        f"{'CR-GT':>10s} | "
        f"{'GT-EP':>10s} | "
        f"{'GT/EP5':>9s} | "
        f"{'GT/CRGT':>9s} | "
        f"{'H-gap':>8s}"
    )
    print("-" * 150)

    for row in summary_rows:
        print(
            f"{row['snr_db']:5.1f} | "
            f"{row['lmmse']:10.6f} | "
            f"{row['ep5']:10.6f} | "
            f"{row['crgt_ep5']:10.6f} | "
            f"{row['gt_ep']:10.6f} | "
            f"{row['gt_ep_vs_ep5_percent']:+8.3f}% | "
            f"{row['gt_ep_vs_crgt_percent']:+8.3f}% | "
            f"{row['oracle_H_gap_recovered_percent']:+7.2f}%"
        )

    print("=" * 150)
    print(f"Summary JSON : {summary_json}")
    print(f"Summary CSV  : {summary_csv}")
    print("=" * 150)


if __name__ == "__main__":
    main()