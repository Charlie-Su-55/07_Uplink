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
from detectors.classical.lmmse import LMMSESoftDetector
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.csi_robust_gt import CSIRobustGraphTransformer


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sufficient_statistics(y_white, h_white):
    z = torch.einsum("bmk,bm->bk", h_white.conj(), y_white)
    gram = torch.einsum("bmk,bml->bkl", h_white.conj(), h_white)
    return z, gram


def whiten_selected(y, h, ruu):
    chol = torch.linalg.cholesky(ruu)

    if chol.shape[0] == 1 and y.shape[0] > 1:
        chol = chol.expand(y.shape[0], -1, -1)

    y_white = torch.linalg.solve_triangular(chol, y.unsqueeze(-1), upper=False).squeeze(-1)
    h_white = torch.linalg.solve_triangular(chol, h, upper=False)

    return y_white, h_white


def ep5_llr(ep, z, gram):
    out = ep(z, gram, return_iterations=(5,))
    return out["iterations"][5]["llr"].reshape(-1, 16, 4).float()


def hard_bits(llr):
    return llr > 0


def count_errors(llr, bits):
    return (hard_bits(llr) != (bits > 0.5)).sum().item()


def paired_fixed_broken(reference_llr, method_llr, bits):
    truth = bits > 0.5
    ref_wrong = hard_bits(reference_llr) != truth
    method_wrong = hard_bits(method_llr) != truth

    fixed = (ref_wrong & ~method_wrong).sum().item()
    broken = (~ref_wrong & method_wrong).sum().item()

    return fixed, broken


def paired_bootstrap(rows, baseline_key, method_key, num_bootstrap=10000, seed=12345):
    baseline = torch.tensor([row[baseline_key] for row in rows], dtype=torch.float64)
    method = torch.tensor([row[method_key] for row in rows], dtype=torch.float64)

    diff = baseline - method
    mean_diff = diff.mean().item()

    if len(rows) < 2:
        return mean_diff, float("nan"), float("nan")

    generator = torch.Generator()
    generator.manual_seed(seed)

    n = len(rows)
    samples = []
    remaining = num_bootstrap

    while remaining > 0:
        current = min(1000, remaining)
        idx = torch.randint(0, n, (current, n), generator=generator)
        samples.append(diff[idx].mean(dim=1))
        remaining -= current

    samples = torch.cat(samples)

    return (
        mean_diff,
        torch.quantile(samples, 0.025).item(),
        torch.quantile(samples, 0.975).item(),
    )


def nmse_db(error_power, reference_power):
    value = error_power / max(reference_power, 1e-30)
    return 10.0 * math.log10(max(value, 1e-30))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--checkpoint", default="ckp/csi_robust_gt_256rx_16ue_lsH_estR_5db/best.pth")
    parser.add_argument("--channels", type=int, default=500)
    parser.add_argument("--re-per-channel", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--eval-seed", type=int, default=20260910)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output-dir", default="results/csi_robust_gt_eval")

    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    system_cfg = load_yaml(train_cfg["system_config"])
    tcfg = train_cfg.get("training", {})

    snr_db = float(tcfg.get("snr_db", 5.0))
    system_cfg["link"]["rx_snr_db"] = snr_db

    device = system_cfg["general"]["device"]
    precision = system_cfg["general"]["precision"]

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but unavailable.")

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = int(args.eval_seed)

    torch.manual_seed(int(args.eval_seed))
    torch.set_float32_matmul_precision("high")

    dataset = UplinkUMADataset(system_cfg)

    model = CSIRobustGraphTransformer(
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    checkpoint_step = checkpoint.get("step", -1)
    checkpoint_metrics = checkpoint.get("metrics", {})

    lmmse = LMMSESoftDetector(system_cfg)
    ep = ExpectationPropagationDetector(system_cfg, num_iterations=5, damping=0.5)

    methods = [
        "ls_estR_lmmse",
        "ls_estR_ep5",
        "gt_estR_ep5",
        "trueH_estR_ep5",
        "trueH_trueR_ep5",
    ]

    totals = {key: 0 for key in methods}
    rows = []

    total_bits = 0

    gt_vs_ep5_fixed = 0
    gt_vs_ep5_broken = 0
    gt_vs_lmmse_fixed = 0
    gt_vs_lmmse_broken = 0

    h_ls_error = 0.0
    h_gt_error = 0.0
    h_true_power = 0.0

    delta_rms_sum = 0.0
    delta_rms_count = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"crgt_{args.channels}ch_{args.re_per_channel}re_{snr_db:g}db"
    json_path = output_dir / f"{tag}.json"
    csv_path = output_dir / f"{tag}_per_channel.csv"

    start_time = time.time()

    print("=" * 140)
    print("CSI-Robust Graph Transformer | Independent Long Evaluation")
    print("=" * 140)
    print(f"System               : 256 Rx / 16 users / 16-QAM")
    print(f"SNR                  : {snr_db:.1f} dB")
    print(f"Checkpoint           : {args.checkpoint}")
    print(f"Checkpoint step      : {checkpoint_step}")
    print(f"Independent channels : {args.channels}")
    print(f"Random RE/channel    : {args.re_per_channel}")
    print(f"Bits/channel         : {args.re_per_channel * 16 * 4}")
    print(f"Total planned bits   : {args.channels * args.re_per_channel * 16 * 4}")
    print(f"Evaluation seed      : {args.eval_seed}")
    print("Main comparison      : LS-H + Est-R LMMSE / EP5 / CR-GT+EP5")
    print("Oracle references    : TrueH+EstR EP5 / TrueH+TrueR EP5")
    print("=" * 140)

    with torch.no_grad():
        for channel_idx in range(1, args.channels + 1):
            batch = dataset.sample(1)

            data_idx = list(dataset.channel.resource_grid.data_symbols)

            y_full = batch["y"][:, data_idx]
            h_ls_full = batch["h_hat_ls"][:, data_idx]
            h_true_full = batch["h_true"][:, data_idx]

            bits_full = batch["bits"].reshape(-1, 16, 4).float()

            num_re_total = bits_full.shape[0]
            count = min(int(args.re_per_channel), num_re_total)

            idx = torch.randperm(num_re_total, device=bits_full.device)[:count]

            bits = bits_full[idx]

            y = y_full.reshape(-1, 256)[idx]
            h_ls = h_ls_full.reshape(-1, 256, 16)[idx]
            h_true = h_true_full.reshape(-1, 256, 16)[idx]

            ruu_hat = batch["ruu_hat"]
            ruu_true = batch["ruu_true"]

            lmmse_full = lmmse(y_full, h_ls_full, ruu_hat)["llr"].reshape(-1, 16, 4)
            llr_lmmse = lmmse_full[idx].float()

            y_est_white, h_ls_est_white = whiten_selected(y, h_ls, ruu_hat)
            _, h_true_est_white = whiten_selected(y, h_true, ruu_hat)

            y_true_white, h_true_true_white = whiten_selected(y, h_true, ruu_true)

            z_ls, gram_ls = sufficient_statistics(y_est_white, h_ls_est_white)
            z_true_est, gram_true_est = sufficient_statistics(y_est_white, h_true_est_white)
            z_true_true, gram_true_true = sufficient_statistics(y_true_white, h_true_true_white)

            gt_out = model(y_est_white, h_ls_est_white)

            llr_ep5 = ep5_llr(ep, z_ls, gram_ls)
            llr_gt = ep5_llr(ep, gt_out["z_refined"], gt_out["gram_refined"])
            llr_true_est = ep5_llr(ep, z_true_est, gram_true_est)
            llr_true_true = ep5_llr(ep, z_true_true, gram_true_true)

            outputs = {
                "ls_estR_lmmse": llr_lmmse,
                "ls_estR_ep5": llr_ep5,
                "gt_estR_ep5": llr_gt,
                "trueH_estR_ep5": llr_true_est,
                "trueH_trueR_ep5": llr_true_true,
            }

            num_bits = bits.numel()
            total_bits += num_bits

            row = {
                "channel": channel_idx,
                "num_bits": num_bits,
            }

            for key, llr in outputs.items():
                errors = count_errors(llr, bits)
                totals[key] += errors
                row[key] = errors / num_bits
                row[f"{key}_errors"] = errors

            fixed, broken = paired_fixed_broken(llr_ep5, llr_gt, bits)
            gt_vs_ep5_fixed += fixed
            gt_vs_ep5_broken += broken

            fixed, broken = paired_fixed_broken(llr_lmmse, llr_gt, bits)
            gt_vs_lmmse_fixed += fixed
            gt_vs_lmmse_broken += broken

            ls_error = (h_ls_est_white - h_true_est_white).abs().square().sum().item()
            gt_error = (gt_out["h_refined"] - h_true_est_white).abs().square().sum().item()
            true_power = h_true_est_white.abs().square().sum().item()

            h_ls_error += ls_error
            h_gt_error += gt_error
            h_true_power += true_power

            delta_rms_sum += gt_out["relative_delta_rms"].sum().item()
            delta_rms_count += gt_out["relative_delta_rms"].numel()

            row["ls_whitened_nmse"] = ls_error / max(true_power, 1e-30)
            row["gt_whitened_nmse"] = gt_error / max(true_power, 1e-30)
            row["gt_vs_ep5_delta_ber"] = row["ls_estR_ep5"] - row["gt_estR_ep5"]
            row["gt_vs_lmmse_delta_ber"] = row["ls_estR_lmmse"] - row["gt_estR_ep5"]

            rows.append(row)

            if channel_idx == 1 or channel_idx % args.progress_every == 0 or channel_idx == args.channels:
                elapsed = time.time() - start_time
                rate = channel_idx / max(elapsed, 1e-9)
                eta = (args.channels - channel_idx) / max(rate, 1e-9)

                print(
                    f"channel={channel_idx:>4d}/{args.channels} | "
                    f"LMMSE={totals['ls_estR_lmmse'] / total_bits:.6e} | "
                    f"EP5={totals['ls_estR_ep5'] / total_bits:.6e} | "
                    f"GT+EP5={totals['gt_estR_ep5'] / total_bits:.6e} | "
                    f"TrueH+EP5={totals['trueH_estR_ep5'] / total_bits:.6e} | "
                    f"elapsed={elapsed / 60:.1f}m | ETA={eta / 60:.1f}m"
                )

    ber = {key: totals[key] / total_bits for key in methods}

    gt_vs_lmmse_abs = ber["ls_estR_lmmse"] - ber["gt_estR_ep5"]
    gt_vs_ep5_abs = ber["ls_estR_ep5"] - ber["gt_estR_ep5"]
    ep5_vs_lmmse_abs = ber["ls_estR_lmmse"] - ber["ls_estR_ep5"]

    gt_vs_lmmse_rel = gt_vs_lmmse_abs / max(ber["ls_estR_lmmse"], 1e-30)
    gt_vs_ep5_rel = gt_vs_ep5_abs / max(ber["ls_estR_ep5"], 1e-30)
    ep5_vs_lmmse_rel = ep5_vs_lmmse_abs / max(ber["ls_estR_lmmse"], 1e-30)

    oracle_h_gap = ber["ls_estR_ep5"] - ber["trueH_estR_ep5"]
    oracle_total_gap = ber["ls_estR_ep5"] - ber["trueH_trueR_ep5"]

    recovered_h_gap = gt_vs_ep5_abs / max(oracle_h_gap, 1e-30)
    recovered_total_gap = gt_vs_ep5_abs / max(oracle_total_gap, 1e-30)

    ls_nmse_db = nmse_db(h_ls_error, h_true_power)
    gt_nmse_db = nmse_db(h_gt_error, h_true_power)

    bootstrap = {}

    comparisons = {
        "gt_vs_ep5": ("ls_estR_ep5", "gt_estR_ep5"),
        "gt_vs_lmmse": ("ls_estR_lmmse", "gt_estR_ep5"),
        "ep5_vs_lmmse": ("ls_estR_lmmse", "ls_estR_ep5"),
    }

    for name, (baseline_key, method_key) in comparisons.items():
        mean_diff, low, high = paired_bootstrap(
            rows,
            baseline_key,
            method_key,
            num_bootstrap=args.bootstrap,
            seed=args.eval_seed + 17,
        )

        bootstrap[name] = {
            "baseline_minus_method": mean_diff,
            "ci95_low": low,
            "ci95_high": high,
        }

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "checkpoint_metrics": checkpoint_metrics,
        "snr_db": snr_db,
        "channels": args.channels,
        "re_per_channel": args.re_per_channel,
        "total_bits": total_bits,
        "eval_seed": args.eval_seed,
        "ber": ber,
        "gt_vs_lmmse_absolute_ber_reduction": gt_vs_lmmse_abs,
        "gt_vs_lmmse_relative_ber_reduction": gt_vs_lmmse_rel,
        "gt_vs_ep5_absolute_ber_reduction": gt_vs_ep5_abs,
        "gt_vs_ep5_relative_ber_reduction": gt_vs_ep5_rel,
        "ep5_vs_lmmse_absolute_ber_reduction": ep5_vs_lmmse_abs,
        "ep5_vs_lmmse_relative_ber_reduction": ep5_vs_lmmse_rel,
        "oracle_H_gap_recovered_by_gt": recovered_h_gap,
        "oracle_total_gap_recovered_by_gt": recovered_total_gap,
        "ls_whitened_nmse_db": ls_nmse_db,
        "gt_whitened_nmse_db": gt_nmse_db,
        "relative_delta_rms": delta_rms_sum / max(delta_rms_count, 1),
        "gt_vs_ep5_fixed_bits": gt_vs_ep5_fixed,
        "gt_vs_ep5_broken_bits": gt_vs_ep5_broken,
        "gt_vs_ep5_net_fixed": gt_vs_ep5_fixed - gt_vs_ep5_broken,
        "gt_vs_lmmse_fixed_bits": gt_vs_lmmse_fixed,
        "gt_vs_lmmse_broken_bits": gt_vs_lmmse_broken,
        "gt_vs_lmmse_net_fixed": gt_vs_lmmse_fixed - gt_vs_lmmse_broken,
        "paired_channel_bootstrap": bootstrap,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "per_channel": rows,
        }, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start_time

    print()
    print("=" * 140)
    print("FINAL INDEPENDENT EVALUATION")
    print("=" * 140)

    print(f"LS-H + Est-R LMMSE       : {ber['ls_estR_lmmse']:.8e}")
    print(f"LS-H + Est-R EP5         : {ber['ls_estR_ep5']:.8e}")
    print(f"LS-H + Est-R CR-GT + EP5 : {ber['gt_estR_ep5']:.8e}")
    print(f"TrueH + Est-R EP5        : {ber['trueH_estR_ep5']:.8e}")
    print(f"TrueH + TrueR EP5        : {ber['trueH_trueR_ep5']:.8e}")

    print()
    print("GAIN OVER LMMSE")
    print("-" * 100)
    print(f"EP5 absolute BER reduction     : {ep5_vs_lmmse_abs:+.8e}")
    print(f"EP5 relative BER reduction     : {100.0 * ep5_vs_lmmse_rel:+.3f}%")
    print(f"CR-GT absolute BER reduction   : {gt_vs_lmmse_abs:+.8e}")
    print(f"CR-GT relative BER reduction   : {100.0 * gt_vs_lmmse_rel:+.3f}%")

    print()
    print("GAIN OVER EP5")
    print("-" * 100)
    print(f"CR-GT absolute BER reduction   : {gt_vs_ep5_abs:+.8e}")
    print(f"CR-GT relative BER reduction   : {100.0 * gt_vs_ep5_rel:+.3f}%")
    print(f"Fixed EP5-wrong bits           : {gt_vs_ep5_fixed}")
    print(f"Broken EP5-correct bits        : {gt_vs_ep5_broken}")
    print(f"Net fixed bits                 : {gt_vs_ep5_fixed - gt_vs_ep5_broken}")

    print()
    print("CSI / ORACLE HEADROOM")
    print("-" * 100)
    print(f"LS whitened CSI NMSE           : {ls_nmse_db:+.3f} dB")
    print(f"GT whitened CSI NMSE           : {gt_nmse_db:+.3f} dB")
    print(f"Mean relative ΔH RMS           : {delta_rms_sum / max(delta_rms_count, 1):.4f}")
    print(f"True-H oracle gap recovered    : {100.0 * recovered_h_gap:+.3f}%")
    print(f"Full oracle gap recovered      : {100.0 * recovered_total_gap:+.3f}%")

    print()
    print("PAIRED CHANNEL BOOTSTRAP")
    print("-" * 100)

    for name, item in bootstrap.items():
        print(
            f"{name:<20s} "
            f"ΔBER={item['baseline_minus_method']:+.6e} "
            f"95%CI=[{item['ci95_low']:+.6e}, {item['ci95_high']:+.6e}]"
        )

    print()
    print(f"Runtime : {elapsed / 60:.2f} min")
    print(f"JSON    : {json_path}")
    print(f"CSV     : {csv_path}")
    print("=" * 140)


if __name__ == "__main__":
    main()