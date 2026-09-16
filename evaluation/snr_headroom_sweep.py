import argparse
import copy
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


EP_STAGES = (5, 20)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bit_errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


def whiten(y, h, ruu):
    chol = torch.linalg.cholesky(ruu)
    a = chol[:, None, None]
    y_white = torch.linalg.solve_triangular(a, y.unsqueeze(-1), upper=False).squeeze(-1)
    h_white = torch.linalg.solve_triangular(a, h, upper=False)
    return y_white, h_white


def sufficient_statistics(y_white, h_white):
    z = torch.einsum("bmk,bm->bk", h_white.conj(), y_white)
    gram = torch.einsum("bmk,bml->bkl", h_white.conj(), h_white)
    return z, gram


def flatten_llr(llr):
    return llr.reshape(-1, 16, 4)


def get_ep_iterations(ep, z, gram):
    out = ep(z, gram, return_iterations=EP_STAGES)
    return {stage: out["iterations"][stage]["llr"].reshape(-1, 16, 4).float() for stage in EP_STAGES}


def bootstrap_mean_ci(values, num_bootstrap=10000, seed=12345):
    x = torch.tensor(values, dtype=torch.float64)

    if x.numel() < 2:
        mean = x.mean().item() if x.numel() else float("nan")
        return mean, float("nan"), float("nan")

    g = torch.Generator()
    g.manual_seed(seed)

    n = x.numel()
    chunk = 1000
    samples = []
    remaining = num_bootstrap

    while remaining > 0:
        current = min(chunk, remaining)
        idx = torch.randint(0, n, (current, n), generator=g)
        samples.append(x[idx].mean(dim=1))
        remaining -= current

    boot = torch.cat(samples)
    return x.mean().item(), torch.quantile(boot, 0.025).item(), torch.quantile(boot, 0.975).item()


def snr_at_target(curve, target_ber):
    points = sorted(curve, key=lambda x: x[0])
    target_log = math.log10(target_ber)

    for i in range(len(points) - 1):
        snr1, ber1 = points[i]
        snr2, ber2 = points[i + 1]

        if ber1 <= 0 or ber2 <= 0:
            continue

        log1 = math.log10(ber1)
        log2 = math.log10(ber2)

        if (log1 - target_log) * (log2 - target_log) > 0:
            continue

        if abs(log2 - log1) < 1e-12:
            return 0.5 * (snr1 + snr2)

        alpha = (target_log - log1) / (log2 - log1)
        return snr1 + alpha * (snr2 - snr1)

    return None


def print_equal_ber_gain(summary_rows, target_bers):
    curves = {}

    for key in ["est_lmmse", "est_ep5", "est_ep20", "oracle_lmmse", "oracle_ep5", "oracle_ep20"]:
        curves[key] = [(row["snr_db"], row[key]) for row in summary_rows]

    print()
    print("=" * 126)
    print("EQUAL-BER SNR GAIN | log10(BER) linear interpolation")
    print("=" * 126)
    print(f"{'Target BER':>12s} | {'LMMSE SNR':>10s} | {'EP5 SNR':>10s} | {'EP5 gain':>10s} | {'Oracle EP5':>11s} | {'R gain':>10s}")
    print("-" * 126)

    gain_rows = []

    for target in target_bers:
        snr_lmmse = snr_at_target(curves["est_lmmse"], target)
        snr_ep5 = snr_at_target(curves["est_ep5"], target)
        snr_oracle_ep5 = snr_at_target(curves["oracle_ep5"], target)

        ep_gain = None if snr_lmmse is None or snr_ep5 is None else snr_lmmse - snr_ep5
        covariance_gain = None if snr_ep5 is None or snr_oracle_ep5 is None else snr_ep5 - snr_oracle_ep5

        if snr_lmmse is None and snr_ep5 is None and snr_oracle_ep5 is None:
            continue

        def fmt(x):
            return "N/A" if x is None else f"{x:.3f}"

        print(f"{target:12.3e} | {fmt(snr_lmmse):>10s} | {fmt(snr_ep5):>10s} | {fmt(ep_gain):>10s} | {fmt(snr_oracle_ep5):>11s} | {fmt(covariance_gain):>10s}")

        gain_rows.append({
            "target_ber": target,
            "est_lmmse_snr_db": snr_lmmse,
            "est_ep5_snr_db": snr_ep5,
            "ep5_vs_lmmse_snr_gain_db": ep_gain,
            "oracle_ep5_snr_db": snr_oracle_ep5,
            "oracle_vs_est_ep5_snr_gain_db": covariance_gain,
        })

    return gain_rows


def run_one_snr(base_system_cfg, train_cfg, snr_db, channels, re_per_channel, progress_every):
    system_cfg = copy.deepcopy(base_system_cfg)
    system_cfg["link"]["rx_snr_db"] = float(snr_db)

    receiver_cfg = train_cfg["receiver"]
    device = system_cfg["general"]["device"]
    base_seed = int(system_cfg["general"]["seed"])

    sionna_config.device = device
    sionna_config.precision = system_cfg["general"]["precision"]
    sionna_config.seed = base_seed
    torch.manual_seed(base_seed)

    dataset = UplinkUMADataset(system_cfg)
    lmmse = LMMSESoftDetector(system_cfg)
    ep = ExpectationPropagationDetector(system_cfg, num_iterations=20, damping=0.5)

    num_rx = dataset.channel.bs_array.num_ant
    num_streams = int(system_cfg["general"]["num_ues"])

    if num_rx != 256:
        raise RuntimeError(f"Expected 256 physical Rx, got {num_rx}.")
    if num_streams != 16:
        raise RuntimeError(f"Expected 16 streams, got {num_streams}.")

    totals = {
        "est_lmmse": 0,
        "est_ep5": 0,
        "est_ep20": 0,
        "oracle_lmmse": 0,
        "oracle_ep5": 0,
        "oracle_ep20": 0,
    }

    per_channel = []
    total_bits = 0
    start_time = time.time()

    with torch.no_grad():
        for channel_idx in range(1, channels + 1):
            batch = dataset.sample(1)
            data_idx = list(dataset.channel.resource_grid.data_symbols)

            if "ruu_true" not in batch:
                raise KeyError("Dataset output does not contain 'ruu_true'.")

            y = batch["y"][:, data_idx]
            h = batch[receiver_cfg["channel_source"]][:, data_idx]
            ruu_hat = batch[receiver_cfg["covariance_source"]]
            ruu_true = batch["ruu_true"]

            bits_flat = batch["bits"].reshape(-1, 16, 4).float()
            n_re = bits_flat.shape[0]
            count = min(int(re_per_channel), n_re)

            idx = torch.randperm(n_re, device=bits_flat.device)[:count]
            bits = bits_flat[idx]

            est_lmmse = flatten_llr(lmmse(y, h, ruu_hat)["llr"])[idx]
            oracle_lmmse = flatten_llr(lmmse(y, h, ruu_true)["llr"])[idx]

            y_est_white, h_est_white = whiten(y, h, ruu_hat)
            y_oracle_white, h_oracle_white = whiten(y, h, ruu_true)

            y_est_sel = y_est_white.reshape(-1, 256)[idx]
            h_est_sel = h_est_white.reshape(-1, 256, 16)[idx]
            y_oracle_sel = y_oracle_white.reshape(-1, 256)[idx]
            h_oracle_sel = h_oracle_white.reshape(-1, 256, 16)[idx]

            z_est, gram_est = sufficient_statistics(y_est_sel, h_est_sel)
            z_oracle, gram_oracle = sufficient_statistics(y_oracle_sel, h_oracle_sel)

            est_ep = get_ep_iterations(ep, z_est, gram_est)
            oracle_ep = get_ep_iterations(ep, z_oracle, gram_oracle)

            num_bits = bits.numel()
            total_bits += num_bits

            outputs = {
                "est_lmmse": est_lmmse,
                "est_ep5": est_ep[5],
                "est_ep20": est_ep[20],
                "oracle_lmmse": oracle_lmmse,
                "oracle_ep5": oracle_ep[5],
                "oracle_ep20": oracle_ep[20],
            }

            row = {"snr_db": float(snr_db), "channel": channel_idx, "num_bits": num_bits}

            for key, llr in outputs.items():
                errors = bit_errors(llr, bits)
                totals[key] += errors
                row[key] = errors / num_bits

            per_channel.append(row)

            if channel_idx == 1 or channel_idx % progress_every == 0 or channel_idx == channels:
                elapsed = time.time() - start_time
                rate = channel_idx / max(elapsed, 1e-9)
                eta = (channels - channel_idx) / max(rate, 1e-9)

                print(
                    f"SNR={snr_db:>5.1f} dB | channel={channel_idx:>4d}/{channels} | "
                    f"LMMSE={totals['est_lmmse'] / total_bits:.6e} | "
                    f"EP5={totals['est_ep5'] / total_bits:.6e} | "
                    f"EP20={totals['est_ep20'] / total_bits:.6e} | "
                    f"OracleEP5={totals['oracle_ep5'] / total_bits:.6e} | "
                    f"elapsed={elapsed / 60:.1f}m | ETA={eta / 60:.1f}m"
                )

    result = {
        "snr_db": float(snr_db),
        "channels": channels,
        "re_per_channel": re_per_channel,
        "total_bits": total_bits,
    }

    for key, errors in totals.items():
        result[key] = errors / total_bits
        result[f"{key}_errors"] = int(errors)

    result["ep5_gain_vs_lmmse_rel"] = (result["est_lmmse"] - result["est_ep5"]) / max(result["est_lmmse"], 1e-30)
    result["ep20_gain_vs_ep5_rel"] = (result["est_ep5"] - result["est_ep20"]) / max(result["est_ep5"], 1e-30)
    result["oracle_ep5_gain_vs_est_ep5_rel"] = (result["est_ep5"] - result["oracle_ep5"]) / max(result["est_ep5"], 1e-30)
    result["oracle_ep20_gain_vs_est_ep20_rel"] = (result["est_ep20"] - result["oracle_ep20"]) / max(result["est_ep20"], 1e-30)

    for key in ["est_lmmse", "est_ep5", "est_ep20", "oracle_lmmse", "oracle_ep5", "oracle_ep20"]:
        mean, low, high = bootstrap_mean_ci([row[key] for row in per_channel])
        result[f"{key}_channel_mean"] = mean
        result[f"{key}_channel_ci95_low"] = low
        result[f"{key}_channel_ci95_high"] = high

    return result, per_channel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--snrs", type=float, nargs="+", default=[0, 2, 4, 5, 6, 8, 10])
    parser.add_argument("--channels", type=int, default=500)
    parser.add_argument("--re-per-channel", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-dir", default="results/snr_headroom")
    parser.add_argument("--target-bers", type=float, nargs="+", default=[0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001])
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    base_system_cfg = load_yaml(train_cfg["system_config"])
    device = base_system_cfg["general"]["device"]

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Configured device is {device}, but CUDA is unavailable.")

    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_tag = "_".join(f"{x:g}" for x in args.snrs)
    tag = f"snr_{snr_tag}_{args.channels}ch_{args.re_per_channel}re"

    summary_json = output_dir / f"{tag}.json"
    summary_csv = output_dir / f"{tag}_summary.csv"
    per_channel_csv = output_dir / f"{tag}_per_channel.csv"
    gain_csv = output_dir / f"{tag}_equal_ber_gain.csv"

    print("=" * 150)
    print("07_Uplink | SNR Headroom Sweep")
    print("=" * 150)
    print(f"SNR points           : {args.snrs}")
    print(f"Channels/SNR         : {args.channels}")
    print(f"Random RE/channel    : {args.re_per_channel}")
    print(f"Bits/SNR             : {args.channels * args.re_per_channel * 16 * 4}")
    print(f"Channel source       : {train_cfg['receiver']['channel_source']}")
    print(f"Estimated covariance : {train_cfg['receiver']['covariance_source']}")
    print("Oracle covariance    : ruu_true")
    print("Detectors            : Est-R LMMSE / EP5 / EP20 + Oracle-R LMMSE / EP5 / EP20")
    print("YAML files           : READ ONLY; no configuration file is modified")
    print("=" * 150)

    all_results = []
    all_per_channel = []
    total_start = time.time()

    for i, snr_db in enumerate(args.snrs, start=1):
        print()
        print("=" * 150)
        print(f"SNR POINT {i}/{len(args.snrs)} | {snr_db:g} dB")
        print("=" * 150)

        result, rows = run_one_snr(
            base_system_cfg=base_system_cfg,
            train_cfg=train_cfg,
            snr_db=snr_db,
            channels=args.channels,
            re_per_channel=args.re_per_channel,
            progress_every=args.progress_every,
        )

        all_results.append(result)
        all_per_channel.extend(rows)

        print("-" * 110)
        print(f"SNR {snr_db:g} dB final:")
        print(f"  Est-R LMMSE : {result['est_lmmse']:.8e}")
        print(f"  Est-R EP5   : {result['est_ep5']:.8e}")
        print(f"  Est-R EP20  : {result['est_ep20']:.8e}")
        print(f"  Oracle LMMSE: {result['oracle_lmmse']:.8e}")
        print(f"  Oracle EP5  : {result['oracle_ep5']:.8e}")
        print(f"  Oracle EP20 : {result['oracle_ep20']:.8e}")
        print(f"  EP5 vs LMMSE relative BER reduction : {100 * result['ep5_gain_vs_lmmse_rel']:+.3f}%")
        print(f"  EP20 vs EP5 relative BER reduction  : {100 * result['ep20_gain_vs_ep5_rel']:+.3f}%")
        print(f"  Oracle-R EP5 vs Est-R EP5           : {100 * result['oracle_ep5_gain_vs_est_ep5_rel']:+.3f}%")

    gain_rows = print_equal_ber_gain(all_results, args.target_bers)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "snrs_db": args.snrs,
            "channels_per_snr": args.channels,
            "re_per_channel": args.re_per_channel,
            "summary": all_results,
            "equal_ber_snr_gain": gain_rows,
        }, f, indent=2)

    summary_fields = list(all_results[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(all_results)

    per_channel_fields = list(all_per_channel[0].keys())
    with open(per_channel_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_channel_fields)
        writer.writeheader()
        writer.writerows(all_per_channel)

    if gain_rows:
        gain_fields = list(gain_rows[0].keys())
        with open(gain_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=gain_fields)
            writer.writeheader()
            writer.writerows(gain_rows)

    elapsed = time.time() - total_start

    print()
    print("=" * 150)
    print("FINAL SNR SWEEP SUMMARY")
    print("=" * 150)
    print(f"{'SNR':>6s} | {'Est LMMSE':>12s} | {'Est EP5':>12s} | {'Est EP20':>12s} | {'Oracle LMMSE':>12s} | {'Oracle EP5':>12s} | {'Oracle EP20':>12s}")
    print("-" * 150)

    for row in all_results:
        print(
            f"{row['snr_db']:6.1f} | "
            f"{row['est_lmmse']:12.5e} | "
            f"{row['est_ep5']:12.5e} | "
            f"{row['est_ep20']:12.5e} | "
            f"{row['oracle_lmmse']:12.5e} | "
            f"{row['oracle_ep5']:12.5e} | "
            f"{row['oracle_ep20']:12.5e}"
        )

    print()
    print(f"Total runtime : {elapsed / 3600:.3f} h")
    print(f"JSON          : {summary_json}")
    print(f"Summary CSV   : {summary_csv}")
    print(f"Per-channel   : {per_channel_csv}")

    if gain_rows:
        print(f"SNR gain CSV  : {gain_csv}")

    print("=" * 150)


if __name__ == "__main__":
    main()