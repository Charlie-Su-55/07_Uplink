#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config
from sionna.phy.mapping import Mapper
from sionna.phy.nr import TBEncoder, TBDecoder
from sionna.phy.nr.utils import decode_mcs_index, calculate_tb_size

from data.dataset import UplinkUMADataset
from detectors.classical.lmmse import LMMSESoftDetector
from detectors.classical.ep import ExpectationPropagationDetector


MOD_NAMES = {2: "QPSK", 4: "16-QAM", 6: "64-QAM"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_int(x):
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def as_float(x):
    return float(x.item()) if isinstance(x, torch.Tensor) else float(x)


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class MCSRuntime:
    table: int
    mcs: int
    qm: int
    target_rate: float
    spectral_efficiency: float
    num_re: int
    num_coded_bits: int
    tb_size: int
    encoder: TBEncoder
    decoder: TBDecoder
    mapper: Mapper


def build_mcs_runtime(cfg, table, mcs, num_re, num_streams, bp_iters):
    device = cfg["general"]["device"]
    precision = cfg["general"]["precision"]

    qm_t, rate_t = decode_mcs_index(
        mcs,
        table_index=table,
        is_pusch=True,
        transform_precoding=False,
        device=device,
    )
    qm = as_int(qm_t)
    rate = as_float(rate_t)

    if qm not in MOD_NAMES:
        raise ValueError(f"T{table} MCS{mcs} uses unsupported Qm={qm}; scout supports QPSK/16-QAM/64-QAM.")

    num_coded_bits = int(num_re * qm)
    tb_result = calculate_tb_size(
        modulation_order=qm,
        target_coderate=rate,
        num_coded_bits=num_coded_bits,
        num_layers=1,
        device=device,
    )
    tb_size = as_int(tb_result[0])

    encoder = TBEncoder(
        target_tb_size=tb_size,
        num_coded_bits=num_coded_bits,
        target_coderate=rate,
        num_bits_per_symbol=qm,
        num_layers=1,
        n_rnti=list(range(1, num_streams + 1)),
        n_id=[1] * num_streams,
        channel_type="PUSCH",
        use_scrambler=True,
        precision=precision,
        device=device,
    )
    decoder = TBDecoder(
        encoder,
        num_bp_iter=bp_iters,
        precision=precision,
        device=device,
    )
    mapper = Mapper("qam", qm, precision=precision, device=device)

    if int(encoder.n) != num_coded_bits:
        raise RuntimeError(f"TBEncoder output length mismatch: {encoder.n} != {num_coded_bits}")

    return MCSRuntime(
        table=int(table),
        mcs=int(mcs),
        qm=qm,
        target_rate=rate,
        spectral_efficiency=qm * rate,
        num_re=int(num_re),
        num_coded_bits=num_coded_bits,
        tb_size=tb_size,
        encoder=encoder,
        decoder=decoder,
        mapper=mapper,
    )


@torch.no_grad()
def codec_identity_test(runtime, num_streams, num_data_symbols, num_subcarriers, device):
    info = torch.randint(0, 2, (1, num_streams, runtime.tb_size), dtype=torch.float32, device=device)
    coded = runtime.encoder(info)
    expected = (1, num_streams, runtime.num_coded_bits)
    if tuple(coded.shape) != expected:
        raise RuntimeError(f"Unexpected encoded shape {tuple(coded.shape)}, expected {expected}")

    bit_grid = coded.reshape(
        1, num_streams, num_data_symbols, num_subcarriers, runtime.qm
    ).permute(0, 2, 3, 1, 4).contiguous()
    perfect_llr = 40.0 * (2.0 * bit_grid - 1.0)
    llr_cw = perfect_llr.permute(0, 3, 1, 2, 4).contiguous().reshape(
        1, num_streams, runtime.num_coded_bits
    )
    decoded, crc_ok = runtime.decoder(llr_cw)

    if not (decoded == info).all().item() or not bool(crc_ok.all().item()):
        raise RuntimeError("Codec identity test failed.")

    symbols = runtime.mapper(coded)
    if tuple(symbols.shape) != (1, num_streams, runtime.num_re):
        raise RuntimeError(f"Mapper shape mismatch: got {tuple(symbols.shape)}")

    print(
        f"Codec identity PASS | T{runtime.table} MCS{runtime.mcs} | "
        f"{MOD_NAMES[runtime.qm]} | R={runtime.target_rate:.4f} | "
        f"TB={runtime.tb_size} | G={runtime.num_coded_bits}"
    )


def coded_bits_to_symbols(coded, runtime, num_data_symbols, num_subcarriers, num_streams):
    symbols = runtime.mapper(coded)
    return symbols.reshape(
        1, num_streams, num_data_symbols, num_subcarriers
    ).permute(0, 2, 3, 1).contiguous()


def llr_grid_to_codeword(llr, runtime, num_streams):
    return llr.permute(0, 3, 1, 2, 4).contiguous().reshape(
        llr.shape[0], num_streams, runtime.num_coded_bits
    )


@torch.no_grad()
def run_ep_chunks(ep, z, gram, qm, num_streams, chunk_size):
    z = z.reshape(-1, num_streams)
    gram = gram.reshape(-1, num_streams, num_streams)
    outputs = []
    for start in range(0, z.shape[0], chunk_size):
        stop = min(start + chunk_size, z.shape[0])
        outputs.append(ep(z[start:stop], gram[start:stop], return_iterations=(5,))["llr"])
    return torch.cat(outputs, dim=0).reshape(-1, num_streams, qm)


@torch.no_grad()
def build_coded_sample(dataset, runtime, num_streams, alignment_check=False):
    batch = dataset.sample(1)
    data_idx = list(dataset.channel.resource_grid.data_symbols)
    num_data_symbols = len(data_idx)
    num_subcarriers = int(dataset.channel.resource_grid.num_subcarriers)

    y_old = batch["y"][:, data_idx]
    y_clean = batch["y_clean"][:, data_idx]
    h_true = batch["h_true"][:, data_idx]
    h_lmmse = batch["h_hat_lmmse"][:, data_idx]
    x_old = batch["x_data"]

    expected_x = (1, num_data_symbols, num_subcarriers, num_streams)
    if tuple(x_old.shape) != expected_x:
        raise RuntimeError(f"Unexpected original x_data shape: {tuple(x_old.shape)}, expected {expected_x}")

    desired_old = torch.einsum("bsfmk,bsfk->bsfm", h_true, x_old)

    if alignment_check:
        numerator = (y_clean - desired_old).abs().square().sum().sqrt()
        denominator = y_clean.abs().square().sum().sqrt().clamp_min(1e-12)
        relative_error = float((numerator / denominator).item())
        print(f"Physical reconstruction check | ||y_clean-Hx||/||y_clean|| = {relative_error:.3e}")
        if relative_error > 1e-4:
            raise RuntimeError("h_true/x_data do not reproduce y_clean.")

    residual = y_old - desired_old
    info_bits = torch.randint(
        0, 2, (1, num_streams, runtime.tb_size), dtype=torch.float32, device=y_old.device
    )
    coded_bits = runtime.encoder(info_bits)
    x_new = coded_bits_to_symbols(
        coded_bits, runtime, num_data_symbols, num_subcarriers, num_streams
    )
    desired_new = torch.einsum("bsfmk,bsfk->bsfm", h_true, x_new)
    y_new = desired_new + residual

    return {
        "y": y_new,
        "h_lmmse": h_lmmse,
        "ruu_hat": batch["ruu_hat"],
        "info_bits": info_bits,
        "num_data_symbols": num_data_symbols,
        "num_subcarriers": num_subcarriers,
    }


@torch.no_grad()
def evaluate_one_channel(sample, frontend, ep, runtime, num_streams, ep_chunk):
    physical = frontend(sample["y"], sample["h_lmmse"], sample["ruu_hat"])
    llr_lmmse = physical["llr"]

    expected = (
        1,
        sample["num_data_symbols"],
        sample["num_subcarriers"],
        num_streams,
        runtime.qm,
    )
    if tuple(llr_lmmse.shape) != expected:
        raise RuntimeError(f"LMMSE LLR shape mismatch: got {tuple(llr_lmmse.shape)}, expected {expected}")

    ep_flat = run_ep_chunks(
        ep,
        physical["z"],
        physical["gram"],
        runtime.qm,
        num_streams,
        ep_chunk,
    )
    llr_ep5 = ep_flat.reshape(*expected)

    lmmse_cw = llr_grid_to_codeword(llr_lmmse, runtime, num_streams)
    ep5_cw = llr_grid_to_codeword(llr_ep5, runtime, num_streams)

    lmmse_bits, lmmse_crc = runtime.decoder(lmmse_cw)
    ep5_bits, ep5_crc = runtime.decoder(ep5_cw)
    truth = sample["info_bits"]

    lmmse_block_error = (lmmse_bits != truth).any(dim=-1)
    ep5_block_error = (ep5_bits != truth).any(dim=-1)

    return {
        "lmmse_block_errors": int(lmmse_block_error.sum().item()),
        "ep5_block_errors": int(ep5_block_error.sum().item()),
        "lmmse_bit_errors": int((lmmse_bits != truth).sum().item()),
        "ep5_bit_errors": int((ep5_bits != truth).sum().item()),
        "lmmse_crc_fail": int((~lmmse_crc).sum().item()),
        "ep5_crc_fail": int((~ep5_crc).sum().item()),
    }


@torch.no_grad()
def evaluate_snr_point(base_cfg, runtime, snr_db, channels, seed, ep_chunk, verbose=True):
    cfg = copy.deepcopy(base_cfg)
    cfg["modulation"]["bits_per_symbol"] = runtime.qm
    cfg["link"]["rx_snr_db"] = float(snr_db)
    cfg["general"]["seed"] = int(seed)

    device = cfg["general"]["device"]
    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(seed)
    torch.manual_seed(int(seed))

    dataset = UplinkUMADataset(cfg)
    frontend = LMMSESoftDetector(cfg)
    ep = ExpectationPropagationDetector(cfg, num_iterations=5, damping=0.5)
    num_streams = int(cfg["general"]["num_ues"]) * int(cfg["general"]["streams_per_ue"])

    totals = {
        "lmmse_block_errors": 0,
        "ep5_block_errors": 0,
        "lmmse_bit_errors": 0,
        "ep5_bit_errors": 0,
        "lmmse_crc_fail": 0,
        "ep5_crc_fail": 0,
    }

    alignment_checked = False
    for channel_idx in range(1, channels + 1):
        sample = build_coded_sample(
            dataset,
            runtime,
            num_streams,
            alignment_check=not alignment_checked,
        )
        alignment_checked = True

        result = evaluate_one_channel(
            sample,
            frontend,
            ep,
            runtime,
            num_streams,
            ep_chunk,
        )
        for key in totals:
            totals[key] += result[key]

        if verbose and (channel_idx % max(channels // 4, 1) == 0 or channel_idx == channels):
            n_blocks = channel_idx * num_streams
            print(
                f"    {channel_idx:4d}/{channels} | "
                f"LMMSE BLER={totals['lmmse_block_errors']/n_blocks:.4f} | "
                f"EP5 BLER={totals['ep5_block_errors']/n_blocks:.4f}"
            )

    total_blocks = channels * num_streams
    total_info_bits = total_blocks * runtime.tb_size

    return {
        "snr_db": float(snr_db),
        "channels": int(channels),
        "total_blocks": int(total_blocks),
        "tb_size": int(runtime.tb_size),
        "lmmse_bler": totals["lmmse_block_errors"] / total_blocks,
        "ep5_bler": totals["ep5_block_errors"] / total_blocks,
        "lmmse_ber": totals["lmmse_bit_errors"] / total_info_bits,
        "ep5_ber": totals["ep5_bit_errors"] / total_info_bits,
        "lmmse_crc_fail_rate": totals["lmmse_crc_fail"] / total_blocks,
        "ep5_crc_fail_rate": totals["ep5_crc_fail"] / total_blocks,
    }


def interpolate_snr_at_bler(points, key, target):
    points = sorted(points, key=lambda x: x["snr_db"])
    for p0, p1 in zip(points[:-1], points[1:]):
        b0 = float(p0[key])
        b1 = float(p1[key])

        if b0 == target:
            return float(p0["snr_db"])
        if b1 == target:
            return float(p1["snr_db"])
        if (b0 - target) * (b1 - target) > 0:
            continue

        n0 = max(int(p0["total_blocks"]), 1)
        n1 = max(int(p1["total_blocks"]), 1)
        y0 = math.log10(max(b0, 0.5 / n0))
        y1 = math.log10(max(b1, 0.5 / n1))
        yt = math.log10(target)
        x0 = float(p0["snr_db"])
        x1 = float(p1["snr_db"])

        if abs(y1 - y0) < 1e-12:
            return 0.5 * (x0 + x1)

        alpha = (yt - y0) / (y1 - y0)
        return x0 + alpha * (x1 - x0)

    return None


def choose_formal_grid(coarse_points, target, step, margin):
    estimates = []
    for key in ("lmmse_bler", "ep5_bler"):
        x = interpolate_snr_at_bler(coarse_points, key, target)
        if x is not None:
            estimates.append(x)

    if estimates:
        low = min(estimates) - margin
        high = max(estimates) + margin
    else:
        best = min(
            coarse_points,
            key=lambda p: min(
                abs(math.log10(max(p["lmmse_bler"], 0.5 / p["total_blocks"])) - math.log10(target)),
                abs(math.log10(max(p["ep5_bler"], 0.5 / p["total_blocks"])) - math.log10(target)),
            ),
        )
        center = float(best["snr_db"])
        low = center - margin
        high = center + margin

    low = math.floor(low / step) * step
    high = math.ceil(high / step) * step
    count = int(round((high - low) / step)) + 1
    return [round(low + i * step, 6) for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--tables", default="1,2")
    parser.add_argument("--mcs", default="4,5,11,19")
    parser.add_argument("--coarse-snrs", default="0,2,4,6,8,10,12,14,16,18,20")
    parser.add_argument("--coarse-channels", type=int, default=20)
    parser.add_argument("--formal-channels", type=int, default=100)
    parser.add_argument("--formal-step-db", type=float, default=0.5)
    parser.add_argument("--formal-margin-db", type=float, default=1.0)
    parser.add_argument("--target-bler", type=float, default=0.1)
    parser.add_argument("--bp-iters", type=int, default=20)
    parser.add_argument("--ep-chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output-dir", default="results/mcs_scout")
    parser.add_argument("--skip-formal", action="store_true")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    base_cfg = load_yaml(train_cfg["system_config"])
    device = base_cfg["general"]["device"]

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        base_cfg["general"]["device"] = device

    sionna_config.device = device
    sionna_config.precision = base_cfg["general"]["precision"]
    torch.set_float32_matmul_precision("high")

    tables = parse_int_list(args.tables)
    mcs_indices = parse_int_list(args.mcs)
    coarse_snrs = parse_float_list(args.coarse_snrs)
    num_streams = int(base_cfg["general"]["num_ues"]) * int(base_cfg["general"]["streams_per_ue"])

    probe_dataset = UplinkUMADataset(copy.deepcopy(base_cfg))
    num_data_symbols = len(probe_dataset.channel.resource_grid.data_symbols)
    num_subcarriers = int(probe_dataset.channel.resource_grid.num_subcarriers)
    num_re = num_data_symbols * num_subcarriers

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    curve_rows = []
    summary_rows = []

    print("=" * 132)
    print("MCS OPERATING-POINT SCOUT | LMMSE-H + estimated Ruu | LMMSE vs EP5")
    print("=" * 132)
    print(f"System              : 256 Rx / {num_streams} streams")
    print(f"Data RE / UE / slot : {num_re}")
    print(f"Target BLER         : {args.target_bler}")
    print(f"MCS tables          : {tables}")
    print(f"MCS indices         : {mcs_indices}")
    print("=" * 132)

    for table in tables:
        for mcs in mcs_indices:
            cfg_mcs = copy.deepcopy(base_cfg)
            qm_t, _ = decode_mcs_index(
                mcs,
                table_index=table,
                is_pusch=True,
                transform_precoding=False,
                device=device,
            )
            qm = as_int(qm_t)
            if qm not in MOD_NAMES:
                print(f"SKIP T{table} MCS{mcs}: Qm={qm}")
                continue

            cfg_mcs["modulation"]["bits_per_symbol"] = qm
            runtime = build_mcs_runtime(
                cfg_mcs, table, mcs, num_re, num_streams, args.bp_iters
            )

            print()
            print("=" * 132)
            print(
                f"T{table} MCS{mcs} | {MOD_NAMES[qm]} | Qm={qm} | "
                f"R={runtime.target_rate:.6f} | SE={runtime.spectral_efficiency:.4f} | "
                f"TB={runtime.tb_size} | G={runtime.num_coded_bits}"
            )
            print("=" * 132)

            torch.manual_seed(args.seed + table * 1000 + mcs)
            codec_identity_test(
                runtime, num_streams, num_data_symbols, num_subcarriers, device
            )

            common_seed = args.seed + table * 100000 + mcs * 1000
            coarse_points = []

            print("COARSE SWEEP")
            for snr in coarse_snrs:
                print(f"  SNR={snr:+.1f} dB")
                point = evaluate_snr_point(
                    base_cfg,
                    runtime,
                    snr,
                    args.coarse_channels,
                    common_seed,
                    args.ep_chunk,
                    verbose=False,
                )
                coarse_points.append(point)
                print(
                    f"    LMMSE BLER={point['lmmse_bler']:.5f} | "
                    f"EP5 BLER={point['ep5_bler']:.5f}"
                )

            if args.skip_formal:
                formal_points = coarse_points
            else:
                formal_snrs = choose_formal_grid(
                    coarse_points,
                    args.target_bler,
                    args.formal_step_db,
                    args.formal_margin_db,
                )
                print("FORMAL GRID: " + ", ".join(f"{x:+.1f}" for x in formal_snrs) + " dB")
                formal_points = []
                for snr in formal_snrs:
                    print(f"  FORMAL SNR={snr:+.1f} dB")
                    point = evaluate_snr_point(
                        base_cfg,
                        runtime,
                        snr,
                        args.formal_channels,
                        common_seed,
                        args.ep_chunk,
                        verbose=True,
                    )
                    formal_points.append(point)

            for point in formal_points:
                curve_rows.append({
                    "table": table,
                    "mcs": mcs,
                    "modulation": MOD_NAMES[qm],
                    "qm": qm,
                    "target_rate": runtime.target_rate,
                    "spectral_efficiency": runtime.spectral_efficiency,
                    "tb_size": runtime.tb_size,
                    "num_coded_bits": runtime.num_coded_bits,
                    **point,
                })

            lmmse_snr = interpolate_snr_at_bler(formal_points, "lmmse_bler", args.target_bler)
            ep5_snr = interpolate_snr_at_bler(formal_points, "ep5_bler", args.target_bler)

            result = {
                "table": table,
                "mcs": mcs,
                "modulation": MOD_NAMES[qm],
                "qm": qm,
                "target_rate": runtime.target_rate,
                "spectral_efficiency": runtime.spectral_efficiency,
                "tb_size": runtime.tb_size,
                "num_coded_bits": runtime.num_coded_bits,
                "target_bler": args.target_bler,
                "lmmse_snr_at_target_db": lmmse_snr,
                "ep5_snr_at_target_db": ep5_snr,
                "ep5_gain_vs_lmmse_db": None if lmmse_snr is None or ep5_snr is None else lmmse_snr - ep5_snr,
                "coarse_points": coarse_points,
                "formal_points": formal_points,
            }
            results.append(result)
            summary_rows.append({
                "table": table,
                "mcs": mcs,
                "modulation": MOD_NAMES[qm],
                "qm": qm,
                "target_rate": runtime.target_rate,
                "spectral_efficiency": runtime.spectral_efficiency,
                "tb_size": runtime.tb_size,
                "num_coded_bits": runtime.num_coded_bits,
                "target_bler": args.target_bler,
                "lmmse_snr_at_target_db": lmmse_snr,
                "ep5_snr_at_target_db": ep5_snr,
                "ep5_gain_vs_lmmse_db": None if lmmse_snr is None or ep5_snr is None else lmmse_snr - ep5_snr,
            })

            with open(output_dir / "mcs_scout_results.json", "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2)
            save_csv(output_dir / "mcs_scout_curves.csv", curve_rows)
            save_csv(output_dir / "mcs_bler_summary.csv", summary_rows)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("=" * 132)
    print("FINAL MCS SCOUT TABLE")
    print("=" * 132)
    print(f"{'Table':>5s} {'MCS':>5s} {'Mod':>8s} {'Qm':>3s} {'R':>8s} {'SE':>8s} {'LMMSE@10%':>12s} {'EP5@10%':>12s} {'EP gain':>10s}")
    print("-" * 132)
    for row in summary_rows:
        lmmse = "N/A" if row["lmmse_snr_at_target_db"] is None else f"{row['lmmse_snr_at_target_db']:.3f}"
        ep5 = "N/A" if row["ep5_snr_at_target_db"] is None else f"{row['ep5_snr_at_target_db']:.3f}"
        gain = "N/A" if row["ep5_gain_vs_lmmse_db"] is None else f"{row['ep5_gain_vs_lmmse_db']:+.3f}"
        print(
            f"{row['table']:5d} {row['mcs']:5d} {row['modulation']:>8s} {row['qm']:3d} "
            f"{row['target_rate']:8.4f} {row['spectral_efficiency']:8.4f} "
            f"{lmmse:>12s} {ep5:>12s} {gain:>10s}"
        )
    print("-" * 132)
    print(f"Summary: {output_dir / 'mcs_bler_summary.csv'}")


if __name__ == "__main__":
    main()
