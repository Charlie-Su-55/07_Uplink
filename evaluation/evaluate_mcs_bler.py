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

import numpy as np
import torch
import yaml

from sionna.phy import config as sionna_config
from sionna.phy.mapping import Mapper
from sionna.phy.nr import TBEncoder, TBDecoder
from sionna.phy.nr.utils import decode_mcs_index, calculate_tb_size

from data.dataset import UplinkUMADataset
from detectors.classical.lmmse import LMMSESoftDetector
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.gt_ep_detector import GraphTransformerEPDetector
from models.baselines.detr_ep_detector import DETREPDetector

SPECIALIST_CHECKPOINTS = {
    (1, 11): {
        "gt": "ckp/mcs_specialists/t1_mcs11_16qam_gt_ep_lmmseH_estR_snr7to9db/best.pth",
        "detr": "ckp/mcs_specialists/t1_mcs11_16qam_detr_ep_lmmseH_estR_snr7to9db/best.pth",
    },
    (2, 5): {
        "gt": "ckp/mcs_specialists/t2_mcs5_16qam_gt_ep_lmmseH_estR_snr6p5to8p5db/best.pth",
        "detr": "ckp/mcs_specialists/t2_mcs5_16qam_detr_ep_lmmseH_estR_snr6p5to8p5db/best.pth",
    },
}

MOD_NAMES = {2: "QPSK", 4: "16-QAM", 6: "64-QAM"}
DETECTOR_LABELS = {"lmmse": "LMMSE", "ep5": "EP5", "gt": "GT-EP", "detr": "DETR-EP"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_int(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def as_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def make_neural_model(cfg, arch):
    common = dict(
        cfg=cfg,
        num_users=16,
        num_iterations=5,
        damping=0.5,
        d_model=128,
        num_heads=8,
        ffn_dim=256,
        dropout=0.05,
        max_logit_correction=4.0,
    )
    if arch == "gt":
        return GraphTransformerEPDetector(
            num_layers=4,
            edge_dim=32,
            edge_mode="full",
            message_mode="cross_user",
            **common,
        )
    if arch == "detr":
        return DETREPDetector(num_layers=3, **common)
    raise ValueError(f"Unknown architecture: {arch}")


def extract_state(checkpoint):
    for key in ("model_state", "state", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        raise KeyError("Checkpoint does not contain model_state/state/state_dict.")

    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        cleaned[new_key] = value
    return cleaned


def load_neural_model(base_cfg, qm, arch, path, device):
    cfg = copy.deepcopy(base_cfg)
    cfg["modulation"]["bits_per_symbol"] = int(qm)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {DETECTOR_LABELS[arch]} checkpoint: {path}")

    model = make_neural_model(cfg, arch)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(extract_state(checkpoint), strict=True)
    model = model.to(device)
    model.eval()

    print(f"Loaded {DETECTOR_LABELS[arch]:<7} | step={checkpoint.get('step', '?')} | {path}")
    return model


def load_specialist_models(base_cfg, table, mcs, qm, device):
    key = (int(table), int(mcs))
    if key not in SPECIALIST_CHECKPOINTS:
        raise KeyError(f"No specialist checkpoint mapping for T{table} MCS{mcs}.")

    paths = SPECIALIST_CHECKPOINTS[key]

    return {
        "gt": load_neural_model(base_cfg, qm, "gt", paths["gt"], device),
        "detr": load_neural_model(base_cfg, qm, "detr", paths["detr"], device),
    }
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
        raise ValueError(f"Unsupported modulation order Qm={qm}.")

    num_coded_bits = int(num_re * qm)

    tb_result = calculate_tb_size(
        modulation_order=qm,
        target_coderate=rate,
        num_coded_bits=num_coded_bits,
        num_layers=1,
        device=device,
    )

    tb_size = as_int(tb_result[0])

    n_rnti = list(range(1, num_streams + 1))
    n_id = [1] * num_streams

    encoder = TBEncoder(
        target_tb_size=tb_size,
        num_coded_bits=num_coded_bits,
        target_coderate=rate,
        num_bits_per_symbol=qm,
        num_layers=1,
        n_rnti=n_rnti,
        n_id=n_id,
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

    mapper = Mapper(
        "qam",
        qm,
        precision=precision,
        device=device,
    )

    if int(encoder.n) != num_coded_bits:
        raise RuntimeError(
            f"TBEncoder output length mismatch: {encoder.n} != {num_coded_bits}"
        )

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
    info = torch.randint(
        0,
        2,
        (1, num_streams, runtime.tb_size),
        dtype=torch.float32,
        device=device,
    )

    coded = runtime.encoder(info)

    expected_shape = (1, num_streams, runtime.num_coded_bits)
    if tuple(coded.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected encoded shape {tuple(coded.shape)}, expected {expected_shape}"
        )

    bit_grid = coded.reshape(
        1,
        num_streams,
        num_data_symbols,
        num_subcarriers,
        runtime.qm,
    ).permute(0, 2, 3, 1, 4).contiguous()

    perfect_llr_grid = 40.0 * (2.0 * bit_grid - 1.0)

    llr_cw = perfect_llr_grid.permute(
        0, 3, 1, 2, 4
    ).contiguous().reshape(
        1,
        num_streams,
        runtime.num_coded_bits,
    )

    decoded, crc_ok = runtime.decoder(llr_cw)

    exact = (decoded == info).all().item()
    crc_exact = bool(crc_ok.all().item())

    symbols = runtime.mapper(coded)

    if tuple(symbols.shape) != (1, num_streams, runtime.num_re):
        raise RuntimeError(
            f"Mapper shape mismatch: got {tuple(symbols.shape)}, "
            f"expected {(1, num_streams, runtime.num_re)}"
        )

    if not exact or not crc_exact:
        raise RuntimeError(
            f"Codec identity test failed: exact={exact}, CRC={crc_exact}"
        )

    print(
        f"Codec identity PASS | T{runtime.table} MCS{runtime.mcs} | "
        f"{MOD_NAMES[runtime.qm]} | R={runtime.target_rate:.4f} | "
        f"TB={runtime.tb_size} | G={runtime.num_coded_bits}"
    )


def coded_bits_to_grid(coded, runtime, num_data_symbols, num_subcarriers, num_streams):
    bit_grid = coded.reshape(
        1,
        num_streams,
        num_data_symbols,
        num_subcarriers,
        runtime.qm,
    ).permute(0, 2, 3, 1, 4).contiguous()

    return bit_grid


def coded_bits_to_symbols(coded, runtime, num_data_symbols, num_subcarriers, num_streams):
    symbols = runtime.mapper(coded)

    symbols = symbols.reshape(
        1,
        num_streams,
        num_data_symbols,
        num_subcarriers,
    ).permute(0, 2, 3, 1).contiguous()

    return symbols


def llr_grid_to_codeword(llr, runtime, num_streams):
    return llr.permute(
        0, 3, 1, 2, 4
    ).contiguous().reshape(
        llr.shape[0],
        num_streams,
        runtime.num_coded_bits,
    )


@torch.no_grad()
def run_neural_chunks(model, z, gram, qm, chunk_size):
    z = z.reshape(-1, 16)
    gram = gram.reshape(-1, 16, 16)

    outputs = []

    for start in range(0, z.shape[0], chunk_size):
        stop = min(start + chunk_size, z.shape[0])
        out = model(
            z[start:stop],
            gram[start:stop],
            return_iterations=(5,),
        )
        outputs.append(out["llr"])

    return torch.cat(outputs, dim=0).reshape(-1, 16, qm)


@torch.no_grad()
def build_coded_sample(dataset, runtime, num_streams, alignment_check=False):
    batch = dataset.sample(1)

    data_idx = list(dataset.channel.resource_grid.data_symbols)
    num_data_symbols = len(data_idx)
    num_subcarriers = dataset.channel.resource_grid.num_subcarriers

    y_old = batch["y"][:, data_idx]
    y_clean = batch["y_clean"][:, data_idx]

    h_true = batch["h_true"][:, data_idx]
    h_lmmse = batch["h_hat_lmmse"][:, data_idx]

    x_old = batch["x_data"]

    if tuple(x_old.shape) != (
        1,
        num_data_symbols,
        num_subcarriers,
        num_streams,
    ):
        raise RuntimeError(
            f"Unexpected original x_data shape: {tuple(x_old.shape)}"
        )

    desired_old = torch.einsum(
        "bsfmk,bsfk->bsfm",
        h_true,
        x_old,
    )

    if alignment_check:
        numerator = (
            y_clean - desired_old
        ).abs().square().sum().sqrt()

        denominator = (
            y_clean.abs().square().sum().sqrt().clamp_min(1e-12)
        )

        relative_error = float(
            (numerator / denominator).item()
        )

        print(
            f"Physical reconstruction check | "
            f"||y_clean-Hx||/||y_clean|| = {relative_error:.3e}"
        )

        if relative_error > 1e-4:
            raise RuntimeError(
                "h_true/x_data do not reproduce y_clean. "
                "Abort coded-payload injection rather than using a wrong signal model."
            )

    residual = y_old - desired_old

    info_bits = torch.randint(
        0,
        2,
        (1, num_streams, runtime.tb_size),
        dtype=torch.float32,
        device=y_old.device,
    )

    coded_bits = runtime.encoder(info_bits)

    x_new = coded_bits_to_symbols(
        coded_bits,
        runtime,
        num_data_symbols,
        num_subcarriers,
        num_streams,
    )

    desired_new = torch.einsum(
        "bsfmk,bsfk->bsfm",
        h_true,
        x_new,
    )

    y_new = desired_new + residual

    return {
        "y": y_new,
        "h_lmmse": h_lmmse,
        "h_true": h_true,
        "ruu_hat": batch["ruu_hat"],
        "info_bits": info_bits,
        "coded_bits": coded_bits,
        "num_data_symbols": num_data_symbols,
        "num_subcarriers": num_subcarriers,
    }


@torch.no_grad()
def evaluate_one_channel(sample, frontend, classical_ep, models, runtime, num_streams, gt_chunk):
    physical = frontend(
        sample["y"],
        sample["h_lmmse"],
        sample["ruu_hat"],
    )

    expected = (
        1,
        sample["num_data_symbols"],
        sample["num_subcarriers"],
        num_streams,
        runtime.qm,
    )

    llrs = {
        "lmmse": physical["llr"],
        "ep5": classical_ep(
            physical["z"],
            physical["gram"],
            return_iterations=(5,),
        )["llr"],
    }

    for name, model in models.items():
        flat = run_neural_chunks(
            model,
            physical["z"],
            physical["gram"],
            runtime.qm,
            gt_chunk,
        )
        llrs[name] = flat.reshape(expected)

    for name, llr in llrs.items():
        if tuple(llr.shape) != expected:
            raise RuntimeError(
                f"{name} LLR shape mismatch: got {tuple(llr.shape)}, expected {expected}"
            )

    truth = sample["info_bits"]
    block_masks = {}
    result = {}

    for name, llr in llrs.items():
        cw = llr_grid_to_codeword(llr, runtime, num_streams)
        bits_hat, crc_ok = runtime.decoder(cw)

        block_error = (bits_hat != truth).any(dim=-1)
        block_masks[name] = block_error

        result[f"{name}_block_errors"] = int(block_error.sum().item())
        result[f"{name}_bit_errors"] = int((bits_hat != truth).sum().item())
        result[f"{name}_crc_fail"] = int((~crc_ok).sum().item())

    ep_mask = block_masks["ep5"]

    for name in ("gt", "detr"):
        if name in block_masks:
            result[f"{name}_fixed_vs_ep"] = int(
                (ep_mask & (~block_masks[name])).sum().item()
            )
            result[f"{name}_broken_vs_ep"] = int(
                ((~ep_mask) & block_masks[name]).sum().item()
            )

    return result

@torch.no_grad()
def evaluate_snr_point(base_cfg, runtime, models, snr_db, channels, seed, gt_chunk, verbose=True):
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
    classical_ep = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    num_streams = (
        int(cfg["general"]["num_ues"])
        * int(cfg["general"]["streams_per_ue"])
    )

    detector_names = ["lmmse", "ep5"] + [
        name for name in ("gt", "detr") if name in models
    ]

    totals = {}
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
            classical_ep,
            models,
            runtime,
            num_streams,
            gt_chunk,
        )

        for key, value in result.items():
            totals[key] = totals.get(key, 0) + int(value)

        if verbose and (
            channel_idx % max(channels // 4, 1) == 0
            or channel_idx == channels
        ):
            n_blocks = channel_idx * num_streams
            status = " | ".join(
                f"{DETECTOR_LABELS[name]} BLER="
                f"{totals[f'{name}_block_errors']/n_blocks:.4f}"
                for name in detector_names
            )
            print(f"    {channel_idx:4d}/{channels} | {status}")

    total_blocks = channels * num_streams
    total_info_bits = total_blocks * runtime.tb_size

    point = {
        "snr_db": float(snr_db),
        "channels": int(channels),
        "total_blocks": int(total_blocks),
        "tb_size": int(runtime.tb_size),
    }

    for name in detector_names:
        point[f"{name}_block_errors"] = totals[f"{name}_block_errors"]
        point[f"{name}_bit_errors"] = totals[f"{name}_bit_errors"]
        point[f"{name}_bler"] = totals[f"{name}_block_errors"] / total_blocks
        point[f"{name}_ber"] = totals[f"{name}_bit_errors"] / total_info_bits
        point[f"{name}_crc_fail_rate"] = (
            totals[f"{name}_crc_fail"] / total_blocks
        )

    for name in ("gt", "detr"):
        fixed_key = f"{name}_fixed_vs_ep"
        broken_key = f"{name}_broken_vs_ep"

        if fixed_key in totals:
            point[fixed_key] = totals[fixed_key]
            point[broken_key] = totals[broken_key]
            point[f"{name}_net_fixed_vs_ep"] = (
                totals[fixed_key] - totals[broken_key]
            )

    return point

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

        floor0 = 0.5 / n0
        floor1 = 0.5 / n1

        y0 = math.log10(max(b0, floor0))
        y1 = math.log10(max(b1, floor1))
        yt = math.log10(target)

        x0 = float(p0["snr_db"])
        x1 = float(p1["snr_db"])

        if abs(y1 - y0) < 1e-12:
            return 0.5 * (x0 + x1)

        alpha = (yt - y0) / (y1 - y0)
        return x0 + alpha * (x1 - x0)

    return None


def choose_formal_grid(coarse_points, target, step, margin):
    bler_keys = [
        key for key in coarse_points[0]
        if key.endswith("_bler")
    ]

    estimates = []

    for key in bler_keys:
        x = interpolate_snr_at_bler(
            coarse_points,
            key,
            target,
        )
        if x is not None:
            estimates.append(x)

    if estimates:
        low = min(estimates) - margin
        high = max(estimates) + margin
    else:
        best = min(
            coarse_points,
            key=lambda p: min(
                abs(
                    math.log10(
                        max(
                            p[key],
                            0.5 / p["total_blocks"],
                        )
                    )
                    - math.log10(target)
                )
                for key in bler_keys
            ),
        )

        center = float(best["snr_db"])
        low = center - margin
        high = center + margin

    low = math.floor(low / step) * step
    high = math.ceil(high / step) * step

    count = int(round((high - low) / step)) + 1

    return [
        round(low + i * step, 6)
        for i in range(count)
    ]

def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(text):
    return [
        int(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def parse_float_list(text):
    return [
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--tables", default="1,2")
    parser.add_argument("--mcs", default="4,5,11,19")
    parser.add_argument("--coarse-snrs", default="-12,-8,-4,0,4,8,12,16,20,24")
    parser.add_argument("--coarse-channels", type=int, default=20)
    parser.add_argument("--formal-channels", type=int, default=100)
    parser.add_argument("--formal-step-db", type=float, default=1.0)
    parser.add_argument("--formal-margin-db", type=float, default=2.0)
    parser.add_argument("--target-bler", type=float, default=0.1)
    parser.add_argument("--bp-iters", type=int, default=20)
    parser.add_argument("--gt-chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output-dir", default="results/mcs_bler")
    parser.add_argument("--skip-formal", action="store_true")
    parser.add_argument("--operating-point", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    OPERATING_SNRS = {
        (1, 4): 4.0,
        (1, 5): 4.0,
        (1, 11): 8.0,
        (1, 19): 12.0,
        (2, 4): 8.0,
        (2, 5): 8.0,
        (2, 11): 8.0,
        (2, 19): 12.0,
    }

    train_cfg = load_yaml(args.config)
    base_cfg = load_yaml(train_cfg["system_config"])

    device = base_cfg["general"]["device"]

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"
        base_cfg["general"]["device"] = device

    sionna_config.device = device
    sionna_config.precision = base_cfg["general"]["precision"]
    torch.set_float32_matmul_precision("high")

    tables = parse_int_list(args.tables)
    mcs_indices = parse_int_list(args.mcs)
    coarse_snrs = parse_float_list(args.coarse_snrs)

    num_streams = (
        int(base_cfg["general"]["num_ues"])
        * int(base_cfg["general"]["streams_per_ue"])
    )

    probe_dataset = UplinkUMADataset(copy.deepcopy(base_cfg))
    num_data_symbols = len(
        probe_dataset.channel.resource_grid.data_symbols
    )
    num_subcarriers = int(
        probe_dataset.channel.resource_grid.num_subcarriers
    )
    num_re = num_data_symbols * num_subcarriers

    print("=" * 136)
    print("3GPP NR MCS BLER EVALUATION | LMMSE / EP5 / GT-EP / DETR-EP")
    print("=" * 136)
    print(f"System              : 256 Rx / {num_streams} streams")
    print(f"Data symbols        : {num_data_symbols}")
    print(f"Subcarriers         : {num_subcarriers}")
    print(f"Data RE / UE / slot : {num_re}")
    print(f"Target BLER         : {args.target_bler}")
    print(f"MCS tables          : {tables}")
    print(f"MCS indices         : {mcs_indices}")
    print(f"Coarse channels/SNR : {args.coarse_channels}")
    print(f"Formal channels/SNR : {args.formal_channels}")
    print(f"Baseline only       : {args.baseline_only}")
    print("=" * 136)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    curve_rows = []
    summary_rows = []
    model_cache = {}

    def save_progress():
        payload = {
            "config": {
                "mode": (
                    "operating_point"
                    if args.operating_point
                    else "bler_crossing"
                ),
                "target_bler": args.target_bler,
                "coarse_channels": args.coarse_channels,
                "formal_channels": args.formal_channels,
                "formal_step_db": args.formal_step_db,
                "formal_margin_db": args.formal_margin_db,
                "seed": args.seed,
                "baseline_only": args.baseline_only,
            },
            "results": all_results,
        }

        with open(
            output_dir / "mcs_bler_results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, indent=2)

        save_csv(
            output_dir / "mcs_bler_curves.csv",
            curve_rows,
        )
        save_csv(
            output_dir / "mcs_bler_summary.csv",
            summary_rows,
        )

    for table in tables:
        for mcs in mcs_indices:
            qm_t, _ = decode_mcs_index(
                mcs,
                table_index=table,
                is_pusch=True,
                transform_precoding=False,
                device=device,
            )
            qm = as_int(qm_t)

            if qm not in MOD_NAMES:
                print(
                    f"SKIP | T{table} MCS{mcs} "
                    f"uses unsupported Qm={qm}."
                )
                continue

            key = (table, mcs)

            if (
                not args.baseline_only
                and key not in SPECIALIST_CHECKPOINTS
            ):
                print(
                    f"SKIP | T{table} MCS{mcs}: "
                    f"no GT/DETR specialist mapping yet."
                )
                continue

            cfg_mcs = copy.deepcopy(base_cfg)
            cfg_mcs["modulation"]["bits_per_symbol"] = qm

            runtime = build_mcs_runtime(
                cfg=cfg_mcs,
                table=table,
                mcs=mcs,
                num_re=num_re,
                num_streams=num_streams,
                bp_iters=args.bp_iters,
            )

            if args.baseline_only:
                models = {}
            else:
                if key not in model_cache:
                    model_cache[key] = load_specialist_models(
                        base_cfg,
                        table,
                        mcs,
                        qm,
                        device,
                    )
                models = model_cache[key]

            detector_names = ["lmmse", "ep5"] + [
                name
                for name in ("gt", "detr")
                if name in models
            ]

            print()
            print("=" * 136)
            print(
                f"T{table} MCS{mcs} | "
                f"{MOD_NAMES[qm]} | "
                f"Qm={qm} | "
                f"R={runtime.target_rate:.6f} | "
                f"SE={runtime.spectral_efficiency:.4f} | "
                f"TB={runtime.tb_size} | "
                f"G={runtime.num_coded_bits}"
            )
            print("=" * 136)

            torch.manual_seed(
                args.seed + table * 1000 + mcs
            )

            codec_identity_test(
                runtime,
                num_streams,
                num_data_symbols,
                num_subcarriers,
                device,
            )

            common_seed = (
                args.seed
                + table * 100000
                + mcs * 1000
            )

            if args.operating_point:
                snr = OPERATING_SNRS[key]

                print()
                print(
                    f"OPERATING-POINT EVALUATION | "
                    f"SNR={snr:+.1f} dB"
                )

                point = evaluate_snr_point(
                    base_cfg=base_cfg,
                    runtime=runtime,
                    models=models,
                    snr_db=snr,
                    channels=args.formal_channels,
                    seed=common_seed,
                    gt_chunk=args.gt_chunk,
                    verbose=True,
                )

                metadata = {
                    "mode": "operating_point",
                    "table": table,
                    "mcs": mcs,
                    "modulation": MOD_NAMES[qm],
                    "qm": qm,
                    "target_rate": runtime.target_rate,
                    "spectral_efficiency": runtime.spectral_efficiency,
                    "tb_size": runtime.tb_size,
                    "num_coded_bits": runtime.num_coded_bits,
                }

                result = {
                    **metadata,
                    **point,
                }

                all_results.append(result)
                summary_rows.append(result)
                curve_rows.append(result)

                print()

                for name in detector_names:
                    print(
                        f"{DETECTOR_LABELS[name]:<8} | "
                        f"BLER={point[f'{name}_bler']:.6f} | "
                        f"BER={point[f'{name}_ber']:.6e} | "
                        f"block errors="
                        f"{point[f'{name}_block_errors']} | "
                        f"bit errors="
                        f"{point[f'{name}_bit_errors']}"
                    )

                save_progress()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                continue

            print()
            print("COARSE SWEEP")

            coarse_points = []

            for snr in coarse_snrs:
                print(
                    f"  SNR={snr:+.1f} dB"
                )

                point = evaluate_snr_point(
                    base_cfg=base_cfg,
                    runtime=runtime,
                    models=models,
                    snr_db=snr,
                    channels=args.coarse_channels,
                    seed=common_seed,
                    gt_chunk=args.gt_chunk,
                    verbose=False,
                )

                coarse_points.append(point)

                status = " | ".join(
                    f"{DETECTOR_LABELS[name]}="
                    f"{point[f'{name}_bler']:.5f}"
                    for name in detector_names
                )

                extra = []

                for name in ("gt", "detr"):
                    key_nf = f"{name}_net_fixed_vs_ep"

                    if key_nf in point:
                        extra.append(
                            f"{DETECTOR_LABELS[name]} "
                            f"net-vs-EP="
                            f"{point[key_nf]:+d}"
                        )

                if extra:
                    status += (
                        " | "
                        + " | ".join(extra)
                    )

                print(
                    f"    {status}"
                )

            if args.skip_formal:
                print()
                print(
                    "FORMAL SWEEP SKIPPED | "
                    "using coarse points for crossing diagnostic"
                )

                formal_points = coarse_points

            else:
                formal_snrs = choose_formal_grid(
                    coarse_points,
                    target=args.target_bler,
                    step=args.formal_step_db,
                    margin=args.formal_margin_db,
                )

                print()
                print(
                    "FORMAL GRID: "
                    + ", ".join(
                        f"{x:+.1f}"
                        for x in formal_snrs
                    )
                    + " dB"
                )

                formal_points = []

                for snr in formal_snrs:
                    print()
                    print(
                        f"  FORMAL SNR="
                        f"{snr:+.1f} dB"
                    )

                    point = evaluate_snr_point(
                        base_cfg=base_cfg,
                        runtime=runtime,
                        models=models,
                        snr_db=snr,
                        channels=args.formal_channels,
                        seed=common_seed,
                        gt_chunk=args.gt_chunk,
                        verbose=True,
                    )

                    formal_points.append(point)

            metadata = {
                "table": table,
                "mcs": mcs,
                "modulation": MOD_NAMES[qm],
                "qm": qm,
                "target_rate": runtime.target_rate,
                "spectral_efficiency": runtime.spectral_efficiency,
                "tb_size": runtime.tb_size,
                "num_coded_bits": runtime.num_coded_bits,
            }

            for point in formal_points:
                curve_rows.append({
                    **metadata,
                    **point,
                })

            snr_at_target = {
                name: interpolate_snr_at_bler(
                    formal_points,
                    f"{name}_bler",
                    args.target_bler,
                )
                for name in detector_names
            }

            ep_snr = snr_at_target["ep5"]

            gain_vs_ep = {
                name: (
                    None
                    if (
                        ep_snr is None
                        or snr_at_target[name] is None
                    )
                    else (
                        ep_snr
                        - snr_at_target[name]
                    )
                )
                for name in detector_names
                if name != "ep5"
            }

            result = {
                **metadata,
                "target_bler": args.target_bler,
                "snr_at_target_db": snr_at_target,
                "gain_vs_ep5_db": gain_vs_ep,
                "coarse_points": coarse_points,
                "formal_points": formal_points,
            }

            all_results.append(result)

            summary = {
                **metadata,
                "target_bler": args.target_bler,
            }

            for name in detector_names:
                summary[
                    f"{name}_snr_at_target_db"
                ] = snr_at_target[name]

            for name, gain in gain_vs_ep.items():
                summary[
                    f"{name}_gain_vs_ep5_db"
                ] = gain

            summary_rows.append(summary)

            print()
            print("-" * 136)
            print(
                f"T{table} MCS{mcs} | "
                f"target BLER="
                f"{args.target_bler:g}"
            )

            for name in detector_names:
                snr_value = snr_at_target[name]

                snr_text = (
                    "N/A"
                    if snr_value is None
                    else f"{snr_value:.3f} dB"
                )

                if name == "ep5":
                    print(
                        f"  {DETECTOR_LABELS[name]:<8} "
                        f"@ target: {snr_text}"
                    )

                else:
                    gain = gain_vs_ep.get(name)

                    gain_text = (
                        "N/A"
                        if gain is None
                        else (
                            f"{gain:+.3f} dB "
                            f"vs EP5"
                        )
                    )

                    print(
                        f"  {DETECTOR_LABELS[name]:<8} "
                        f"@ target: {snr_text} | "
                        f"{gain_text}"
                    )

            print("-" * 136)

            save_progress()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("=" * 136)
    print("FINAL SUMMARY")
    print("=" * 136)

    if args.operating_point:
        for row in summary_rows:
            print(
                f"T{row['table']} "
                f"MCS{row['mcs']} | "
                f"{row['modulation']} | "
                f"SNR={row['snr_db']:+.1f} dB"
            )

            names = ["lmmse", "ep5"] + [
                name
                for name in ("gt", "detr")
                if f"{name}_bler" in row
            ]

            for name in names:
                print(
                    f"  {DETECTOR_LABELS[name]:<8} "
                    f"BLER="
                    f"{row[f'{name}_bler']:.6f} | "
                    f"BER="
                    f"{row[f'{name}_ber']:.6e}"
                )

    else:
        for row in summary_rows:
            print(
                f"T{row['table']} "
                f"MCS{row['mcs']} | "
                f"{row['modulation']} | "
                f"target BLER="
                f"{row['target_bler']:g}"
            )

            names = ["lmmse", "ep5"] + [
                name
                for name in ("gt", "detr")
                if (
                    f"{name}_snr_at_target_db"
                    in row
                )
            ]

            ep_snr = row.get(
                "ep5_snr_at_target_db"
            )

            for name in names:
                x = row.get(
                    f"{name}_snr_at_target_db"
                )

                snr_text = (
                    "N/A"
                    if x is None
                    else f"{x:.3f} dB"
                )

                if name == "ep5":
                    print(
                        f"  {DETECTOR_LABELS[name]:<8} "
                        f"{snr_text}"
                    )

                else:
                    gain = (
                        None
                        if (
                            ep_snr is None
                            or x is None
                        )
                        else ep_snr - x
                    )

                    gain_text = (
                        "N/A"
                        if gain is None
                        else (
                            f"{gain:+.3f} dB "
                            f"vs EP5"
                        )
                    )

                    print(
                        f"  {DETECTOR_LABELS[name]:<8} "
                        f"{snr_text} | "
                        f"{gain_text}"
                    )

    print("=" * 136)


if __name__ == "__main__":
    main()