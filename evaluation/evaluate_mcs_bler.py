#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
import warnings
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
DETECTOR_LABELS = {"lmmse": "LMMSE", "ep5": "EP5", "gt": "GT-EP", "detr": "DETR-EP", "flow": "Flow-Matching"}


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


def load_neural_model(base_cfg, qm, arch, path, device, table=None, mcs=None):
    cfg = copy.deepcopy(base_cfg)
    cfg["modulation"]["bits_per_symbol"] = int(qm)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {DETECTOR_LABELS[arch]} checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if arch == "flow":
        from models.baselines.flow_matching_detector import flow_from_checkpoint
        model = flow_from_checkpoint(cfg, checkpoint)
    else:
        model = make_neural_model(cfg, arch)
    expected = {"arch": "flow_matching" if arch == "flow" else f"{arch}_ep", "bits_per_symbol": int(qm), "csi": "lmmseH", "covariance": "estimated_Ruu"}
    if table is not None:
        expected.update(mcs_table=int(table), mcs_index=int(mcs))
    missing = []
    for key, value in expected.items():
        if key not in checkpoint:
            missing.append(key)
        elif checkpoint[key] != value:
            raise ValueError(f"Checkpoint {path}: {key}={checkpoint[key]!r}, expected {value!r}")
    if missing and arch == "flow":
        raise ValueError(f"Flow checkpoint {path}: missing required metadata {missing}")
    if missing:
        warnings.warn(f"Legacy checkpoint {path}: unverified metadata {missing}", RuntimeWarning)
    model.load_state_dict(extract_state(checkpoint), strict=True)
    model = model.to(device)
    model.eval()
    model.checkpoint_metadata = {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "step": checkpoint.get("step"), "unverified_fields": missing,
        "model_config": checkpoint.get("model_config"),
        **{key: checkpoint.get(key) for key in expected},
    }

    print(f"Loaded {DETECTOR_LABELS[arch]:<7} | step={checkpoint.get('step', '?')} | {path}")
    return model


def load_specialist_models(base_cfg, table, mcs, qm, device, checkpoint_map=None):
    checkpoint_map = SPECIALIST_CHECKPOINTS if checkpoint_map is None else checkpoint_map
    key = (int(table), int(mcs))
    if key not in checkpoint_map:
        raise KeyError(f"No specialist checkpoint mapping for T{table} MCS{mcs}.")

    paths = checkpoint_map[key]

    return {
        name: load_neural_model(base_cfg, qm, name, paths[name], device, table, mcs)
        for name in ("gt", "detr", "flow") if name in paths
    }


def read_checkpoint_map(path=None):
    mapping = copy.deepcopy(SPECIALIST_CHECKPOINTS)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            overrides = json.load(f)
        for key, paths in overrides.items():
            table, mcs = map(int, key.split(":"))
            if (not {"gt", "detr"}.issubset(paths) or not set(paths).issubset({"gt", "detr", "flow"})
                    or not all(isinstance(p, str) and p for p in paths.values())):
                raise ValueError(f"Checkpoint map {key}: expected nonempty gt/detr paths and optional flow path")
            mapping[(table, mcs)] = paths
    return mapping


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
    streams = z.shape[-1]
    if gram.shape != (*z.shape, streams) or chunk_size <= 0:
        raise ValueError("Invalid z/G dimensions or chunk_size")
    z = z.reshape(-1, streams)
    gram = gram.reshape(-1, streams, streams)

    outputs = []

    for start in range(0, z.shape[0], chunk_size):
        stop = min(start + chunk_size, z.shape[0])
        out = model(
            z[start:stop],
            gram[start:stop],
            return_iterations=(5,),
        )
        if out["llr"].shape != (stop - start, streams, qm) or not torch.isfinite(out["llr"]).all():
            raise RuntimeError("Neural detector returned invalid LLR dimensions or values")
        outputs.append(out["llr"])

    return torch.cat(outputs, dim=0).reshape(-1, streams, qm)


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

    for name in ("gt", "detr", "flow"):
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
        name for name in ("gt", "detr", "flow") if name in models
    ]

    totals = {}
    channel_counts = []
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

        channel_counts.append({"channel": channel_idx, **result})

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
        "seed": int(seed),
        "total_info_bits": int(total_info_bits),
        "channel_counts": channel_counts,
    }

    for name in detector_names:
        point[f"{name}_block_errors"] = totals[f"{name}_block_errors"]
        point[f"{name}_bit_errors"] = totals[f"{name}_bit_errors"]
        point[f"{name}_bler"] = totals[f"{name}_block_errors"] / total_blocks
        point[f"{name}_ber"] = totals[f"{name}_bit_errors"] / total_info_bits
        point[f"{name}_crc_fail_rate"] = (
            totals[f"{name}_crc_fail"] / total_blocks
        )

    for name in ("gt", "detr", "flow"):
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
    """Interpolate log BLER only inside both the observed and floored bracket."""
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("Target BLER must be finite and strictly between 0 and 1.")
    points = sorted(points, key=lambda x: x["snr_db"])
    for i, point in enumerate(points):
        if not math.isfinite(float(point["snr_db"])) or not math.isfinite(float(point[key])):
            raise ValueError("SNR and BLER observations must be finite.")
        if not 0.0 <= float(point[key]) <= 1.0 or int(point["total_blocks"]) <= 0:
            raise ValueError("BLER must be in [0,1] and total_blocks must be positive.")
        if i and point["snr_db"] == points[i - 1]["snr_db"]:
            raise ValueError("Duplicate SNR observations; aggregate counts before interpolation.")
    for point in points:
        if float(point[key]) == target:
            return float(point["snr_db"])
    for p0, p1 in zip(points, points[1:]):
        b0, b1 = float(p0[key]), float(p1[key])
        if (b0 - target) * (b1 - target) > 0:
            continue
        y0 = math.log10(max(b0, 0.5 / int(p0["total_blocks"])))
        y1 = math.log10(max(b1, 0.5 / int(p1["total_blocks"])))
        yt = math.log10(target)
        # A zero-count floor must not turn interpolation into extrapolation.
        if not min(y0, y1) <= yt <= max(y0, y1):
            continue
        x0, x1 = float(p0["snr_db"]), float(p1["snr_db"])
        if abs(y1 - y0) < 1e-12:
            return 0.5 * (x0 + x1)
        alpha = (yt - y0) / (y1 - y0)
        return x0 + alpha * (x1 - x0)
    return None


def extend_bler_grid(points, keys, target, step, max_extensions, evaluate_point, on_progress=None):
    """Add at most max_extensions SNR points, sharing each point across detectors."""
    if not points:
        raise ValueError("Cannot bracket an empty SNR grid.")
    if not math.isfinite(step) or step <= 0 or max_extensions < 0:
        raise ValueError("Bracket step must be positive; extension budget must be nonnegative.")
    for extension in range(max_extensions):
        directions = set()
        for key in keys:
            if interpolate_snr_at_bler(points, key, target) is not None:
                continue
            values = [float(p[key]) for p in points]
            if max(values) < target:
                directions.add(-1)
            elif min(values) > target:
                directions.add(1)
        if not directions:
            break
        # Alternate when different curves need opposite sides.
        direction = sorted(directions)[extension % len(directions)]
        edge = min(p["snr_db"] for p in points) if direction < 0 else max(p["snr_db"] for p in points)
        snr = round(edge + direction * step, 6)
        if not math.isfinite(snr) or any(p["snr_db"] == snr for p in points):
            raise ValueError("Bracket step cannot produce a new finite SNR point.")
        point = evaluate_point(snr)
        if point["snr_db"] != snr:
            raise ValueError("Evaluator returned a different SNR than requested.")
        points.append(point)
        points.sort(key=lambda p: p["snr_db"])
        if on_progress is not None:
            on_progress()
    messages = []
    for key in keys:
        if interpolate_snr_at_bler(points, key, target) is None:
            values = [float(p[key]) for p in points]
            side = "lower SNR" if max(values) < target else "higher SNR" if min(values) > target else "more blocks (zero-count resolution)"
            messages.append(f"{key}: target {target:g} unresolved; need {side}; extension limit={max_extensions}.")
        ordered = sorted(points, key=lambda p: p["snr_db"])
        if any(b[key] > a[key] for a, b in zip(ordered, ordered[1:])):
            messages.append(f"{key}: nonmonotone observations; inspect channel uncertainty before reporting a crossing.")
    for message in messages:
        warnings.warn(message, RuntimeWarning)
    return messages


def choose_formal_grid(coarse_points, target, step, margin):
    if not coarse_points or not math.isfinite(step) or step <= 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Formal grid requires observations, positive step and nonnegative margin.")
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

    fieldnames = [key for key in rows[0] if key != "channel_counts"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items() if key != "channel_counts"} for row in rows)


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
    parser.add_argument("--tables", default="1")
    parser.add_argument("--mcs", default="11")
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
    parser.add_argument("--checkpoint-map", help='JSON overrides: {"1:11": {"gt": "...", "detr": "...", "flow": "optional..."}}')
    parser.add_argument("--bracket-step-db", type=float, default=0.5)
    parser.add_argument("--max-bracket-extensions", type=int, default=8, help="Maximum additional formal SNR points in total")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing result files in output-dir")
    args = parser.parse_args()
    if not math.isfinite(args.target_bler) or not 0 < args.target_bler < 1:
        parser.error("--target-bler must be finite and in (0,1)")
    for name in ("coarse_channels", "formal_channels", "bp_iters", "gt_chunk"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("formal_step_db", "bracket_step_db"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.formal_margin_db) or args.formal_margin_db < 0 or args.max_bracket_extensions < 0:
        parser.error("Formal margin and bracket extension limit must be nonnegative")
    if args.operating_point and args.skip_formal:
        parser.error("--operating-point and --skip-formal are mutually exclusive")

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

    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        source_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        source_commit, source_dirty = None, None

    train_cfg = load_yaml(args.config)
    base_cfg = load_yaml(train_cfg["system_config"])

    device = base_cfg["general"]["device"]

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device unavailable; run this 256Rx evaluation on the GPU server.")

    sionna_config.device = device
    sionna_config.precision = base_cfg["general"]["precision"]
    torch.set_float32_matmul_precision("high")

    tables = parse_int_list(args.tables)
    mcs_indices = parse_int_list(args.mcs)
    coarse_snrs = sorted(set(parse_float_list(args.coarse_snrs)))
    if not tables or not mcs_indices or len(set(tables)) != len(tables) or len(set(mcs_indices)) != len(mcs_indices):
        parser.error("Tables/MCS lists must be nonempty and contain no duplicates")
    if not set(tables) <= {1, 2}:
        parser.error("Only MCS tables 1 and 2 are supported")
    if not coarse_snrs or not all(math.isfinite(x) for x in coarse_snrs):
        parser.error("Coarse SNR grid must be nonempty and finite")
    checkpoint_map = read_checkpoint_map(args.checkpoint_map)
    for table in tables:
        for mcs in mcs_indices:
            qm, _ = decode_mcs_index(mcs, table_index=table, is_pusch=True, transform_precoding=False, device=device)
            if as_int(qm) not in MOD_NAMES:
                parser.error(f"Unsupported modulation for T{table} MCS{mcs}")
            if args.operating_point and (table, mcs) not in OPERATING_SNRS:
                parser.error(f"No operating SNR for T{table} MCS{mcs}; use crossing mode")
            if not args.baseline_only:
                if (table, mcs) not in checkpoint_map:
                    parser.error(f"No specialist mapping for T{table} MCS{mcs}; supply --checkpoint-map")
                for path in checkpoint_map[(table, mcs)].values():
                    if not Path(path).is_file():
                        raise FileNotFoundError(f"Missing specialist checkpoint: {path}")
    output_dir = Path(args.output_dir)
    output_files = [output_dir / f"mcs_bler_{suffix}" for suffix in ("results.json", "curves.csv", "summary.csv")]
    if not args.overwrite and any(path.exists() for path in output_files):
        raise FileExistsError("Result files exist; choose another --output-dir or explicitly use --overwrite")
    covariance_path = Path("data/cache/uma_lmmse_ft_cov.pt")
    covariance_hash = hashlib.sha256(covariance_path.read_bytes()).hexdigest()

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
            "config": {**vars(args), "system": base_cfg, "source_commit": source_commit, "source_dirty": source_dirty,
                       "covariance_cache": {"path": str(covariance_path), "sha256": covariance_hash},
                       "python": sys.version, "torch": torch.__version__,
                       "sionna": importlib.metadata.version("sionna")},
            "results": all_results,
        }

        output_path = output_dir / "mcs_bler_results.json"
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output_path)

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

            key = (table, mcs)

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
                        checkpoint_map,
                    )
                models = model_cache[key]

            detector_names = ["lmmse", "ep5"] + [
                name
                for name in ("gt", "detr", "flow")
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

            metadata = {
                "mode": "operating_point" if args.operating_point else "coarse_diagnostic" if args.skip_formal else "formal",
                "table": table, "mcs": mcs, "modulation": MOD_NAMES[qm], "qm": qm,
                "target_rate": runtime.target_rate, "spectral_efficiency": runtime.spectral_efficiency,
                "tb_size": runtime.tb_size, "num_coded_bits": runtime.num_coded_bits,
                "seed": common_seed, "target_bler": args.target_bler,
                "checkpoints": {name: model.checkpoint_metadata for name, model in models.items()},
            }

            if args.operating_point:
                result = {**metadata, "status": "running"}
                all_results.append(result)
                save_progress()
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

                result.update(point)
                result["status"] = "complete"
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
            formal_points = []
            result = {**metadata, "status": "running", "coarse_points": coarse_points, "formal_points": formal_points}
            all_results.append(result)
            save_progress()

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
                save_progress()

                status = " | ".join(
                    f"{DETECTOR_LABELS[name]}="
                    f"{point[f'{name}_bler']:.5f}"
                    for name in detector_names
                )

                extra = []

                for name in ("gt", "detr", "flow"):
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
                result["formal_points"] = formal_points

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

                formal_points = result["formal_points"]

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
                    save_progress()

            def evaluate_extension(snr):
                print(f"  BRACKET EXTENSION SNR={snr:+.3f} dB")
                return evaluate_snr_point(base_cfg, runtime, models, snr, args.formal_channels,
                                          common_seed, args.gt_chunk, verbose=True)

            result["warnings"] = extend_bler_grid(
                formal_points, [f"{name}_bler" for name in detector_names], args.target_bler,
                args.bracket_step_db, 0 if args.skip_formal else args.max_bracket_extensions,
                evaluate_extension, on_progress=save_progress,
            )

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

            result.update({
                "status": "complete" if all(x is not None for x in snr_at_target.values()) else "unresolved",
                "snr_at_target_db": snr_at_target,
                "gain_vs_ep5_db": gain_vs_ep,
            })

            summary = {**metadata, "status": result["status"], "warnings": result["warnings"]}

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
                for name in ("gt", "detr", "flow")
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
                for name in ("gt", "detr", "flow")
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