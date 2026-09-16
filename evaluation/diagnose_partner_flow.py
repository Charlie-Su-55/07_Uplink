#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.lmmse import LMMSESoftDetector
from detectors.classical.ep import ExpectationPropagationDetector
from training.train_gt_detr_lmmse import make_model


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hard_errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


def infer_platform_constellation(batch, qm):
    x = batch["x_data"].reshape(-1).to(torch.complex64)
    bits = batch["bits"].reshape(-1, qm).round().long()

    if x.shape[0] != bits.shape[0]:
        raise RuntimeError(f"x_data / bits mismatch: {x.shape[0]} vs {bits.shape[0]}")

    weights = (2 ** torch.arange(qm - 1, -1, -1, device=bits.device)).long()
    labels = (bits * weights).sum(-1)

    points = []
    bit_table = []

    for index in range(2 ** qm):
        mask = labels == index

        if not bool(mask.any()):
            raise RuntimeError(f"Constellation label {index} did not appear in this batch.")

        symbols = x[mask]
        point = symbols[0]
        max_error = (symbols - point).abs().max().item()

        if max_error > 1e-6:
            raise RuntimeError(
                f"Label {index} maps to inconsistent constellation points: {max_error:.3e}"
            )

        points.append(point)
        bit_table.append(bits[mask][0])

    points = torch.stack(points).to(torch.complex64)
    bit_table = torch.stack(bit_table).to(torch.float32)

    return points, bit_table


def load_neural_model(cfg, arch, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    model = make_model(cfg, arch)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    return model, checkpoint


@torch.inference_mode()
def run_ep_chunks(ep, z, gram, output_shape, chunk=256):
    z_flat = z.reshape(-1, z.shape[-1])
    gram_flat = gram.reshape(-1, gram.shape[-2], gram.shape[-1])

    outputs = []

    for start in range(0, z_flat.shape[0], chunk):
        stop = min(start + chunk, z_flat.shape[0])

        out = ep(
            z_flat[start:stop],
            gram_flat[start:stop],
            return_iterations=(5,),
        )

        outputs.append(out["llr"].float())

    return torch.cat(outputs, dim=0).reshape(output_shape)


@torch.inference_mode()
def run_neural_chunks(model, z, gram, output_shape, chunk=256):
    z_flat = z.reshape(-1, z.shape[-1])
    gram_flat = gram.reshape(-1, gram.shape[-2], gram.shape[-1])

    outputs = []

    for start in range(0, z_flat.shape[0], chunk):
        stop = min(start + chunk, z_flat.shape[0])

        out = model(
            z_flat[start:stop],
            gram_flat[start:stop],
            return_iterations=(5,),
        )

        outputs.append(out["llr"].float())

    return torch.cat(outputs, dim=0).reshape(output_shape)


def tensor_stats(name, llr, bits):
    errors = hard_errors(llr, bits)
    total = bits.numel()
    ber = errors / total

    return {
        "name": name,
        "errors": errors,
        "total": total,
        "ber": ber,
        "mean_abs": llr.abs().mean().item(),
        "rms": llr.square().mean().sqrt().item(),
        "minimum": llr.min().item(),
        "maximum": llr.max().item(),
    }


def compare_llrs(reference, candidate):
    a = reference.float().reshape(-1)
    b = candidate.float().reshape(-1)

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    corr = (
        (a_centered * b_centered).sum()
        / (
            a_centered.square().sum().sqrt()
            * b_centered.square().sum().sqrt()
        ).clamp_min(1e-12)
    ).item()

    hard_disagreement = ((a > 0) != (b > 0)).float().mean().item()

    alpha = (
        (a * b).sum()
        / b.square().sum().clamp_min(1e-12)
    ).item()

    scaled_mae = (a - alpha * b).abs().mean().item()
    raw_mae = (a - b).abs().mean().item()

    return {
        "corr": corr,
        "hard_disagreement": hard_disagreement,
        "alpha": alpha,
        "raw_mae": raw_mae,
        "scaled_mae": scaled_mae,
    }


def print_detector_table(stats):
    print()
    print("=" * 126)
    print("SAME-BATCH DETECTOR BER")
    print("=" * 126)
    print(
        f"{'Detector':<24}"
        f"{'BER':>14}"
        f"{'Errors':>12}"
        f"{'Mean|LLR|':>14}"
        f"{'RMS LLR':>14}"
        f"{'LLR min':>14}"
        f"{'LLR max':>14}"
    )
    print("-" * 126)

    for item in stats:
        print(
            f"{item['name']:<24}"
            f"{item['ber']:>14.8e}"
            f"{item['errors']:>12d}"
            f"{item['mean_abs']:>14.5f}"
            f"{item['rms']:>14.5f}"
            f"{item['minimum']:>14.5f}"
            f"{item['maximum']:>14.5f}"
        )

    print("-" * 126)


def print_alignment(name, result):
    print(
        f"{name:<28}"
        f"corr={result['corr']:+.6f} | "
        f"hard disagree={100.0 * result['hard_disagreement']:.3f}% | "
        f"scale={result['alpha']:+.5f} | "
        f"MAE={result['raw_mae']:.5f} | "
        f"scaled MAE={result['scaled_mae']:.5f}"
    )


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
        "--gt-checkpoint",
        default="ckp/ruu_mismatch/16qam_8db_lmmseH_hatR_gt_ep/best.pth",
    )

    parser.add_argument(
        "--detr-checkpoint",
        default="ckp/ruu_mismatch/16qam_8db_lmmseH_hatR_detr_ep/best.pth",
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

    parser.add_argument(
        "--chunk",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()

    if not bundle.is_dir():
        raise FileNotFoundError(bundle)

    sys.path.insert(0, str(bundle))

    from example_minimal_integration import PartnerFlowDetector

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])

    cfg["modulation"]["bits_per_symbol"] = int(args.qm)
    cfg["link"]["rx_snr_db"] = float(args.snr_db)
    cfg["general"]["seed"] = int(args.seed)
    cfg["general"]["device"] = args.device

    device = torch.device(args.device)

    sionna_config.device = args.device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(args.seed)

    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    print("=" * 126)
    print("PARTNER FLOW DIAGNOSTIC | SAME PHYSICAL BATCH")
    print("=" * 126)
    print(f"Modulation             : Qm={args.qm} / {2 ** args.qm}-QAM")
    print(f"SNR                    : {args.snr_db:.1f} dB")
    print(f"CSI                    : LMMSE H")
    print(f"Covariance             : estimated Ruu")
    print(f"Flow bundle            : {bundle}")
    print(f"GT checkpoint          : {args.gt_checkpoint}")
    print(f"DETR checkpoint        : {args.detr_checkpoint}")
    print("=" * 126)

    dataset = UplinkUMADataset(cfg)

    frontend = LMMSESoftDetector(cfg)

    ep5 = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    gt, gt_ckpt = load_neural_model(
        cfg,
        "gt_ep",
        args.gt_checkpoint,
        device,
    )

    detr, detr_ckpt = load_neural_model(
        cfg,
        "detr_ep",
        args.detr_checkpoint,
        device,
    )

    print(
        f"Loaded GT checkpoint   : step={gt_ckpt.get('step', '?')} "
        f"| saved BER={gt_ckpt.get('best_ber', '?')}"
    )

    print(
        f"Loaded DETR checkpoint : step={detr_ckpt.get('step', '?')} "
        f"| saved BER={detr_ckpt.get('best_ber', '?')}"
    )

    batch = dataset.sample(1)

    data_idx = list(
        dataset.channel.resource_grid.data_symbols
    )

    y = (
        batch["y"][:, data_idx]
        .to(device=device, dtype=torch.complex64)
        .contiguous()
    )

    h = (
        batch["h_hat_lmmse"][:, data_idx]
        .to(device=device, dtype=torch.complex64)
        .contiguous()
    )

    ruu = (
        batch["ruu_hat"]
        .to(device=device)
        .contiguous()
    )

    bits = (
        batch["bits"]
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )

    expected = (
        y.shape[0],
        y.shape[1],
        y.shape[2],
        h.shape[-1],
        args.qm,
    )

    print()
    print("Physical tensors:")
    print(f"  y    : {tuple(y.shape)} | {y.dtype}")
    print(f"  h    : {tuple(h.shape)} | {h.dtype}")
    print(f"  ruu  : {tuple(ruu.shape)} | {ruu.dtype}")
    print(f"  bits : {tuple(bits.shape)} | {bits.dtype}")

    if tuple(bits.shape) != expected:
        raise RuntimeError(
            f"Unexpected bit shape {tuple(bits.shape)}, expected {expected}"
        )

    platform_points, platform_bits = infer_platform_constellation(
        batch,
        args.qm,
    )

    print()
    print(
        f"Platform constellation : {len(platform_points)} points | "
        f"average power={platform_points.abs().square().mean().item():.6f}"
    )

    flow = PartnerFlowDetector(
        Qm=args.qm,
        bundle_dir=bundle,
        device=device,
        llr_convention="log_p1_over_p0",
        seed=args.seed,
        constellation_points=platform_points,
        constellation_bits=platform_bits,
    )

    print()
    print("Running our LMMSE frontend...")

    with torch.inference_mode():
        physical = frontend(
            y,
            h,
            ruu,
        )

    llr_lmmse = physical["llr"].float()

    z = physical["z"]
    gram = physical["gram"]

    if tuple(llr_lmmse.shape) != expected:
        raise RuntimeError(
            f"Our LMMSE output shape {tuple(llr_lmmse.shape)} != {expected}"
        )

    print("Running EP5...")

    llr_ep5 = run_ep_chunks(
        ep5,
        z,
        gram,
        expected,
        chunk=args.chunk,
    )

    print("Running GT-EP...")

    llr_gt = run_neural_chunks(
        gt,
        z,
        gram,
        expected,
        chunk=args.chunk,
    )

    print("Running DETR-EP...")

    llr_detr = run_neural_chunks(
        detr,
        z,
        gram,
        expected,
        chunk=args.chunk,
    )

    print("Running Partner Flow components...")

    # Diagnostic use of the underlying detector:
    # detect() returns final fused LLR, raw Flow LLR, and the internal LMMSE LLR.
    with torch.inference_mode():
        flow_out = flow._detector.detect(
            y,
            h,
            ruu,
        )

    llr_flow = flow_out["llr"].float()
    llr_flow_raw = flow_out["raw_flow_llr"].float()
    llr_flow_lmmse = flow_out["lmmse_llr"].float()

    for name, value in {
        "Flow final": llr_flow,
        "Flow raw": llr_flow_raw,
        "Flow internal LMMSE": llr_flow_lmmse,
    }.items():
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"{name} shape {tuple(value.shape)} != {expected}"
            )

        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"{name} contains NaN/Inf"
            )

    stats = [
        tensor_stats(
            "Our LMMSE",
            llr_lmmse,
            bits,
        ),
        tensor_stats(
            "EP5",
            llr_ep5,
            bits,
        ),
        tensor_stats(
            "GT-EP",
            llr_gt,
            bits,
        ),
        tensor_stats(
            "DETR-EP",
            llr_detr,
            bits,
        ),
        tensor_stats(
            "Flow internal LMMSE",
            llr_flow_lmmse,
            bits,
        ),
        tensor_stats(
            "Flow raw",
            llr_flow_raw,
            bits,
        ),
        tensor_stats(
            "Flow final",
            llr_flow,
            bits,
        ),
    ]

    print_detector_table(stats)

    print()
    print("=" * 126)
    print("FRONTEND ALIGNMENT DIAGNOSTIC")
    print("=" * 126)
    print("Reference = our LMMSE LLR")
    print()

    print_alignment(
        "Flow internal LMMSE",
        compare_llrs(
            llr_lmmse,
            llr_flow_lmmse,
        ),
    )

    print_alignment(
        "Flow raw",
        compare_llrs(
            llr_lmmse,
            llr_flow_raw,
        ),
    )

    print_alignment(
        "Flow final",
        compare_llrs(
            llr_lmmse,
            llr_flow,
        ),
    )

    print()
    print("=" * 126)
    print("FLOW INTERNAL EFFECT")
    print("=" * 126)

    raw_stats = compare_llrs(
        llr_flow_lmmse,
        llr_flow_raw,
    )

    final_stats = compare_llrs(
        llr_flow_lmmse,
        llr_flow,
    )

    fusion_stats = compare_llrs(
        llr_flow_raw,
        llr_flow,
    )

    print_alignment(
        "Raw vs Flow LMMSE",
        raw_stats,
    )

    print_alignment(
        "Final vs Flow LMMSE",
        final_stats,
    )

    print_alignment(
        "Final vs Raw",
        fusion_stats,
    )

    ber = {
        item["name"]: item["ber"]
        for item in stats
    }

    print()
    print("=" * 126)
    print("AUTOMATIC DIAGNOSTIC")
    print("=" * 126)

    our_lmmse = ber["Our LMMSE"]
    flow_lmmse = ber["Flow internal LMMSE"]
    flow_raw = ber["Flow raw"]
    flow_final = ber["Flow final"]

    frontend_gap = abs(
        flow_lmmse - our_lmmse
    )

    print(
        f"|Flow internal LMMSE - Our LMMSE| BER gap : "
        f"{frontend_gap:.6e}"
    )

    print(
        f"Flow raw   - Flow internal LMMSE BER delta : "
        f"{flow_raw - flow_lmmse:+.6e}"
    )

    print(
        f"Flow final - Flow internal LMMSE BER delta : "
        f"{flow_final - flow_lmmse:+.6e}"
    )

    print(
        f"Flow final - Flow raw BER delta            : "
        f"{flow_final - flow_raw:+.6e}"
    )

    print()

    alignment = compare_llrs(
        llr_lmmse,
        llr_flow_lmmse,
    )

    if (
        frontend_gap < 0.01
        and alignment["hard_disagreement"] < 0.02
    ):
        print(
            "DIAGNOSIS: Physical frontend is broadly aligned. "
            "The main degradation is likely in the pretrained Flow neural/candidate stage "
            "under this deployment distribution."
        )
    elif alignment["corr"] < 0.0:
        print(
            "DIAGNOSIS: Strong sign/bit-order inconsistency is still present. "
            "Do NOT retrain yet."
        )
    else:
        print(
            "DIAGNOSIS: Flow internal LMMSE and our LMMSE are materially different. "
            "Inspect whitening/LMMSE/noise normalization before attributing the gap "
            "to Flow training-distribution mismatch."
        )

    print("=" * 126)


if __name__ == "__main__":
    main()
