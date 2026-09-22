#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import json
import hashlib
import math
import warnings
import importlib.metadata
import sys
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.lmmse import LMMSESoftDetector
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.gt_ep_detector import GraphTransformerEPDetector
from models.baselines.detr_ep_detector import DETREPDetector


MOD_NAMES = {2: "qpsk", 4: "16qam", 6: "64qam"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hard_errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


def load_checkpoint(model, path, device, arch):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    expected = {"arch": arch, "bits_per_symbol": model.bits_per_symbol, "csi": "lmmseH", "covariance": "estimated_Ruu"}
    missing = []
    for key, value in expected.items():
        if key not in ckpt:
            missing.append(key)
        elif ckpt[key] != value:
            raise ValueError(f"Checkpoint {path}: {key}={ckpt[key]!r}, expected {value!r}")
    if missing:
        warnings.warn(f"Legacy checkpoint {path}: unverified metadata {missing}", RuntimeWarning)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "unverified_fields": missing,
            **{key: ckpt.get(key) for key in ("step", "best_ber", "arch", "csi", "covariance", "bits_per_symbol", "mcs_table", "mcs_index", "args")}}


def make_models(cfg, device, gt_path, detr_path, flow_path=None):
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
    gt = GraphTransformerEPDetector(
        num_layers=4,
        edge_dim=32,
        edge_mode="full",
        message_mode="cross_user",
        **common,
    ).to(device)
    detr = DETREPDetector(num_layers=3, **common).to(device)

    meta = {
        "gt_ep": load_checkpoint(gt, gt_path, device, "gt_ep"),
        "detr_ep": load_checkpoint(detr, detr_path, device, "detr_ep"),
    }
    models = {"gt_ep": gt, "detr_ep": detr}
    if flow_path is not None:
        from models.baselines.flow_matching_detector import flow_from_checkpoint
        checkpoint = torch.load(flow_path, map_location="cpu", weights_only=False)
        flow = flow_from_checkpoint(cfg, checkpoint).to(device).eval()
        models["flow_matching"] = flow
        meta["flow_matching"] = load_checkpoint(flow, flow_path, device, "flow_matching")
        meta["flow_matching"]["model_config"] = flow.model_config
    return models, meta


def paired_bootstrap_ci(reference, candidate, num_bootstrap=20000, seed=20260915, batch_size=2000):
    reference = torch.as_tensor(reference, dtype=torch.float64, device="cpu")
    candidate = torch.as_tensor(candidate, dtype=torch.float64, device="cpu")
    delta = reference - candidate
    n = delta.numel()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    means = []
    for start in range(0, int(num_bootstrap), int(batch_size)):
        count = min(int(batch_size), int(num_bootstrap) - start)
        idx = torch.randint(0, n, (count, n), generator=generator)
        means.append(delta[idx].mean(dim=1))
    means = torch.cat(means)
    q = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=means.dtype))
    return delta.mean().item(), q[0].item(), q[1].item()


def sample_channel_re(dataset, frontend, ep, re_per_channel, re_generator):
    batch = dataset.sample(1)
    data_idx = list(dataset.channel.resource_grid.data_symbols)
    y = batch["y"][:, data_idx]
    h_lmmse = batch["h_hat_lmmse"][:, data_idx]
    h_true = batch["h_true"][:, data_idx]
    bits = batch["bits"]

    physical = frontend(y, h_lmmse, batch["ruu_hat"])
    true_physical = frontend(y, h_true, batch["ruu_hat"])

    num_streams = int(batch["metadata"]["num_streams"])
    bits_per_symbol = int(batch["metadata"]["bits_per_symbol"])

    z = physical["z"].reshape(-1, num_streams)
    gram = physical["gram"].reshape(-1, num_streams, num_streams)
    lmmse_llr = physical["llr"].reshape(-1, num_streams, bits_per_symbol)
    z_true = true_physical["z"].reshape(-1, num_streams)
    gram_true = true_physical["gram"].reshape(-1, num_streams, num_streams)
    bits = bits.reshape(-1, num_streams, bits_per_symbol).float()

    count = min(int(re_per_channel), z.shape[0])
    idx = torch.randperm(z.shape[0], generator=re_generator)[:count].to(z.device)

    return {
        "z": z[idx],
        "gram": gram[idx],
        "lmmse_llr": lmmse_llr[idx],
        "true_ep5_llr": ep(z_true[idx], gram_true[idx], return_iterations=(5,))["llr"],
        "bits": bits[idx],
    }


def default_output_path(args):
    snr = f"{args.snr_db:g}".replace("-", "m").replace(".", "p")
    controls = "gt_vs_detr_vs_flow" if getattr(args, "flow_checkpoint", None) else "gt_vs_detr"
    return Path(f"results/ep_refiners/lmmseH_{controls}_{MOD_NAMES[args.bits_per_symbol]}_"
                f"{args.channels}ch_{args.re_per_channel}re_{snr}db_seed{args.seed}.json")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--bits-per-symbol", type=int, choices=[2, 4, 6], default=4)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--channels", type=int, default=500)
    parser.add_argument("--re-per-channel", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--gt-checkpoint", required=True)
    parser.add_argument("--detr-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", help="Optional trained masked Flow control")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("channels", "re_per_channel", "chunk_size", "print_every", "bootstrap"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.snr_db):
        parser.error("--snr-db must be finite")
    output = Path(args.output) if args.output else default_output_path(args)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Result exists: {output}; choose --output or explicitly use --overwrite")
    for path in [args.gt_checkpoint, args.detr_checkpoint] + ([args.flow_checkpoint] if args.flow_checkpoint else []):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        source_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        source_commit, source_dirty = None, None

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])
    cfg["modulation"]["bits_per_symbol"] = int(args.bits_per_symbol)
    cfg["link"]["rx_snr_db"] = float(args.snr_db)
    cfg["general"]["seed"] = int(args.seed)

    device = cfg["general"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device unavailable; run this 256Rx evaluation on the GPU server.")
    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(args.seed)
    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    dataset = UplinkUMADataset(cfg)
    frontend = LMMSESoftDetector(cfg)
    ep = ExpectationPropagationDetector(cfg, num_iterations=5, damping=0.5)
    models, checkpoint_meta = make_models(cfg, device, args.gt_checkpoint, args.detr_checkpoint, args.flow_checkpoint)

    names = ["lmmse", "ep5", "gt_ep", "detr_ep"]
    if "flow_matching" in models:
        names += ["raw_flow", "flow_matching"]
    names += ["trueH_ep5"]

    total_errors = {name: 0 for name in names}
    channel_bers = {name: [] for name in names}
    total_bits = 0
    re_generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 99991)

    print("=" * 136)
    print("LMMSE-CSI INDEPENDENT EVALUATION | SAME CHANNELS / SAME REs")
    print("=" * 136)
    print("System              : 256 Rx / 16 streams")
    print(f"Modulation          : {MOD_NAMES[args.bits_per_symbol].upper()}")
    print(f"SNR                 : {args.snr_db:.1f} dB")
    print("CSI / covariance    : LMMSE H / estimated Ruu")
    print(f"Independent set     : {args.channels} channels × {args.re_per_channel} RE")
    print("Parameters          : " + " | ".join(
        f"{name}={sum(p.numel() for p in model.parameters()):,}" for name, model in models.items()
    ))
    for name, meta in checkpoint_meta.items():
        print(f"Checkpoint {name:8s}: step={meta.get('step')} | best={meta.get('best_ber')}")
    print("=" * 136)

    for ch in range(1, args.channels + 1):
        sample = sample_channel_re(dataset, frontend, ep, args.re_per_channel, re_generator)
        z, gram, bits = sample["z"], sample["gram"], sample["bits"]

        per_channel_errors = {
            "lmmse": hard_errors(sample["lmmse_llr"], bits),
            "ep5": hard_errors(ep(z, gram, return_iterations=(5,))["llr"], bits),
            "trueH_ep5": hard_errors(sample["true_ep5_llr"], bits),
        }

        for name, model in models.items():
            errors = 0
            raw_errors = 0 if name == "flow_matching" else None

            for start in range(0, z.shape[0], args.chunk_size):
                stop = min(start + args.chunk_size, z.shape[0])
                out = model(z[start:stop], gram[start:stop], return_iterations=(5,))
                llr = out["llr"]

                if llr.shape != bits[start:stop].shape or not torch.isfinite(llr).all():
                    raise RuntimeError(f"{name}: invalid LLR shape or nonfinite output")
                errors += hard_errors(llr, bits[start:stop])

                if name == "flow_matching":
                    raw_llr = out["raw_flow_llr"]
                    if raw_llr.shape != bits[start:stop].shape or not torch.isfinite(raw_llr).all():
                        raise RuntimeError("raw_flow: invalid LLR shape or nonfinite output")
                    raw_errors += hard_errors(raw_llr, bits[start:stop])

            per_channel_errors[name] = errors
            if name == "flow_matching":
                per_channel_errors["raw_flow"] = raw_errors

        nbits = bits.numel()
        total_bits += nbits
        for name in names:
            total_errors[name] += per_channel_errors[name]
            channel_bers[name].append(per_channel_errors[name] / nbits)

        if ch % args.print_every == 0 or ch == args.channels:
            current = {name: total_errors[name] / total_bits for name in names}
            print(
                f"channel={ch:4d}/{args.channels} | LMMSE={current['lmmse']:.6e} | "
                f"EP5={current['ep5']:.6e} | GT={current['gt_ep']:.6e} | "
                f"DETR={current['detr_ep']:.6e}"
                + (
                    f" | RawFlow={current['raw_flow']:.6e} | Flow={current['flow_matching']:.6e}"
                    if "flow_matching" in current else ""
                )
                + f" | TrueH={current['trueH_ep5']:.6e}"
            )

    ber = {name: total_errors[name] / total_bits for name in names}
    ep5 = ber["ep5"]
    lmmse = ber["lmmse"]

    print()
    print("=" * 136)
    print("FINAL BER")
    print("=" * 136)
    print(f"{'Detector':20s} {'BER':>14s} {'gain vs EP5':>16s} {'gain vs LMMSE':>18s}")
    print("-" * 136)
    rows = [
        ("lmmse", "LMMSE"),
        ("ep5", "EP5"),
        ("gt_ep", "GT-EP"),
        ("detr_ep", "DETR-EP"),
    ]
    if "flow_matching" in models:
        rows += [
            ("raw_flow", "Raw-Flow"),
            ("flow_matching", "Flow-Matching"),
        ]
    rows += [("trueH_ep5", "TrueH+EstR EP5")]

    for key, label in rows:
        gain_ep = 100.0 * (ep5 - ber[key]) / max(ep5, 1e-12)
        gain_lm = 100.0 * (lmmse - ber[key]) / max(lmmse, 1e-12)
        print(f"{label:20s} {ber[key]:14.8e} {gain_ep:+15.3f}% {gain_lm:+17.3f}%")

    comparisons = [
        ("gt_ep", "ep5", "GT-EP vs EP5"),
        ("detr_ep", "ep5", "DETR-EP vs EP5"),
        ("detr_ep", "gt_ep", "DETR-EP vs GT-EP"),
    ]
    if "flow_matching" in models:
        comparisons.extend([
            ("raw_flow", "ep5", "Raw-Flow vs EP5"),
            ("raw_flow", "gt_ep", "Raw-Flow vs GT-EP"),
            ("raw_flow", "detr_ep", "Raw-Flow vs DETR-EP"),
            ("flow_matching", "ep5", "Flow vs EP5"),
            ("flow_matching", "gt_ep", "Flow vs GT-EP"),
            ("flow_matching", "detr_ep", "Flow vs DETR-EP"),
            ("raw_flow", "flow_matching", "Raw-Flow vs Flow"),
        ])
    ci_results = {}
    print()
    print("PAIRED CHANNEL BOOTSTRAP 95% CI | positive ΔBER means candidate is better")
    print("-" * 136)

    for i, (candidate, reference, label) in enumerate(comparisons):
        mean_delta, lo, hi = paired_bootstrap_ci(
            channel_bers[reference],
            channel_bers[candidate],
            args.bootstrap,
            args.seed + 1000 + i,
        )
        relative = 100.0 * (ber[reference] - ber[candidate]) / max(ber[reference], 1e-12)
        verdict = "ROBUST+" if lo > 0 else ("ROBUST-" if hi < 0 else "CI crosses 0")
        ci_results[label] = {
            "candidate": candidate,
            "reference": reference,
            "delta_ber": mean_delta,
            "ci95_low": lo,
            "ci95_high": hi,
            "relative_gain_percent": relative,
            "verdict": verdict,
        }
        print(
            f"{label:24s} ΔBER={mean_delta:+.8e} [{lo:+.8e}, {hi:+.8e}] | "
            f"rel={relative:+.3f}% | {verdict}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "setting": {
                "config": args.config,
                "bits_per_symbol": args.bits_per_symbol,
                "snr_db": args.snr_db,
                "channels": args.channels,
                "re_per_channel": args.re_per_channel,
                "seed": args.seed,
                "csi": "LMMSE H",
                "covariance": "estimated Ruu",
                "total_bits": total_bits,
                "actual_re_per_channel": int(sample["bits"].shape[0]),
                "args": vars(args), "system": cfg, "source_commit": source_commit, "source_dirty": source_dirty,
                "python": sys.version, "torch": torch.__version__, "sionna": importlib.metadata.version("sionna"),
                "covariance_cache": {"path": "data/cache/uma_lmmse_ft_cov.pt",
                                     "sha256": hashlib.sha256(Path("data/cache/uma_lmmse_ft_cov.pt").read_bytes()).hexdigest()},
            },
            "checkpoints": checkpoint_meta,
            "ber": ber,
            "errors": total_errors,
            "paired_ci": ci_results,
            "channel_ber": channel_bers,
        }, f, indent=2)

    print("=" * 136)
    print(f"Saved               : {output}")
    print("=" * 136)


if __name__ == "__main__":
    main()
