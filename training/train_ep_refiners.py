#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.gt_ep_residual import ResidualGraphTransformerEPDetector
from models.baselines.detr_ep_detector import DETREPDetector
from training.train_gt_ep_detector import sample_re, build_validation_set, validate, check_exact_ep_anchor, hard_errors


MOD_NAMES = {2: "qpsk", 4: "16qam", 6: "64qam"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_model(cfg, arch):
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
    if arch == "residual_gt":
        return ResidualGraphTransformerEPDetector(num_layers=4, edge_dim=32, **common)
    if arch == "detr_ep":
        return DETREPDetector(num_layers=3, **common)
    raise ValueError(f"Unknown arch: {arch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--arch", choices=["residual_gt", "detr_ep"], required=True)
    parser.add_argument("--bits-per-symbol", type=int, choices=[2, 4, 6], required=True)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scheduler-steps", type=int, default=1000)
    parser.add_argument("--re-per-step", type=int, default=128)
    parser.add_argument("--val-channels", type=int, default=32)
    parser.add_argument("--val-re-per-channel", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])
    bps = int(args.bits_per_symbol)
    mod_name = MOD_NAMES[bps]
    cfg["modulation"]["bits_per_symbol"] = bps
    cfg["link"]["rx_snr_db"] = float(args.snr_db)
    cfg["general"]["seed"] = int(args.seed)

    device = cfg["general"]["device"]
    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(args.seed)
    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    tag = "residual_gt_ep" if args.arch == "residual_gt" else "detr_ep"
    output_dir = Path(args.output_dir or f"ckp/{tag}_256rx_16ue_{mod_name}_lsH_estR_5db")
    history_path = Path(args.history or f"results/raw/{tag}_256rx_16ue_{mod_name}_lsH_estR_5db_history.json")

    if args.fresh:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if history_path.exists():
            history_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pth"

    dataset = UplinkUMADataset(cfg)
    model = make_model(cfg, args.arch).to(device)
    classical_ep = ExpectationPropagationDetector(cfg, num_iterations=5, damping=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.scheduler_steps), eta_min=args.lr * 0.05)
    num_params = sum(p.numel() for p in model.parameters())

    print("=" * 128)
    print(f"EP NEURAL REFINER TRAINING | {args.arch}")
    print("=" * 128)
    print("System              : 256 Rx / 16 streams")
    print(f"Modulation          : {mod_name.upper()}")
    print(f"Training SNR        : {args.snr_db:.1f} dB")
    print("CSI / covariance    : LS H / estimated Ruu")
    print(f"Parameters          : {num_params:,}")
    print(f"Steps               : {args.steps}")
    print(f"Scheduler horizon   : {args.scheduler_steps}")
    print(f"RE / step           : {args.re_per_step}")
    print(f"Validation          : {args.val_channels} ch × {args.val_re_per_channel} RE")
    print(f"Output              : {output_dir}")
    print("=" * 128)

    sionna_config.seed = int(args.seed) + 2000
    torch.manual_seed(int(args.seed) + 2000)
    validation = build_validation_set(dataset, args.val_channels, args.val_re_per_channel)
    check_exact_ep_anchor(model, classical_ep, validation, device)
    initial = validate(model, classical_ep, validation, device)

    print(f"Initial | EP5={initial['ber_ep5']:.8e} | Neural={initial['ber_gt']:.8e} | TrueH={initial['ber_trueH_ep5']:.8e}")
    best_ber = initial["ber_gt"]
    best_step = 0
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": 0,
        "best_ber": best_ber, "arch": args.arch, "bits_per_symbol": bps, "modulation": mod_name,
        "metrics": initial, "args": vars(args),
    }, checkpoint_path)

    history = [{"step": 0, "lr": args.lr, "arch": args.arch, "bits_per_symbol": bps, "modulation": mod_name, **initial}]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    running_loss = 0.0
    running_ber = 0.0
    running_count = 0
    sionna_config.seed = int(args.seed)
    torch.manual_seed(int(args.seed))

    for step in range(1, args.steps + 1):
        model.train()
        sample = sample_re(dataset, args.re_per_step)
        z, gram, bits = sample["z_ls"], sample["gram_ls"], sample["bits"]
        if bits.shape[-1] != bps:
            raise RuntimeError(f"Dataset returned {bits.shape[-1]} bits/symbol, expected {bps}.")

        optimizer.zero_grad(set_to_none=True)
        out = model(z, gram, return_iterations=(5,))
        llr = out["llr"]
        if llr.shape != bits.shape:
            raise RuntimeError(f"LLR shape {tuple(llr.shape)} != bits shape {tuple(bits.shape)}.")
        loss = F.binary_cross_entropy_with_logits(llr, bits)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {float(grad_norm)}")
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            batch_ber = hard_errors(llr, bits) / bits.numel()
        running_loss += loss.item()
        running_ber += batch_ber
        running_count += 1

        if step % 25 == 0 or step == args.steps:
            residual_text = ""
            if "residual_rms" in out:
                residual_text = f" | residual={out['residual_rms'].mean().item():.3f}"
            print(
                f"step={step:4d} | BCE={running_loss / running_count:.5f} | BER={running_ber / running_count:.5f} | "
                f"grad={float(grad_norm):.3f} | corr={out['correction_rms'].mean().item():.3f} | "
                f"valid={out['valid_update_fraction'].mean().item():.3f}{residual_text} | lr={optimizer.param_groups[0]['lr']:.3e}"
            )
            running_loss = 0.0
            running_ber = 0.0
            running_count = 0

        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(model, classical_ep, validation, device)
            corr_text = ",".join(f"{v:.3f}" for v in metrics["correction_rms"])
            valid_text = ",".join(f"{v:.3f}" for v in metrics["valid_update_fraction"])
            print(
                f"VAL step={step:4d} | EP5={metrics['ber_ep5']:.6e} | Neural={metrics['ber_gt']:.6e} | "
                f"TrueH={metrics['ber_trueH_ep5']:.6e} | gain={100.0 * metrics['relative_gain_vs_ep5']:+.3f}% | "
                f"recover={100.0 * metrics['oracle_gap_recovered']:+.2f}% | corr=[{corr_text}] | valid=[{valid_text}]"
            )
            history.append({"step": step, "lr": optimizer.param_groups[0]["lr"], "arch": args.arch, "bits_per_symbol": bps, "modulation": mod_name, **metrics})
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if metrics["ber_gt"] < best_ber:
                best_ber = metrics["ber_gt"]
                best_step = step
                torch.save({
                    "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step,
                    "best_ber": best_ber, "arch": args.arch, "bits_per_symbol": bps, "modulation": mod_name,
                    "metrics": metrics, "args": vars(args),
                }, checkpoint_path)
                print(f"  NEW BEST | BER={best_ber:.8e} @ step={best_step}")

    print()
    print("=" * 128)
    print("TRAINING FINISHED")
    print("=" * 128)
    print(f"Architecture        : {args.arch}")
    print(f"Modulation          : {mod_name.upper()}")
    print(f"Initial EP5 BER     : {initial['ber_ep5']:.8e}")
    print(f"Best neural BER     : {best_ber:.8e}")
    print(f"Best step           : {best_step}")
    print(f"Relative gain       : {100.0 * (initial['ber_ep5'] - best_ber) / max(initial['ber_ep5'], 1e-12):+.3f}%")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"History             : {history_path}")
    print("=" * 128)


if __name__ == "__main__":
    main()
