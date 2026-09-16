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
from models.graph.gt_ep_detector import GraphTransformerEPDetector
from models.baselines.detr_ep_detector import DETREPDetector
from training.train_gt_ep_detector import sample_re, build_validation_set, hard_errors


MOD_NAMES = {2: "qpsk", 4: "16qam", 6: "64qam"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def snr_tag(x):
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


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

    if arch == "gt_ep":
        return GraphTransformerEPDetector(
            num_layers=4,
            edge_dim=32,
            edge_mode="full",
            message_mode="cross_user",
            **common,
        )

    if arch == "detr_ep":
        return DETREPDetector(
            num_layers=3,
            **common,
        )

    raise ValueError(f"Unknown architecture: {arch}")


def resolve_snr_range(args):
    if args.snr_min_db is None and args.snr_max_db is None:
        snr_min = float(args.snr_db)
        snr_max = float(args.snr_db)
    elif args.snr_min_db is None or args.snr_max_db is None:
        raise ValueError("Use both --snr-min-db and --snr-max-db, or neither.")
    else:
        snr_min = float(args.snr_min_db)
        snr_max = float(args.snr_max_db)

    if snr_min > snr_max:
        raise ValueError(f"Invalid SNR range: {snr_min} > {snr_max}")

    val_snr = float(args.val_snr_db) if args.val_snr_db is not None else 0.5 * (snr_min + snr_max)
    return snr_min, snr_max, val_snr


def build_snr_schedule(num_steps, snr_min, snr_max, seed):
    if abs(snr_max - snr_min) < 1e-12:
        return [float(snr_min)] * int(num_steps)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 314159)
    values = torch.empty(int(num_steps), dtype=torch.float64)
    values.uniform_(snr_min, snr_max, generator=generator)
    return values.tolist()


@torch.no_grad()
def validate(model, classical_ep, validation, device, chunk_size=256):
    model.eval()

    total_bits = 0
    errors_ep5 = 0
    errors_neural = 0
    errors_true = 0
    correction_sum = torch.zeros(5, dtype=torch.float64)
    validity_sum = torch.zeros(5, dtype=torch.float64)
    num_chunks = 0
    total_re = validation["bits"].shape[0]

    for start in range(0, total_re, chunk_size):
        end = min(start + chunk_size, total_re)

        z = validation["z_lmmse"][start:end].to(device)
        gram = validation["gram_lmmse"][start:end].to(device)
        z_true = validation["z_true"][start:end].to(device)
        gram_true = validation["gram_true"][start:end].to(device)
        bits = validation["bits"][start:end].to(device)

        ep_out = classical_ep(z, gram, return_iterations=(5,))
        true_out = classical_ep(z_true, gram_true, return_iterations=(5,))
        neural_out = model(z, gram, return_iterations=(5,))

        errors_ep5 += hard_errors(ep_out["llr"].float(), bits)
        errors_neural += hard_errors(neural_out["llr"].float(), bits)
        errors_true += hard_errors(true_out["llr"].float(), bits)
        total_bits += bits.numel()

        correction_sum += neural_out["correction_rms"].double().cpu()
        validity_sum += neural_out["valid_update_fraction"].double().cpu()
        num_chunks += 1

    ber_ep5 = errors_ep5 / total_bits
    ber_neural = errors_neural / total_bits
    ber_true = errors_true / total_bits
    oracle_gap = ber_ep5 - ber_true

    return {
        "ber_ep5": ber_ep5,
        "ber_neural": ber_neural,
        "ber_trueH_ep5": ber_true,
        "relative_gain_vs_ep5": (ber_ep5 - ber_neural) / max(ber_ep5, 1e-12),
        "oracle_gap_recovered": 0.0 if oracle_gap <= 0 else (ber_ep5 - ber_neural) / oracle_gap,
        "correction_rms": (correction_sum / max(num_chunks, 1)).tolist(),
        "valid_update_fraction": (validity_sum / max(num_chunks, 1)).tolist(),
    }


@torch.no_grad()
def check_exact_ep_anchor(model, classical_ep, validation, device):
    model.eval()

    z = validation["z_lmmse"][:256].to(device)
    gram = validation["gram_lmmse"][:256].to(device)

    llr_ep = classical_ep(z, gram, return_iterations=(5,))["llr"]
    llr_neural = model(z, gram, return_iterations=(5,))["llr"]

    max_diff = (llr_ep - llr_neural).abs().max().item()
    print(f"Initial max |Neural-EP - EP5| LLR difference : {max_diff:.6e}")

    if max_diff > 1e-5:
        raise RuntimeError("Zero-initialized neural EP refiner does not reproduce EP5.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--arch", choices=["gt_ep", "detr_ep"], required=True)
    parser.add_argument("--bits-per-symbol", type=int, choices=[2, 4, 6], default=4)

    # Backward-compatible fixed-SNR mode.
    parser.add_argument("--snr-db", type=float, default=5.0)

    # MCS specialist range mode.
    parser.add_argument("--snr-min-db", type=float, default=None)
    parser.add_argument("--snr-max-db", type=float, default=None)
    parser.add_argument("--val-snr-db", type=float, default=None)

    parser.add_argument("--steps", type=int, default=300)
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

    snr_min, snr_max, val_snr = resolve_snr_range(args)
    train_snr_schedule = build_snr_schedule(
        args.steps,
        snr_min,
        snr_max,
        args.seed,
    )

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])

    bps = int(args.bits_per_symbol)
    mod_name = MOD_NAMES[bps]

    cfg["modulation"]["bits_per_symbol"] = bps
    cfg["link"]["rx_snr_db"] = float(val_snr)
    cfg["general"]["seed"] = int(args.seed)

    device = cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]
    torch.set_float32_matmul_precision("high")

    arch_tag = "gt_ep" if args.arch == "gt_ep" else "detr_ep"

    if abs(snr_max - snr_min) < 1e-12:
        snr_name = f"snr{snr_tag(snr_min)}db"
    else:
        snr_name = f"snr{snr_tag(snr_min)}to{snr_tag(snr_max)}db"

    output_dir = Path(
        args.output_dir
        or f"ckp/{arch_tag}_256rx_16ue_{mod_name}_lmmseH_estR_{snr_name}"
    )

    history_path = Path(
        args.history
        or f"results/raw/{arch_tag}_256rx_16ue_{mod_name}_lmmseH_estR_{snr_name}_history.json"
    )

    if args.fresh:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if history_path.exists():
            history_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pth"

    # Dataset starts at validation SNR.
    sionna_config.seed = int(args.seed)
    torch.manual_seed(int(args.seed))
    dataset = UplinkUMADataset(cfg)

    # Reproducible model initialization.
    torch.manual_seed(int(args.seed) + 100)
    model = make_model(cfg, args.arch).to(device)

    classical_ep = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.scheduler_steps),
        eta_min=args.lr * 0.05,
    )

    num_params = sum(p.numel() for p in model.parameters())

    print("=" * 132)
    print(f"MCS-REGION LMMSE-CSI EP REFINER TRAINING | {args.arch}")
    print("=" * 132)
    print("System              : 256 Rx / 16 streams")
    print(f"Modulation          : {mod_name.upper()}")
    print(f"Training SNR range  : [{snr_min:.2f}, {snr_max:.2f}] dB")
    print(f"Validation SNR      : {val_snr:.2f} dB")
    print("CSI / covariance    : LMMSE H / estimated Ruu")
    print("EP state            : covariance-whitened z,G")
    print(f"Parameters          : {num_params:,}")
    print(f"Steps               : {args.steps}")
    print(f"Scheduler horizon   : {args.scheduler_steps}")
    print(f"RE / step           : {args.re_per_step}")
    print(f"Validation          : {args.val_channels} ch × {args.val_re_per_channel} RE")
    print(f"Output              : {output_dir}")
    print("=" * 132)

    # Fixed unseen validation set at the specialist operating point.
    dataset.link.rx_snr_db = float(val_snr)
    sionna_config.seed = int(args.seed) + 2000
    torch.manual_seed(int(args.seed) + 2000)

    validation = build_validation_set(
        dataset,
        args.val_channels,
        args.val_re_per_channel,
    )

    check_exact_ep_anchor(
        model,
        classical_ep,
        validation,
        device,
    )

    initial = validate(
        model,
        classical_ep,
        validation,
        device,
    )

    print(
        f"Initial | "
        f"EP5={initial['ber_ep5']:.8e} | "
        f"Neural={initial['ber_neural']:.8e} | "
        f"TrueH={initial['ber_trueH_ep5']:.8e}"
    )

    best_ber = initial["ber_neural"]
    best_step = 0

    common_meta = {
        "arch": args.arch,
        "bits_per_symbol": bps,
        "modulation": mod_name,
        "csi": "lmmseH",
        "covariance": "estimated_Ruu",
        "snr_min_db": snr_min,
        "snr_max_db": snr_max,
        "val_snr_db": val_snr,
        "args": vars(args),
    }

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": 0,
            "best_ber": best_ber,
            "metrics": initial,
            **common_meta,
        },
        checkpoint_path,
    )

    history = [
        {
            "step": 0,
            "lr": args.lr,
            **common_meta,
            **initial,
        }
    ]

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    running_loss = 0.0
    running_ber = 0.0
    running_count = 0
    running_snrs = []

    for step in range(1, args.steps + 1):
        model.train()

        sampled_snr = float(train_snr_schedule[step - 1])

        # Runtime override only: do not modify YAML.
        dataset.link.rx_snr_db = sampled_snr

        # Same seed + same SNR schedule => GT and DETR receive
        # the same physical channel/topology/interference sequence.
        physical_seed = int(args.seed) * 100000 + step
        sionna_config.seed = physical_seed
        torch.manual_seed(physical_seed)

        sample = sample_re(
            dataset,
            args.re_per_step,
        )

        z = sample["z_lmmse"]
        gram = sample["gram_lmmse"]
        bits = sample["bits"]

        optimizer.zero_grad(set_to_none=True)

        out = model(
            z,
            gram,
            return_iterations=(5,),
        )

        llr = out["llr"]

        if llr.shape != bits.shape:
            raise RuntimeError(
                f"LLR shape {tuple(llr.shape)} != bits shape {tuple(bits.shape)}"
            )

        loss = F.binary_cross_entropy_with_logits(
            llr,
            bits,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            batch_ber = (
                hard_errors(llr, bits)
                / bits.numel()
            )

        running_loss += loss.item()
        running_ber += batch_ber
        running_count += 1
        running_snrs.append(sampled_snr)

        if step % 25 == 0 or step == args.steps:
            snr_mean = sum(running_snrs) / len(running_snrs)

            print(
                f"step={step:4d} | "
                f"SNR(mean/min/max)="
                f"{snr_mean:+.2f}/"
                f"{min(running_snrs):+.2f}/"
                f"{max(running_snrs):+.2f} dB | "
                f"BCE={running_loss / running_count:.5f} | "
                f"BER={running_ber / running_count:.5f} | "
                f"grad={float(grad_norm):.3f} | "
                f"corr={out['correction_rms'].mean().item():.3f} | "
                f"valid={out['valid_update_fraction'].mean().item():.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

            running_loss = 0.0
            running_ber = 0.0
            running_count = 0
            running_snrs = []

        if step % args.val_every == 0 or step == args.steps:
            # Validation remains fixed at val_snr.
            dataset.link.rx_snr_db = float(val_snr)

            metrics = validate(
                model,
                classical_ep,
                validation,
                device,
            )

            corr_text = ",".join(
                f"{v:.3f}"
                for v in metrics["correction_rms"]
            )

            valid_text = ",".join(
                f"{v:.3f}"
                for v in metrics["valid_update_fraction"]
            )

            print(
                f"VAL step={step:4d} | "
                f"SNR={val_snr:+.2f} dB | "
                f"EP5={metrics['ber_ep5']:.6e} | "
                f"Neural={metrics['ber_neural']:.6e} | "
                f"TrueH={metrics['ber_trueH_ep5']:.6e} | "
                f"gain={100.0 * metrics['relative_gain_vs_ep5']:+.3f}% | "
                f"recover={100.0 * metrics['oracle_gap_recovered']:+.2f}% | "
                f"corr=[{corr_text}] | "
                f"valid=[{valid_text}]"
            )

            history.append(
                {
                    "step": step,
                    "lr": optimizer.param_groups[0]["lr"],
                    **common_meta,
                    **metrics,
                }
            )

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if metrics["ber_neural"] < best_ber:
                best_ber = metrics["ber_neural"]
                best_step = step

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": step,
                        "best_ber": best_ber,
                        "metrics": metrics,
                        **common_meta,
                    },
                    checkpoint_path,
                )

                print(
                    f"  NEW BEST | "
                    f"BER={best_ber:.8e} @ step={best_step} | "
                    f"val SNR={val_snr:+.2f} dB"
                )

    print()
    print("=" * 132)
    print("TRAINING FINISHED")
    print("=" * 132)
    print(f"Architecture        : {args.arch}")
    print(f"Train SNR range     : [{snr_min:.2f}, {snr_max:.2f}] dB")
    print(f"Validation SNR      : {val_snr:.2f} dB")
    print(f"Initial EP5 BER     : {initial['ber_ep5']:.8e}")
    print(f"Best neural BER     : {best_ber:.8e}")
    print(f"Best step           : {best_step}")
    print(
        f"Relative gain       : "
        f"{100.0 * (initial['ber_ep5'] - best_ber) / max(initial['ber_ep5'], 1e-12):+.3f}%"
    )
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"History             : {history_path}")
    print("=" * 132)


if __name__ == "__main__":
    main()
