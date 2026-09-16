#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.gt_ep_detector import GraphTransformerEPDetector
from training.train_gt_ep_detector import (
    sample_re,
    build_validation_set,
    validate,
    check_exact_ep_anchor,
    hard_errors,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg, edge_mode, message_mode):
    return GraphTransformerEPDetector(
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
        edge_mode=edge_mode,
        message_mode=message_mode,
    ).to(cfg["general"]["device"])


def whiten(y, h, ruu):
    chol = torch.linalg.cholesky(ruu)

    if chol.shape[0] == 1 and y.shape[0] > 1:
        chol = chol.expand(y.shape[0], -1, -1)

    y_white = torch.linalg.solve_triangular(
        chol,
        y.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)

    h_white = torch.linalg.solve_triangular(
        chol,
        h,
        upper=False,
    )

    return y_white, h_white


def sufficient_statistics(y_white, h_white):
    z = torch.einsum(
        "bmk,bm->bk",
        h_white.conj(),
        y_white,
    )

    gram = torch.einsum(
        "bmk,bml->bkl",
        h_white.conj(),
        h_white,
    )

    return z, gram


def bootstrap(reference, method, n_boot=10000, seed=20260912):
    reference = torch.tensor(
        reference,
        dtype=torch.float64,
    )

    method = torch.tensor(
        method,
        dtype=torch.float64,
    )

    diff = reference - method

    generator = torch.Generator().manual_seed(
        int(seed)
    )

    n = diff.numel()
    samples = []

    remaining = int(n_boot)

    while remaining > 0:
        count = min(
            1000,
            remaining,
        )

        idx = torch.randint(
            0,
            n,
            (count, n),
            generator=generator,
        )

        samples.append(
            diff[idx].mean(dim=1)
        )

        remaining -= count

    samples = torch.cat(
        samples
    )

    return {
        "delta": diff.mean().item(),
        "low": torch.quantile(samples, 0.025).item(),
        "high": torch.quantile(samples, 0.975).item(),
    }


def train_self_only(base_cfg, args):
    cfg = copy.deepcopy(
        base_cfg
    )

    cfg["general"]["seed"] = int(
        args.train_seed
    )

    cfg["link"]["rx_snr_db"] = 5.0

    device = cfg["general"]["device"]

    output_dir = Path(
        args.self_output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = output_dir / "best.pth"
    history_path = output_dir / "history.json"

    print("=" * 130)
    print("TRAIN SELF-ONLY GT-EP")
    print("=" * 130)
    print(f"Seed               : {args.train_seed}")
    print("Edge mode          : no_edge")
    print("Message mode       : self_only")
    print("Cross-user messages: DISABLED")
    print(f"Steps              : {args.steps}")
    print(f"RE / step          : {args.re_per_step}")
    print("=" * 130)

    sionna_config.seed = int(
        args.train_seed
    )

    torch.manual_seed(
        int(args.train_seed)
    )

    dataset = UplinkUMADataset(
        cfg
    )

    model = build_model(
        cfg,
        edge_mode="no_edge",
        message_mode="self_only",
    )

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
        T_max=args.steps,
        eta_min=args.lr * 0.05,
    )

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters          : {num_params:,}"
    )

    sionna_config.seed = (
        int(args.train_seed) + 2000
    )

    torch.manual_seed(
        int(args.train_seed) + 2000
    )

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

    sionna_config.seed = int(
        args.train_seed
    )

    torch.manual_seed(
        int(args.train_seed)
    )

    initial = validate(
        model,
        classical_ep,
        validation,
        device,
    )

    print()
    print(
        f"Initial | "
        f"EP5={initial['ber_ep5']:.8e} | "
        f"SelfOnly={initial['ber_gt']:.8e} | "
        f"TrueH={initial['ber_trueH_ep5']:.8e}"
    )

    best_ber = initial["ber_gt"]
    best_step = 0

    torch.save({
        "model_state": model.state_dict(),
        "step": 0,
        "best_ber": best_ber,
        "edge_mode": "no_edge",
        "message_mode": "self_only",
        "seed": args.train_seed,
        "metrics": initial,
    }, checkpoint_path)

    history = [{
        "step": 0,
        "edge_mode": "no_edge",
        "message_mode": "self_only",
        "seed": args.train_seed,
        "lr": args.lr,
        **initial,
    }]

    running_loss = 0.0
    running_ber = 0.0

    for step in range(
        1,
        args.steps + 1,
    ):
        model.train()

        sample = sample_re(
            dataset,
            args.re_per_step,
        )

        z = sample["z_ls"]
        gram = sample["gram_ls"]
        bits = sample["bits"]

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            z,
            gram,
            return_iterations=(5,),
        )

        llr = output["llr"]

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
                hard_errors(
                    llr,
                    bits,
                )
                / bits.numel()
            )

        running_loss += loss.detach().item()
        running_ber += batch_ber

        if step % 50 == 0:
            print(
                f"step={step:4d} | "
                f"BCE={running_loss / 50.0:.5f} | "
                f"BER={running_ber / 50.0:.5f} | "
                f"grad={float(grad_norm):.3f} | "
                f"corr={output['correction_rms'].mean().item():.3f} | "
                f"valid={output['valid_update_fraction'].mean().item():.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

            running_loss = 0.0
            running_ber = 0.0

        if step % args.val_every == 0:
            metrics = validate(
                model,
                classical_ep,
                validation,
                device,
            )

            print(
                f"VAL step={step:4d} | "
                f"EP5={metrics['ber_ep5']:.6e} | "
                f"SelfOnly={metrics['ber_gt']:.6e} | "
                f"gain={100.0 * metrics['relative_gain_vs_ep5']:+.3f}% | "
                f"recover={100.0 * metrics['oracle_gap_recovered']:+.2f}%"
            )

            history.append({
                "step": step,
                "edge_mode": "no_edge",
                "message_mode": "self_only",
                "seed": args.train_seed,
                "lr": optimizer.param_groups[0]["lr"],
                **metrics,
            })

            with open(
                history_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    history,
                    f,
                    indent=2,
                )

            if metrics["ber_gt"] < best_ber:
                best_ber = metrics["ber_gt"]
                best_step = step

                torch.save({
                    "model_state": model.state_dict(),
                    "step": step,
                    "best_ber": best_ber,
                    "edge_mode": "no_edge",
                    "message_mode": "self_only",
                    "seed": args.train_seed,
                    "metrics": metrics,
                }, checkpoint_path)

                print(
                    f"  NEW BEST | "
                    f"BER={best_ber:.8e} "
                    f"@ step={best_step}"
                )

    print()
    print("=" * 130)
    print("SELF-ONLY TRAINING FINISHED")
    print("=" * 130)
    print(f"EP5 BER      : {initial['ber_ep5']:.8e}")
    print(f"Best BER     : {best_ber:.8e}")
    print(f"Best step    : {best_step}")
    print(
        f"Relative gain: "
        f"{100.0 * (initial['ber_ep5'] - best_ber) / initial['ber_ep5']:+.3f}%"
    )
    print(f"Checkpoint   : {checkpoint_path}")
    print("=" * 130)

    del model
    torch.cuda.empty_cache()

    return checkpoint_path


@torch.no_grad()
def evaluate(base_cfg, self_checkpoint, args):
    cfg = copy.deepcopy(
        base_cfg
    )

    cfg["link"]["rx_snr_db"] = float(
        args.eval_snr
    )

    device = cfg["general"]["device"]

    sionna_config.seed = int(
        args.eval_seed
    )

    torch.manual_seed(
        int(args.eval_seed)
    )

    dataset = UplinkUMADataset(
        cfg
    )

    ep = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    specs = {
        "full": {
            "checkpoint": args.full_checkpoint,
            "edge_mode": "full",
            "message_mode": "cross_user",
        },
        "no_edge": {
            "checkpoint": args.no_edge_checkpoint,
            "edge_mode": "no_edge",
            "message_mode": "cross_user",
        },
        "self_only": {
            "checkpoint": str(self_checkpoint),
            "edge_mode": "no_edge",
            "message_mode": "self_only",
        },
    }

    models = {}

    print()
    print("=" * 130)
    print("INDEPENDENT MESSAGE-PASSING ABLATION")
    print("=" * 130)
    print(f"SNR             : {args.eval_snr:.1f} dB")
    print(f"Channels        : {args.eval_channels}")
    print(f"RE / channel    : {args.eval_re_per_channel}")
    print(
        f"Total bits      : "
        f"{args.eval_channels * args.eval_re_per_channel * 16 * 4}"
    )
    print(f"Evaluation seed : {args.eval_seed}")
    print("=" * 130)

    for name, spec in specs.items():
        model = build_model(
            cfg,
            edge_mode=spec["edge_mode"],
            message_mode=spec["message_mode"],
        )

        checkpoint = torch.load(
            spec["checkpoint"],
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        model.eval()

        models[name] = {
            "model": model,
            "step": checkpoint.get(
                "step",
                -1,
            ),
        }

        print(
            f"{name:9s} checkpoint step: "
            f"{checkpoint.get('step', -1)}"
        )

    total_errors = {
        "ep5": 0,
        "full": 0,
        "no_edge": 0,
        "self_only": 0,
    }

    per_channel = {
        key: []
        for key in total_errors
    }

    total_bits = 0

    for channel_idx in range(
        1,
        args.eval_channels + 1,
    ):
        batch = dataset.sample(1)

        data_idx = list(
            dataset.channel.resource_grid.data_symbols
        )

        y_full = batch["y"][:, data_idx]
        h_ls_full = batch["h_hat_ls"][:, data_idx]
        bits_full = batch["bits"].reshape(
            -1,
            16,
            4,
        ).float()

        num_re = bits_full.shape[0]

        count = min(
            args.eval_re_per_channel,
            num_re,
        )

        idx = torch.randperm(
            num_re,
            device=device,
        )[:count]

        y = y_full.reshape(
            -1,
            256,
        )[idx]

        h_ls = h_ls_full.reshape(
            -1,
            256,
            16,
        )[idx]

        bits = bits_full[idx]

        y_white, h_white = whiten(
            y,
            h_ls,
            batch["ruu_hat"],
        )

        z, gram = sufficient_statistics(
            y_white,
            h_white,
        )

        llr_ep5 = ep(
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llrs = {
            "ep5": llr_ep5,
        }

        for name, item in models.items():
            llrs[name] = item["model"](
                z,
                gram,
                return_iterations=(5,),
            )["llr"].float()

        num_bits = bits.numel()
        total_bits += num_bits

        for name, llr in llrs.items():
            error_count = hard_errors(
                llr,
                bits,
            )

            total_errors[name] += error_count

            per_channel[name].append(
                error_count / num_bits
            )

        if (
            channel_idx == 1
            or channel_idx % 50 == 0
            or channel_idx == args.eval_channels
        ):
            current = {
                key: value / total_bits
                for key, value in total_errors.items()
            }

            print(
                f"{channel_idx:4d}/{args.eval_channels} | "
                f"EP5={current['ep5']:.6e} | "
                f"Full={current['full']:.6e} | "
                f"NoEdge={current['no_edge']:.6e} | "
                f"SelfOnly={current['self_only']:.6e}"
            )

    ber = {
        key: value / total_bits
        for key, value in total_errors.items()
    }

    bootstrap_results = {
        "full_vs_ep5": bootstrap(
            per_channel["ep5"],
            per_channel["full"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 1,
        ),
        "no_edge_vs_ep5": bootstrap(
            per_channel["ep5"],
            per_channel["no_edge"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 2,
        ),
        "self_only_vs_ep5": bootstrap(
            per_channel["ep5"],
            per_channel["self_only"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 3,
        ),
        "full_vs_self_only": bootstrap(
            per_channel["self_only"],
            per_channel["full"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 4,
        ),
        "no_edge_vs_self_only": bootstrap(
            per_channel["self_only"],
            per_channel["no_edge"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 5,
        ),
    }

    print()
    print("=" * 130)
    print("FINAL MESSAGE-PASSING ABLATION")
    print("=" * 130)

    print(
        f"EP5      : {ber['ep5']:.8e}"
    )

    for name in [
        "full",
        "no_edge",
        "self_only",
    ]:
        gain = (
            ber["ep5"] - ber[name]
        ) / ber["ep5"]

        print(
            f"{name:9s}: "
            f"{ber[name]:.8e} | "
            f"gain vs EP5={100.0 * gain:+.3f}%"
        )

    print()
    print("CROSS-USER MESSAGE CONTRIBUTION")

    for key in [
        "full_vs_self_only",
        "no_edge_vs_self_only",
    ]:
        item = bootstrap_results[key]

        print(
            f"{key:22s}: "
            f"ΔBER={item['delta']:+.6e} "
            f"95%CI=[{item['low']:+.6e}, {item['high']:+.6e}]"
        )

    print()
    print("GAIN VS EP5")

    for key in [
        "full_vs_ep5",
        "no_edge_vs_ep5",
        "self_only_vs_ep5",
    ]:
        item = bootstrap_results[key]

        print(
            f"{key:22s}: "
            f"ΔBER={item['delta']:+.6e} "
            f"95%CI=[{item['low']:+.6e}, {item['high']:+.6e}]"
        )

    result = {
        "snr_db": args.eval_snr,
        "channels": args.eval_channels,
        "re_per_channel": args.eval_re_per_channel,
        "total_bits": total_bits,
        "checkpoints": {
            name: {
                "step": item["step"],
                "path": specs[name]["checkpoint"],
            }
            for name, item in models.items()
        },
        "ber": ber,
        "relative_gain_vs_ep5": {
            name: (
                ber["ep5"] - ber[name]
            ) / ber["ep5"]
            for name in [
                "full",
                "no_edge",
                "self_only",
            ]
        },
        "bootstrap": bootstrap_results,
    }

    result_path = Path(
        args.result
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved: {result_path}"
    )
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--re-per-step",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--val-channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--val-re-per-channel",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--full-checkpoint",
        default="ckp/gt_ep_edge_ablation/full_seed42/best.pth",
    )

    parser.add_argument(
        "--no-edge-checkpoint",
        default="ckp/gt_ep_edge_ablation/no_edge_seed42/best.pth",
    )

    parser.add_argument(
        "--self-output",
        default="ckp/gt_ep_message_ablation/self_only_seed42",
    )

    parser.add_argument(
        "--eval-snr",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--eval-channels",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--eval-re-per-channel",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=20260912,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--result",
        default="results/gt_ep_message_ablation/message_ablation_seed42_5db.json",
    )

    args = parser.parse_args()

    train_cfg = load_yaml(
        args.config
    )

    base_cfg = load_yaml(
        train_cfg["system_config"]
    )

    base_cfg["link"]["rx_snr_db"] = 5.0

    device = base_cfg[
        "general"
    ]["device"]

    sionna_config.device = device
    sionna_config.precision = base_cfg[
        "general"
    ]["precision"]

    torch.set_float32_matmul_precision(
        "high"
    )

    self_checkpoint = train_self_only(
        base_cfg,
        args,
    )

    evaluate(
        base_cfg,
        self_checkpoint,
        args,
    )


if __name__ == "__main__":
    main()