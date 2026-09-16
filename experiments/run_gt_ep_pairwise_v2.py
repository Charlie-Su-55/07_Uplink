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
from models.graph.gt_ep_pairwise import PairwiseGraphTransformerEPDetector
from training.train_gt_ep_detector import sample_re, build_validation_set, validate, check_exact_ep_anchor, hard_errors


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_v2(cfg, pairwise_mode="full"):
    return PairwiseGraphTransformerEPDetector(
        cfg=cfg,
        num_users=16,
        num_iterations=5,
        damping=0.5,
        d_model=128,
        num_heads=8,
        num_layers=4,
        edge_dim=48,
        ffn_dim=256,
        dropout=0.05,
        max_logit_correction=4.0,
        pairwise_mode=pairwise_mode,
    ).to(cfg["general"]["device"])


def build_v1(cfg, edge_mode, message_mode):
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

    y_white = torch.linalg.solve_triangular(chol, y.unsqueeze(-1), upper=False).squeeze(-1)
    h_white = torch.linalg.solve_triangular(chol, h, upper=False)

    return y_white, h_white


def sufficient_statistics(y_white, h_white):
    z = torch.einsum("bmk,bm->bk", h_white.conj(), y_white)
    gram = torch.einsum("bmk,bml->bkl", h_white.conj(), h_white)

    return z, gram


def bootstrap(reference, method, n_boot=10000, seed=12345):
    reference = torch.tensor(reference, dtype=torch.float64)
    method = torch.tensor(method, dtype=torch.float64)

    diff = reference - method

    generator = torch.Generator().manual_seed(int(seed))

    n = diff.numel()
    samples = []

    remaining = int(n_boot)

    while remaining > 0:
        count = min(1000, remaining)

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

    samples = torch.cat(samples)

    return {
        "delta": diff.mean().item(),
        "low": torch.quantile(samples, 0.025).item(),
        "high": torch.quantile(samples, 0.975).item(),
    }


def train_v2(base_cfg, args):
    cfg = copy.deepcopy(base_cfg)

    cfg["general"]["seed"] = int(args.train_seed)
    cfg["link"]["rx_snr_db"] = 5.0

    device = cfg["general"]["device"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best.pth"
    history_path = output_dir / "history.json"

    sionna_config.seed = int(args.train_seed)
    torch.manual_seed(int(args.train_seed))

    dataset = UplinkUMADataset(cfg)

    model = build_v2(
        cfg,
        pairwise_mode="full",
    )

    ep = ExpectationPropagationDetector(
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

    print("=" * 140)
    print("PAIRWISE SYMBOL-ENERGY GT-EP v2")
    print("=" * 140)
    print("System             : 256 Rx / 16 users / 16-QAM")
    print("Training SNR       : 5 dB")
    print("Pairwise energy    : -2 Re{a* G_ij b}")
    print("Pairwise belief    : logsumexp_b(log q_j(b) + E_ij(a,b))")
    print("Graph insertion    : inside every EP iteration")
    print("Pairwise mode      : full")
    print(f"Graph layers/pass  : 4")
    print(f"EP iterations      : 5")
    print(f"Parameters         : {num_params:,}")
    print(f"Steps              : {args.steps}")
    print(f"RE / step          : {args.re_per_step}")
    print("=" * 140)

    sionna_config.seed = int(args.train_seed) + 2000
    torch.manual_seed(int(args.train_seed) + 2000)

    validation = build_validation_set(
        dataset,
        args.val_channels,
        args.val_re_per_channel,
    )

    check_exact_ep_anchor(
        model,
        ep,
        validation,
        device,
    )

    sionna_config.seed = int(args.train_seed)
    torch.manual_seed(int(args.train_seed))

    initial = validate(
        model,
        ep,
        validation,
        device,
    )

    print()
    print("INITIAL VALIDATION")
    print("-" * 140)
    print(f"EP5          : {initial['ber_ep5']:.8e}")
    print(f"Pairwise-v2  : {initial['ber_gt']:.8e}")
    print(f"TrueH EP5    : {initial['ber_trueH_ep5']:.8e}")
    print("-" * 140)

    best_ber = initial["ber_gt"]
    best_step = 0

    torch.save({
        "model_state": model.state_dict(),
        "step": 0,
        "best_ber": best_ber,
        "pairwise_mode": "full",
        "seed": args.train_seed,
        "metrics": initial,
    }, checkpoint_path)

    history = [{
        "step": 0,
        "seed": args.train_seed,
        "lr": args.lr,
        **initial,
    }]

    running_loss = 0.0
    running_ber = 0.0

    for step in range(1, args.steps + 1):
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
                hard_errors(llr, bits)
                / bits.numel()
            )

        running_loss += loss.detach().item()
        running_ber += batch_ber

        if step % 25 == 0:
            print(
                f"step={step:4d} | "
                f"BCE={running_loss / 25.0:.5f} | "
                f"BER={running_ber / 25.0:.5f} | "
                f"grad={float(grad_norm):.3f} | "
                f"corr={output['correction_rms'].mean().item():.3f} | "
                f"pair={output['pairwise_message_rms'].mean().item():.3f} | "
                f"valid={output['valid_update_fraction'].mean().item():.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

            running_loss = 0.0
            running_ber = 0.0

        if step % args.val_every == 0:
            metrics = validate(
                model,
                ep,
                validation,
                device,
            )

            print(
                f"VAL step={step:4d} | "
                f"EP5={metrics['ber_ep5']:.6e} | "
                f"Pairwise-v2={metrics['ber_gt']:.6e} | "
                f"gain={100.0 * metrics['relative_gain_vs_ep5']:+.3f}% | "
                f"recover={100.0 * metrics['oracle_gap_recovered']:+.2f}%"
            )

            history.append({
                "step": step,
                "seed": args.train_seed,
                "lr": optimizer.param_groups[0]["lr"],
                **metrics,
            })

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if metrics["ber_gt"] < best_ber:
                best_ber = metrics["ber_gt"]
                best_step = step

                torch.save({
                    "model_state": model.state_dict(),
                    "step": step,
                    "best_ber": best_ber,
                    "pairwise_mode": "full",
                    "seed": args.train_seed,
                    "metrics": metrics,
                }, checkpoint_path)

                print(
                    f"  NEW BEST | BER={best_ber:.8e} @ step={best_step}"
                )

    print()
    print("=" * 140)
    print("TRAINING FINISHED")
    print("=" * 140)
    print(f"Initial EP5 BER : {initial['ber_ep5']:.8e}")
    print(f"Best v2 BER     : {best_ber:.8e}")
    print(f"Best step       : {best_step}")
    print(
        f"Relative gain   : "
        f"{100.0 * (initial['ber_ep5'] - best_ber) / initial['ber_ep5']:+.3f}%"
    )
    print(f"Checkpoint      : {checkpoint_path}")
    print("=" * 140)

    del model
    torch.cuda.empty_cache()

    return checkpoint_path


@torch.no_grad()
def evaluate(base_cfg, v2_checkpoint, args):
    cfg = copy.deepcopy(base_cfg)

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

    dataset = UplinkUMADataset(cfg)

    ep = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    full_v1 = build_v1(
        cfg,
        edge_mode="full",
        message_mode="cross_user",
    )

    full_ckpt = torch.load(
        args.full_v1_checkpoint,
        map_location=device,
        weights_only=False,
    )

    full_v1.load_state_dict(
        full_ckpt["model_state"],
        strict=True,
    )

    full_v1.eval()

    self_only = build_v1(
        cfg,
        edge_mode="no_edge",
        message_mode="self_only",
    )

    self_ckpt = torch.load(
        args.self_only_checkpoint,
        map_location=device,
        weights_only=False,
    )

    self_only.load_state_dict(
        self_ckpt["model_state"],
        strict=True,
    )

    self_only.eval()

    pairwise_v2 = build_v2(
        cfg,
        pairwise_mode="full",
    )

    v2_ckpt = torch.load(
        v2_checkpoint,
        map_location=device,
        weights_only=False,
    )

    pairwise_v2.load_state_dict(
        v2_ckpt["model_state"],
        strict=True,
    )

    pairwise_v2.eval()

    print()
    print("=" * 140)
    print("INDEPENDENT GT-EP v2 EVALUATION")
    print("=" * 140)
    print(f"SNR              : {args.eval_snr:.1f} dB")
    print(f"Channels         : {args.eval_channels}")
    print(f"RE/channel       : {args.eval_re_per_channel}")
    print(f"Bits             : {args.eval_channels * args.eval_re_per_channel * 16 * 4}")
    print(f"Full-v1 step     : {full_ckpt.get('step', -1)}")
    print(f"SelfOnly step    : {self_ckpt.get('step', -1)}")
    print(f"Pairwise-v2 step : {v2_ckpt.get('step', -1)}")
    print("=" * 140)

    names = [
        "ep5",
        "self_only",
        "full_v1",
        "pairwise_v2",
        "trueH_ep5",
    ]

    total_errors = {
        name: 0
        for name in names
    }

    per_channel = {
        name: []
        for name in names
    }

    total_bits = 0

    for channel_idx in range(1, args.eval_channels + 1):
        batch = dataset.sample(1)

        data_idx = list(
            dataset.channel.resource_grid.data_symbols
        )

        y_full = batch["y"][:, data_idx]
        h_ls_full = batch["h_hat_ls"][:, data_idx]
        h_true_full = batch["h_true"][:, data_idx]
        bits_full = batch["bits"].reshape(-1, 16, 4).float()

        num_re = bits_full.shape[0]

        count = min(
            args.eval_re_per_channel,
            num_re,
        )

        idx = torch.randperm(
            num_re,
            device=device,
        )[:count]

        y = y_full.reshape(-1, 256)[idx]
        h_ls = h_ls_full.reshape(-1, 256, 16)[idx]
        h_true = h_true_full.reshape(-1, 256, 16)[idx]
        bits = bits_full[idx]

        y_white, h_ls_white = whiten(
            y,
            h_ls,
            batch["ruu_hat"],
        )

        _, h_true_white = whiten(
            y,
            h_true,
            batch["ruu_hat"],
        )

        z, gram = sufficient_statistics(
            y_white,
            h_ls_white,
        )

        z_true, gram_true = sufficient_statistics(
            y_white,
            h_true_white,
        )

        llr_ep5 = ep(
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_self = self_only(
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_full = full_v1(
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_v2 = pairwise_v2(
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_true = ep(
            z_true,
            gram_true,
            return_iterations=(5,),
        )["llr"].float()

        outputs = {
            "ep5": llr_ep5,
            "self_only": llr_self,
            "full_v1": llr_full,
            "pairwise_v2": llr_v2,
            "trueH_ep5": llr_true,
        }

        nbits = bits.numel()
        total_bits += nbits

        for name, llr in outputs.items():
            err = hard_errors(
                llr,
                bits,
            )

            total_errors[name] += err
            per_channel[name].append(
                err / nbits
            )

        if (
            channel_idx == 1
            or channel_idx % 50 == 0
            or channel_idx == args.eval_channels
        ):
            current = {
                name: total_errors[name] / total_bits
                for name in names
            }

            print(
                f"{channel_idx:4d}/{args.eval_channels} | "
                f"EP5={current['ep5']:.6e} | "
                f"Self={current['self_only']:.6e} | "
                f"V1={current['full_v1']:.6e} | "
                f"V2={current['pairwise_v2']:.6e} | "
                f"TrueH={current['trueH_ep5']:.6e}"
            )

    ber = {
        name: total_errors[name] / total_bits
        for name in names
    }

    comparisons = {
        "v2_vs_ep5": bootstrap(
            per_channel["ep5"],
            per_channel["pairwise_v2"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 1,
        ),
        "v2_vs_self_only": bootstrap(
            per_channel["self_only"],
            per_channel["pairwise_v2"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 2,
        ),
        "v2_vs_full_v1": bootstrap(
            per_channel["full_v1"],
            per_channel["pairwise_v2"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 3,
        ),
        "full_v1_vs_self_only": bootstrap(
            per_channel["self_only"],
            per_channel["full_v1"],
            n_boot=args.bootstrap,
            seed=args.eval_seed + 4,
        ),
    }

    print()
    print("=" * 140)
    print("FINAL GT-EP v2 COMPARISON")
    print("=" * 140)
    print(f"EP5          : {ber['ep5']:.8e}")
    print(f"SelfOnly     : {ber['self_only']:.8e}")
    print(f"Full GT-EP v1: {ber['full_v1']:.8e}")
    print(f"Pairwise v2  : {ber['pairwise_v2']:.8e}")
    print(f"TrueH EP5    : {ber['trueH_ep5']:.8e}")
    print()

    print(
        f"SelfOnly vs EP5 : "
        f"{100.0 * (ber['ep5'] - ber['self_only']) / ber['ep5']:+.3f}%"
    )

    print(
        f"Full-v1 vs EP5  : "
        f"{100.0 * (ber['ep5'] - ber['full_v1']) / ber['ep5']:+.3f}%"
    )

    print(
        f"Pairwise-v2 vs EP5: "
        f"{100.0 * (ber['ep5'] - ber['pairwise_v2']) / ber['ep5']:+.3f}%"
    )

    print(
        f"Pairwise-v2 vs Full-v1: "
        f"{100.0 * (ber['full_v1'] - ber['pairwise_v2']) / ber['full_v1']:+.3f}%"
    )

    print()
    print("PAIRED CHANNEL BOOTSTRAP")

    for name, item in comparisons.items():
        print(
            f"{name:22s}: "
            f"ΔBER={item['delta']:+.6e} "
            f"95%CI=[{item['low']:+.6e}, {item['high']:+.6e}]"
        )

    result = {
        "snr_db": args.eval_snr,
        "channels": args.eval_channels,
        "re_per_channel": args.eval_re_per_channel,
        "total_bits": total_bits,
        "steps": {
            "self_only": self_ckpt.get("step", -1),
            "full_v1": full_ckpt.get("step", -1),
            "pairwise_v2": v2_ckpt.get("step", -1),
        },
        "ber": ber,
        "comparisons": comparisons,
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
    print(f"Saved: {result_path}")
    print("=" * 140)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")

    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--re-per-step", type=int, default=128)

    parser.add_argument("--val-channels", type=int, default=32)
    parser.add_argument("--val-re-per-channel", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=100)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--output-dir", default="ckp/gt_ep_pairwise_v2_seed42")

    parser.add_argument("--full-v1-checkpoint", default="ckp/gt_ep_edge_ablation/full_seed42/best.pth")
    parser.add_argument("--self-only-checkpoint", default="ckp/gt_ep_message_ablation/self_only_seed42/best.pth")

    parser.add_argument("--eval-snr", type=float, default=5.0)
    parser.add_argument("--eval-channels", type=int, default=500)
    parser.add_argument("--eval-re-per-channel", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=20260913)
    parser.add_argument("--bootstrap", type=int, default=10000)

    parser.add_argument("--result", default="results/gt_ep_pairwise_v2/pairwise_v2_seed42_5db.json")

    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    base_cfg = load_yaml(train_cfg["system_config"])

    base_cfg["link"]["rx_snr_db"] = 5.0

    device = base_cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = base_cfg["general"]["precision"]

    torch.set_float32_matmul_precision("high")

    checkpoint = train_v2(
        base_cfg,
        args,
    )

    evaluate(
        base_cfg,
        checkpoint,
        args,
    )


if __name__ == "__main__":
    main()