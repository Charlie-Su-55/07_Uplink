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
from detectors.classical.lmmse import LMMSESoftDetector
from models.graph.gt_ep_detector import GraphTransformerEPDetector
from training.train_gt_ep_detector import sample_re, build_validation_set, validate, check_exact_ep_anchor, hard_errors


MODES = ["full", "no_edge", "shuffled"]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_gt_ep(cfg, edge_mode):
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
    ).to(cfg["general"]["device"])


def train_one(base_cfg, edge_mode, seed, args):
    cfg = copy.deepcopy(base_cfg)
    cfg["general"]["seed"] = int(seed)
    cfg["link"]["rx_snr_db"] = 5.0

    device = cfg["general"]["device"]

    run_dir = Path(args.output_root) / f"{edge_mode}_seed{seed}"
    checkpoint_path = run_dir / "best.pth"
    history_path = run_dir / "history.json"
    done_path = run_dir / "done.json"

    run_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and done_path.exists() and not args.force_retrain:
        print(f"[SKIP] {edge_mode} seed={seed}: completed checkpoint exists.")
        return checkpoint_path

    print()
    print("=" * 130)
    print(f"TRAIN | mode={edge_mode} | seed={seed}")
    print("=" * 130)

    sionna_config.seed = int(seed)
    torch.manual_seed(int(seed))

    dataset = UplinkUMADataset(cfg)

    model = build_gt_ep(cfg, edge_mode)
    classical_ep = ExpectationPropagationDetector(cfg, num_iterations=5, damping=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    sionna_config.seed = int(seed) + 2000
    torch.manual_seed(int(seed) + 2000)

    validation = build_validation_set(dataset, args.val_channels, args.val_re_per_channel)

    check_exact_ep_anchor(model, classical_ep, validation, device)

    sionna_config.seed = int(seed)
    torch.manual_seed(int(seed))

    initial = validate(model, classical_ep, validation, device)

    print(
        f"Initial | EP5={initial['ber_ep5']:.6e} | "
        f"GT={initial['ber_gt']:.6e} | "
        f"TrueH={initial['ber_trueH_ep5']:.6e}"
    )

    best_ber = initial["ber_gt"]
    best_step = 0

    torch.save({
        "model_state": model.state_dict(),
        "step": 0,
        "best_ber": best_ber,
        "edge_mode": edge_mode,
        "seed": seed,
        "metrics": initial,
    }, checkpoint_path)

    history = [{
        "step": 0,
        "edge_mode": edge_mode,
        "seed": seed,
        "lr": args.lr,
        **initial,
    }]

    running_loss = 0.0
    running_ber = 0.0

    for step in range(1, args.steps + 1):
        model.train()

        sample = sample_re(dataset, args.re_per_step)

        z = sample["z_ls"]
        gram = sample["gram_ls"]
        bits = sample["bits"]

        optimizer.zero_grad(set_to_none=True)

        output = model(z, gram, return_iterations=(5,))
        llr = output["llr"]

        loss = F.binary_cross_entropy_with_logits(llr, bits)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: mode={edge_mode}, seed={seed}, step={step}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            batch_ber = hard_errors(llr, bits) / bits.numel()

        running_loss += loss.detach().item()
        running_ber += batch_ber

        if step % 50 == 0:
            print(
                f"{edge_mode:9s} seed={seed} step={step:4d} | "
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
            metrics = validate(model, classical_ep, validation, device)

            print(
                f"VAL {edge_mode:9s} seed={seed} step={step:4d} | "
                f"EP5={metrics['ber_ep5']:.6e} | "
                f"GT={metrics['ber_gt']:.6e} | "
                f"gain={100.0 * metrics['relative_gain_vs_ep5']:+.3f}% | "
                f"recover={100.0 * metrics['oracle_gap_recovered']:+.2f}%"
            )

            history.append({
                "step": step,
                "edge_mode": edge_mode,
                "seed": seed,
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
                    "edge_mode": edge_mode,
                    "seed": seed,
                    "metrics": metrics,
                }, checkpoint_path)

                print(f"  NEW BEST | {edge_mode} seed={seed} | BER={best_ber:.8e} | step={best_step}")

    with open(done_path, "w", encoding="utf-8") as f:
        json.dump({
            "edge_mode": edge_mode,
            "seed": seed,
            "best_step": best_step,
            "best_ber": best_ber,
            "checkpoint": str(checkpoint_path),
        }, f, indent=2)

    print(f"DONE | {edge_mode} seed={seed} | best={best_ber:.8e} @ step={best_step}")

    del model
    torch.cuda.empty_cache()

    return checkpoint_path


def bootstrap(reference, method, n_boot=5000, seed=999):
    reference = torch.tensor(reference, dtype=torch.float64)
    method = torch.tensor(method, dtype=torch.float64)

    diff = reference - method

    generator = torch.Generator().manual_seed(seed)

    n = diff.numel()
    samples = []

    for start in range(0, n_boot, 1000):
        count = min(1000, n_boot - start)
        idx = torch.randint(0, n, (count, n), generator=generator)
        samples.append(diff[idx].mean(dim=1))

    samples = torch.cat(samples)

    return {
        "delta": diff.mean().item(),
        "low": torch.quantile(samples, 0.025).item(),
        "high": torch.quantile(samples, 0.975).item(),
    }


@torch.no_grad()
def evaluate_all(base_cfg, checkpoints, snr_db, args):
    cfg = copy.deepcopy(base_cfg)
    cfg["link"]["rx_snr_db"] = float(snr_db)

    device = cfg["general"]["device"]

    sionna_config.seed = args.eval_seed
    torch.manual_seed(args.eval_seed)

    dataset = UplinkUMADataset(cfg)

    ep = ExpectationPropagationDetector(cfg, num_iterations=5, damping=0.5)
    lmmse = LMMSESoftDetector(cfg)

    models = {}

    for key, spec in checkpoints.items():
        model = build_gt_ep(cfg, spec["edge_mode"])
        ckpt = torch.load(spec["path"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        models[key] = {
            "model": model,
            "step": ckpt.get("step", -1),
            "edge_mode": spec["edge_mode"],
            "seed": spec["seed"],
        }

    total_errors = {"lmmse": 0, "ep5": 0, "trueH_ep5": 0}

    for key in models:
        total_errors[key] = 0

    total_bits = 0

    per_channel = {
        "ep5": [],
        "lmmse": [],
        "trueH_ep5": [],
    }

    for key in models:
        per_channel[key] = []

    for channel_idx in range(1, args.eval_channels + 1):
        batch = dataset.sample(1)

        data_idx = list(dataset.channel.resource_grid.data_symbols)

        y_full = batch["y"][:, data_idx]
        h_ls_full = batch["h_hat_ls"][:, data_idx]
        h_true_full = batch["h_true"][:, data_idx]
        bits_full = batch["bits"].reshape(-1, 16, 4).float()

        num_re = bits_full.shape[0]
        count = min(args.eval_re_per_channel, num_re)

        idx = torch.randperm(num_re, device=device)[:count]

        y = y_full.reshape(-1, 256)[idx]
        h_ls = h_ls_full.reshape(-1, 256, 16)[idx]
        h_true = h_true_full.reshape(-1, 256, 16)[idx]
        bits = bits_full[idx]

        llr_lmmse = lmmse(y_full, h_ls_full, batch["ruu_hat"])["llr"].reshape(-1, 16, 4)[idx].float()

        y_white, h_ls_white = whiten(y, h_ls, batch["ruu_hat"])
        _, h_true_white = whiten(y, h_true, batch["ruu_hat"])

        z, gram = sufficient_statistics(y_white, h_ls_white)
        z_true, gram_true = sufficient_statistics(y_white, h_true_white)

        llr_ep5 = ep(z, gram, return_iterations=(5,))["llr"].float()
        llr_true = ep(z_true, gram_true, return_iterations=(5,))["llr"].float()

        nbits = bits.numel()
        total_bits += nbits

        err_lmmse = hard_errors(llr_lmmse, bits)
        err_ep5 = hard_errors(llr_ep5, bits)
        err_true = hard_errors(llr_true, bits)

        total_errors["lmmse"] += err_lmmse
        total_errors["ep5"] += err_ep5
        total_errors["trueH_ep5"] += err_true

        per_channel["lmmse"].append(err_lmmse / nbits)
        per_channel["ep5"].append(err_ep5 / nbits)
        per_channel["trueH_ep5"].append(err_true / nbits)

        for key, item in models.items():
            llr = item["model"](z, gram, return_iterations=(5,))["llr"].float()
            err = hard_errors(llr, bits)

            total_errors[key] += err
            per_channel[key].append(err / nbits)

        if channel_idx == 1 or channel_idx % 50 == 0 or channel_idx == args.eval_channels:
            current = {key: value / total_bits for key, value in total_errors.items()}

            full_values = [
                current[key]
                for key in models
                if models[key]["edge_mode"] == "full"
            ]

            no_edge_values = [
                current[key]
                for key in models
                if models[key]["edge_mode"] == "no_edge"
            ]

            shuffled_values = [
                current[key]
                for key in models
                if models[key]["edge_mode"] == "shuffled"
            ]

            print(
                f"EVAL SNR={snr_db:>4.1f} | {channel_idx:4d}/{args.eval_channels} | "
                f"EP5={current['ep5']:.6e} | "
                f"Full={sum(full_values) / len(full_values):.6e} | "
                f"NoEdge={sum(no_edge_values) / len(no_edge_values):.6e} | "
                f"Shuffle={sum(shuffled_values) / len(shuffled_values):.6e}"
            )

    ber = {
        key: value / total_bits
        for key, value in total_errors.items()
    }

    model_results = {}

    for key, item in models.items():
        ci = bootstrap(
            per_channel["ep5"],
            per_channel[key],
            n_boot=args.bootstrap,
            seed=args.eval_seed + item["seed"],
        )

        model_results[key] = {
            "edge_mode": item["edge_mode"],
            "seed": item["seed"],
            "step": item["step"],
            "ber": ber[key],
            "relative_gain_vs_ep5": (ber["ep5"] - ber[key]) / ber["ep5"],
            "bootstrap_vs_ep5": ci,
        }

    aggregate = {}

    for mode in MODES:
        values = [
            result["ber"]
            for result in model_results.values()
            if result["edge_mode"] == mode
        ]

        gains = [
            result["relative_gain_vs_ep5"]
            for result in model_results.values()
            if result["edge_mode"] == mode
        ]

        tensor_values = torch.tensor(values, dtype=torch.float64)
        tensor_gains = torch.tensor(gains, dtype=torch.float64)

        aggregate[mode] = {
            "mean_ber": tensor_values.mean().item(),
            "std_ber": tensor_values.std(unbiased=True).item(),
            "mean_gain_vs_ep5": tensor_gains.mean().item(),
            "std_gain_vs_ep5": tensor_gains.std(unbiased=True).item(),
        }

    result = {
        "snr_db": float(snr_db),
        "channels": args.eval_channels,
        "re_per_channel": args.eval_re_per_channel,
        "total_bits": total_bits,
        "baseline": {
            "lmmse": ber["lmmse"],
            "ep5": ber["ep5"],
            "trueH_ep5": ber["trueH_ep5"],
        },
        "models": model_results,
        "aggregate": aggregate,
    }

    print()
    print("=" * 130)
    print(f"SNR {snr_db:.1f} dB | EDGE ABLATION SUMMARY")
    print("=" * 130)
    print(f"LMMSE      : {ber['lmmse']:.8e}")
    print(f"EP5        : {ber['ep5']:.8e}")
    print(f"TrueH EP5  : {ber['trueH_ep5']:.8e}")
    print()

    for mode in MODES:
        a = aggregate[mode]

        print(
            f"{mode:9s} | "
            f"BER={a['mean_ber']:.8e} ± {a['std_ber']:.2e} | "
            f"gain vs EP5={100.0 * a['mean_gain_vs_ep5']:+.3f}% "
            f"± {100.0 * a['std_gain_vs_ep5']:.3f}%"
        )

    print("=" * 130)

    for item in models.values():
        del item["model"]

    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--re-per-step", type=int, default=128)
    parser.add_argument("--val-channels", type=int, default=32)
    parser.add_argument("--val-re-per-channel", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=100)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-snrs", type=float, nargs="+", default=[0, 5, 10])
    parser.add_argument("--eval-channels", type=int, default=500)
    parser.add_argument("--eval-re-per-channel", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=20260911)
    parser.add_argument("--bootstrap", type=int, default=5000)

    parser.add_argument("--output-root", default="ckp/gt_ep_edge_ablation")
    parser.add_argument("--result-root", default="results/gt_ep_edge_ablation")
    parser.add_argument("--force-retrain", action="store_true")

    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    base_cfg = load_yaml(train_cfg["system_config"])

    base_cfg["link"]["rx_snr_db"] = 5.0

    device = base_cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = base_cfg["general"]["precision"]

    torch.set_float32_matmul_precision("high")

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.result_root).mkdir(parents=True, exist_ok=True)

    print("=" * 130)
    print("GT-EP OVERNIGHT EDGE ABLATION")
    print("=" * 130)
    print(f"Modes           : {MODES}")
    print(f"Training seeds  : {args.seeds}")
    print(f"Steps / run     : {args.steps}")
    print(f"Total train runs: {len(MODES) * len(args.seeds)}")
    print(f"Evaluation SNRs : {args.eval_snrs}")
    print(f"Eval channels   : {args.eval_channels}")
    print(f"RE/channel      : {args.eval_re_per_channel}")
    print("=" * 130)

    checkpoints = {}

    for mode in MODES:
        for seed in args.seeds:
            path = train_one(
                base_cfg=base_cfg,
                edge_mode=mode,
                seed=seed,
                args=args,
            )

            key = f"{mode}_seed{seed}"

            checkpoints[key] = {
                "path": str(path),
                "edge_mode": mode,
                "seed": seed,
            }

    all_results = []

    for snr_db in args.eval_snrs:
        result = evaluate_all(
            base_cfg=base_cfg,
            checkpoints=checkpoints,
            snr_db=snr_db,
            args=args,
        )

        all_results.append(result)

        output_path = Path(args.result_root) / f"edge_ablation_{snr_db:g}db.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    summary_path = Path(args.result_root) / "edge_ablation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "modes": MODES,
            "seeds": args.seeds,
            "steps": args.steps,
            "eval_snrs": args.eval_snrs,
            "results": all_results,
        }, f, indent=2)

    print()
    print("#" * 130)
    print("OVERNIGHT EXPERIMENT COMPLETE")
    print("#" * 130)

    for result in all_results:
        print(f"SNR={result['snr_db']:.1f} dB | EP5={result['baseline']['ep5']:.6e}")

        for mode in MODES:
            a = result["aggregate"][mode]

            print(
                f"  {mode:9s}: "
                f"BER={a['mean_ber']:.6e} | "
                f"gain={100.0 * a['mean_gain_vs_ep5']:+.3f}% "
                f"± {100.0 * a['std_gain_vs_ep5']:.3f}%"
            )

    print()
    print(f"Saved summary: {summary_path}")
    print("#" * 130)


if __name__ == "__main__":
    main()