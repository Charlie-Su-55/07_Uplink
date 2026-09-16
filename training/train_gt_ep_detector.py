import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.gt_ep_detector import GraphTransformerEPDetector


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


def sample_re(dataset, re_per_channel):
    batch = dataset.sample(1)
    data_idx = list(dataset.channel.resource_grid.data_symbols)

    num_streams = int(batch["metadata"]["num_streams"])
    bits_per_symbol = int(batch["metadata"]["bits_per_symbol"])

    y_raw = batch["y"][:, data_idx]
    h_ls_raw = batch["h_hat_ls"][:, data_idx]
    h_lmmse_raw = batch["h_hat_lmmse"][:, data_idx]
    h_true_raw = batch["h_true"][:, data_idx]
    bits_raw = batch["bits"]

    num_rx = int(y_raw.shape[-1])

    y = y_raw.reshape(-1, num_rx)
    h_ls = h_ls_raw.reshape(-1, num_rx, num_streams)
    h_lmmse = h_lmmse_raw.reshape(-1, num_rx, num_streams)
    h_true = h_true_raw.reshape(-1, num_rx, num_streams)
    bits = bits_raw.reshape(-1, num_streams, bits_per_symbol).float()

    if not (
        y.shape[0]
        == h_ls.shape[0]
        == h_lmmse.shape[0]
        == h_true.shape[0]
        == bits.shape[0]
    ):
        raise RuntimeError(
            f"RE count mismatch: y={y.shape[0]}, "
            f"h_ls={h_ls.shape[0]}, "
            f"h_lmmse={h_lmmse.shape[0]}, "
            f"h_true={h_true.shape[0]}, "
            f"bits={bits.shape[0]}"
        )

    count = min(int(re_per_channel), y.shape[0])
    idx = torch.randperm(y.shape[0], device=y.device)[:count]

    y = y[idx]
    h_ls = h_ls[idx]
    h_lmmse = h_lmmse[idx]
    h_true = h_true[idx]
    bits = bits[idx]

    y_white, h_ls_white = whiten(y, h_ls, batch["ruu_hat"])
    _, h_lmmse_white = whiten(y, h_lmmse, batch["ruu_hat"])
    _, h_true_white = whiten(y, h_true, batch["ruu_hat"])

    z_ls, gram_ls = sufficient_statistics(y_white, h_ls_white)
    z_lmmse, gram_lmmse = sufficient_statistics(y_white, h_lmmse_white)
    z_true, gram_true = sufficient_statistics(y_white, h_true_white)

    return {
        "z_ls": z_ls,
        "gram_ls": gram_ls,
        "z_lmmse": z_lmmse,
        "gram_lmmse": gram_lmmse,
        "z_true": z_true,
        "gram_true": gram_true,
        "bits": bits,
    }


def build_validation_set(dataset, num_channels, re_per_channel):
    storage = {
        "z_ls": [],
        "gram_ls": [],
        "z_lmmse": [],
        "gram_lmmse": [],
        "z_true": [],
        "gram_true": [],
        "bits": [],
    }

    print(f"Building fixed validation set: {num_channels} channels × {re_per_channel} RE")

    with torch.no_grad():
        for i in range(num_channels):
            sample = sample_re(dataset, re_per_channel)

            for key in storage:
                storage[key].append(sample[key].cpu())

            if (i + 1) % 8 == 0 or i + 1 == num_channels:
                print(f"  validation channel {i + 1}/{num_channels}")

    return {key: torch.cat(value, dim=0) for key, value in storage.items()}


def hard_errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


@torch.no_grad()
def validate(model, classical_ep, validation, device, chunk_size=256):
    model.eval()

    total_bits = 0
    errors_ep5 = 0
    errors_gt = 0
    errors_true = 0

    correction_sum = torch.zeros(5, dtype=torch.float64)
    validity_sum = torch.zeros(5, dtype=torch.float64)
    num_chunks = 0

    total_re = validation["bits"].shape[0]

    for start in range(0, total_re, chunk_size):
        end = min(start + chunk_size, total_re)

        z_ls = validation["z_ls"][start:end].to(device)
        gram_ls = validation["gram_ls"][start:end].to(device)
        z_true = validation["z_true"][start:end].to(device)
        gram_true = validation["gram_true"][start:end].to(device)
        bits = validation["bits"][start:end].to(device)

        ep_out = classical_ep(z_ls, gram_ls, return_iterations=(5,))
        true_out = classical_ep(z_true, gram_true, return_iterations=(5,))
        gt_out = model(z_ls, gram_ls, return_iterations=(5,))

        llr_ep5 = ep_out["llr"].float()
        llr_true = true_out["llr"].float()
        llr_gt = gt_out["llr"].float()

        total_bits += bits.numel()

        errors_ep5 += hard_errors(llr_ep5, bits)
        errors_gt += hard_errors(llr_gt, bits)
        errors_true += hard_errors(llr_true, bits)

        correction_sum += gt_out["correction_rms"].double().cpu()
        validity_sum += gt_out["valid_update_fraction"].double().cpu()

        num_chunks += 1

    ber_ep5 = errors_ep5 / total_bits
    ber_gt = errors_gt / total_bits
    ber_true = errors_true / total_bits

    oracle_gap = ber_ep5 - ber_true

    return {
        "ber_ep5": ber_ep5,
        "ber_gt": ber_gt,
        "ber_trueH_ep5": ber_true,
        "relative_gain_vs_ep5": (ber_ep5 - ber_gt) / max(ber_ep5, 1e-12),
        "oracle_gap_recovered": 0.0 if oracle_gap <= 0 else (ber_ep5 - ber_gt) / oracle_gap,
        "correction_rms": (correction_sum / max(num_chunks, 1)).tolist(),
        "valid_update_fraction": (validity_sum / max(num_chunks, 1)).tolist(),
    }


@torch.no_grad()
def check_exact_ep_anchor(model, classical_ep, validation, device):
    model.eval()

    z = validation["z_ls"][:256].to(device)
    gram = validation["gram_ls"][:256].to(device)

    llr_ep = classical_ep(z, gram, return_iterations=(5,))["llr"]
    llr_gt = model(z, gram, return_iterations=(5,))["llr"]

    max_diff = (llr_ep - llr_gt).abs().max().item()

    print(f"Initial max |GT-EP - EP5| LLR difference : {max_diff:.6e}")

    if max_diff > 1e-5:
        raise RuntimeError("Zero-initialized GT-EP does not reproduce classical EP5.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--re-per-step", type=int, default=128)
    parser.add_argument("--val-channels", type=int, default=32)
    parser.add_argument("--val-re-per-channel", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", default="ckp/gt_ep_256rx_16ue_lsH_estR_5db")
    parser.add_argument("--history", default="results/raw/gt_ep_256rx_16ue_lsH_estR_5db_history.json")

    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    system_cfg = load_yaml(train_cfg["system_config"])
    tcfg = train_cfg.get("training", {})

    snr_db = float(tcfg.get("snr_db", 5.0))
    system_cfg["link"]["rx_snr_db"] = snr_db

    device = system_cfg["general"]["device"]
    precision = system_cfg["general"]["precision"]
    seed = int(system_cfg["general"]["seed"])

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    dataset = UplinkUMADataset(system_cfg)

    model = GraphTransformerEPDetector(
        cfg=system_cfg,
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
    ).to(device)

    classical_ep = ExpectationPropagationDetector(
        system_cfg,
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best.pth"

    num_params = sum(p.numel() for p in model.parameters())

    print("=" * 120)
    print("Graph Transformer Unfolded EP Soft Detector")
    print("=" * 120)
    print("System              : 256 Rx / 16 users / 16-QAM")
    print(f"SNR                 : {snr_db:.1f} dB")
    print("CSI input           : LS H")
    print("Covariance input    : estimated R")
    print("Detection state     : EP cavity 16-QAM symbol beliefs")
    print("Graph nodes         : 16 users")
    print("Graph edges         : normalized/directional Gram coupling")
    print("GT insertion        : inside every EP iteration")
    print("Training objective  : bit-level BCE on final LLR")
    print("Graph layers/pass   : 4")
    print("EP iterations       : 5")
    print("d_model / heads     : 128 / 8")
    print(f"Parameters          : {num_params:,}")
    print(f"Steps               : {args.steps}")
    print(f"RE / step           : {args.re_per_step}")
    print(f"Learning rate       : {args.lr:.2e}")
    print("=" * 120)

    sionna_config.seed = seed + 2000
    torch.manual_seed(seed + 2000)

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

    sionna_config.seed = seed
    torch.manual_seed(seed)

    initial_metrics = validate(
        model,
        classical_ep,
        validation,
        device,
    )

    print()
    print("INITIAL VALIDATION")
    print("-" * 120)
    print(f"EP5             : {initial_metrics['ber_ep5']:.8e}")
    print(f"GT-EP           : {initial_metrics['ber_gt']:.8e}")
    print(f"TrueH EP5       : {initial_metrics['ber_trueH_ep5']:.8e}")
    print(f"GT vs EP5 gain  : {100.0 * initial_metrics['relative_gain_vs_ep5']:+.3f}%")
    print(f"Correction RMS  : {initial_metrics['correction_rms']}")
    print("-" * 120)

    best_ber = initial_metrics["ber_gt"]
    best_step = 0

    history = [{
        "step": 0,
        "lr": args.lr,
        **initial_metrics,
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
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            batch_ber = hard_errors(llr, bits) / bits.numel()

        running_loss += loss.detach().item()
        running_ber += batch_ber

        if step % 25 == 0:
            print(
                f"step={step:5d} | "
                f"BCE={running_loss / 25.0:.5f} | "
                f"BER={running_ber / 25.0:.5f} | "
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

            corr_text = ",".join(
                f"{value:.2f}"
                for value in metrics["correction_rms"]
            )

            print(
                f"VAL step={step:5d} | "
                f"EP5={metrics['ber_ep5']:.6e} | "
                f"GT-EP={metrics['ber_gt']:.6e} | "
                f"TrueH={metrics['ber_trueH_ep5']:.6e} | "
                f"gain={100.0 * metrics['relative_gain_vs_ep5']:+.3f}% | "
                f"recover={100.0 * metrics['oracle_gap_recovered']:+.2f}% | "
                f"corr=[{corr_text}]"
            )

            history.append({
                "step": step,
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
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "best_ber": best_ber,
                    "metrics": metrics,
                    "args": vars(args),
                }, best_path)

                print(f"  NEW BEST: BER={best_ber:.8e} at step {best_step}")

    print()
    print("=" * 120)
    print("TRAINING FINISHED")
    print("=" * 120)
    print(f"Initial EP5 BER : {initial_metrics['ber_ep5']:.8e}")
    print(f"Best GT-EP BER  : {best_ber:.8e}")
    print(f"Best step       : {best_step}")
    print(f"Relative gain   : {100.0 * (initial_metrics['ber_ep5'] - best_ber) / initial_metrics['ber_ep5']:+.3f}%")
    print(f"Checkpoint      : {best_path}")
    print(f"History         : {history_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()