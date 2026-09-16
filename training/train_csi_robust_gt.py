import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.csi_robust_gt import CSIRobustGraphTransformer


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def whiten_selected(y, h_ls, h_true, ruu):
    count = y.shape[0]
    chol = torch.linalg.cholesky(ruu)[0]
    chol = chol[None].expand(count, -1, -1)

    y_white = torch.linalg.solve_triangular(chol, y.unsqueeze(-1), upper=False).squeeze(-1)
    h_ls_white = torch.linalg.solve_triangular(chol, h_ls, upper=False)
    h_true_white = torch.linalg.solve_triangular(chol, h_true, upper=False)

    return y_white, h_ls_white, h_true_white


def sample_selected_re(dataset, re_per_channel):
    batch = dataset.sample(1)
    data_idx = list(dataset.channel.resource_grid.data_symbols)

    y = batch["y"][:, data_idx].reshape(-1, 256)
    h_ls = batch["h_hat_ls"][:, data_idx].reshape(-1, 256, 16)
    h_true = batch["h_true"][:, data_idx].reshape(-1, 256, 16)
    bits = batch["bits"].reshape(-1, 16, 4).float()

    count = min(int(re_per_channel), y.shape[0])
    idx = torch.randperm(y.shape[0], device=y.device)[:count]

    y = y[idx]
    h_ls = h_ls[idx]
    h_true = h_true[idx]
    bits = bits[idx]

    y_white, h_ls_white, h_true_white = whiten_selected(y, h_ls, h_true, batch["ruu_hat"])

    return {
        "y_white": y_white,
        "h_ls_white": h_ls_white,
        "h_true_white": h_true_white,
        "bits": bits,
    }


def sufficient_statistics(y_white, h_white):
    z = torch.einsum("bmk,bm->bk", h_white.conj(), y_white)
    gram = torch.einsum("bmk,bml->bkl", h_white.conj(), h_white)
    return z, gram


def channel_nmse_loss(pred, target):
    err = (pred - target).abs().square().sum(dim=1)
    power = target.abs().square().sum(dim=1).clamp_min(1e-8)
    return (err / power).mean()


def vector_nmse_loss(pred, target):
    err = (pred - target).abs().square().sum(dim=-1)
    power = target.abs().square().sum(dim=-1).clamp_min(1e-8)
    return (err / power).mean()


def normalized_gram(gram):
    diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
    denom = torch.sqrt(diag[:, :, None] * diag[:, None, :]).clamp_min(1e-8)
    rho = gram / denom
    return rho, diag


def graph_geometry_loss(pred_gram, target_gram):
    pred_rho, pred_diag = normalized_gram(pred_gram)
    target_rho, target_diag = normalized_gram(target_gram)

    k = pred_gram.shape[-1]
    mask = torch.ones(k, k, dtype=pred_gram.real.dtype, device=pred_gram.device)
    mask.fill_diagonal_(0.0)

    rho_error = (pred_rho - target_rho).abs().square() * mask[None]
    rho_loss = rho_error.sum() / (pred_gram.shape[0] * mask.sum())

    diag_loss = (torch.log1p(pred_diag) - torch.log1p(target_diag)).square().mean()

    return rho_loss, diag_loss


def training_loss(output, y_white, h_true_white):
    h_refined = output["h_refined"]
    z_refined = output["z_refined"]
    gram_refined = output["gram_refined"]

    z_true, gram_true = sufficient_statistics(y_white, h_true_white)

    loss_h = channel_nmse_loss(h_refined, h_true_white)
    loss_z = vector_nmse_loss(z_refined, z_true)
    loss_rho, loss_diag = graph_geometry_loss(gram_refined, gram_true)

    loss = loss_h + 0.25 * loss_z + 0.50 * loss_rho + 0.10 * loss_diag

    return loss, {
        "loss_h": loss_h.detach().item(),
        "loss_z": loss_z.detach().item(),
        "loss_rho": loss_rho.detach().item(),
        "loss_diag": loss_diag.detach().item(),
    }


def bit_errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


def ep5_llr(ep, z, gram):
    out = ep(z, gram, return_iterations=(5,))
    return out["iterations"][5]["llr"].reshape(-1, 16, 4).float()


def build_validation_set(dataset, num_channels, re_per_channel):
    y_all = []
    h_ls_all = []
    h_true_all = []
    bits_all = []

    print(f"Building fixed validation set: {num_channels} channels × {re_per_channel} RE")

    with torch.no_grad():
        for i in range(num_channels):
            sample = sample_selected_re(dataset, re_per_channel)

            y_all.append(sample["y_white"].cpu())
            h_ls_all.append(sample["h_ls_white"].cpu())
            h_true_all.append(sample["h_true_white"].cpu())
            bits_all.append(sample["bits"].cpu())

            if (i + 1) % 8 == 0 or i + 1 == num_channels:
                print(f"  validation channel {i + 1}/{num_channels}")

    return {
        "y_white": torch.cat(y_all, dim=0),
        "h_ls_white": torch.cat(h_ls_all, dim=0),
        "h_true_white": torch.cat(h_true_all, dim=0),
        "bits": torch.cat(bits_all, dim=0),
    }


@torch.no_grad()
def validate(model, ep, validation, device, chunk_size=256):
    model.eval()

    total_bits = 0
    errors_base = 0
    errors_gt = 0
    errors_true = 0

    h_input_err = 0.0
    h_refined_err = 0.0
    h_power = 0.0
    delta_rms_sum = 0.0
    num_chunks = 0

    total_re = validation["y_white"].shape[0]

    for start in range(0, total_re, chunk_size):
        end = min(start + chunk_size, total_re)

        y = validation["y_white"][start:end].to(device)
        h_ls = validation["h_ls_white"][start:end].to(device)
        h_true = validation["h_true_white"][start:end].to(device)
        bits = validation["bits"][start:end].to(device)

        output = model(y, h_ls)

        z_base, gram_base = sufficient_statistics(y, h_ls)
        z_true, gram_true = sufficient_statistics(y, h_true)

        llr_base = ep5_llr(ep, z_base, gram_base)
        llr_gt = ep5_llr(ep, output["z_refined"], output["gram_refined"])
        llr_true = ep5_llr(ep, z_true, gram_true)

        num_bits = bits.numel()
        total_bits += num_bits

        errors_base += bit_errors(llr_base, bits)
        errors_gt += bit_errors(llr_gt, bits)
        errors_true += bit_errors(llr_true, bits)

        h_input_err += (h_ls - h_true).abs().square().sum().item()
        h_refined_err += (output["h_refined"] - h_true).abs().square().sum().item()
        h_power += h_true.abs().square().sum().item()

        delta_rms_sum += output["relative_delta_rms"].mean().item()
        num_chunks += 1

    ber_base = errors_base / total_bits
    ber_gt = errors_gt / total_bits
    ber_true = errors_true / total_bits

    nmse_input = h_input_err / h_power
    nmse_refined = h_refined_err / h_power

    oracle_gap = ber_base - ber_true
    recovered = 0.0 if oracle_gap <= 0 else (ber_base - ber_gt) / oracle_gap

    return {
        "ber_base_ep5": ber_base,
        "ber_gt_ep5": ber_gt,
        "ber_trueH_ep5": ber_true,
        "ber_relative_gain": (ber_base - ber_gt) / max(ber_base, 1e-12),
        "oracle_gap_recovered": recovered,
        "nmse_input": nmse_input,
        "nmse_refined": nmse_refined,
        "nmse_input_db": 10.0 * math.log10(max(nmse_input, 1e-30)),
        "nmse_refined_db": 10.0 * math.log10(max(nmse_refined, 1e-30)),
        "relative_delta_rms": delta_rms_sum / max(num_chunks, 1),
    }


def save_checkpoint(path, model, optimizer, step, best_ber, metrics, args):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "best_ber": best_ber,
        "metrics": metrics,
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--re-per-step", type=int, default=128)
    parser.add_argument("--val-channels", type=int, default=32)
    parser.add_argument("--val-re-per-channel", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", default="ckp/csi_robust_gt_256rx_16ue_lsH_estR_5db")
    parser.add_argument("--history", default="results/raw/csi_robust_gt_256rx_16ue_lsH_estR_5db_history.json")

    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    system_cfg = load_yaml(train_cfg["system_config"])
    tcfg = train_cfg.get("training", {})

    snr_db = float(tcfg.get("snr_db", 5.0))
    system_cfg["link"]["rx_snr_db"] = snr_db

    device = system_cfg["general"]["device"]
    precision = system_cfg["general"]["precision"]
    seed = int(system_cfg["general"]["seed"])

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but unavailable.")

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    dataset = UplinkUMADataset(system_cfg)

    model = CSIRobustGraphTransformer(
        num_rx=256,
        num_users=16,
        d_model=128,
        num_heads=8,
        num_layers=8,
        edge_dim=32,
        ffn_dim=256,
        dropout=0.05,
        correction_scale=1.0,
    ).to(device)

    ep = ExpectationPropagationDetector(system_cfg, num_iterations=5, damping=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best.pth"
    init_path = output_dir / "init.pth"

    num_params = sum(p.numel() for p in model.parameters())

    print("=" * 120)
    print("CSI-Robust Graph Transformer Training")
    print("=" * 120)
    print(f"System              : 256 Rx / 16 users / 16-QAM")
    print(f"SNR                 : {snr_db:.1f} dB")
    print(f"Channel input       : LS H")
    print(f"Covariance input    : estimated R")
    print(f"Teacher             : true H whitened by the same estimated R")
    print(f"Graph nodes         : 16 users")
    print(f"Graph layers        : 8")
    print(f"d_model / heads     : 128 / 8")
    print(f"Parameters          : {num_params:,}")
    print(f"Steps               : {args.steps}")
    print(f"RE / step           : {args.re_per_step}")
    print(f"Learning rate       : {args.lr:.2e}")
    print("=" * 120)

    sionna_config.seed = seed + 1000
    torch.manual_seed(seed + 1000)

    validation = build_validation_set(dataset, args.val_channels, args.val_re_per_channel)

    sionna_config.seed = seed
    torch.manual_seed(seed)

    initial_metrics = validate(model, ep, validation, device)

    print()
    print("INITIAL VALIDATION")
    print("-" * 120)
    print(f"LS-H + Est-R + EP5 : {initial_metrics['ber_base_ep5']:.8e}")
    print(f"GT   + Est-R + EP5 : {initial_metrics['ber_gt_ep5']:.8e}")
    print(f"TrueH+ Est-R + EP5 : {initial_metrics['ber_trueH_ep5']:.8e}")
    print(f"LS whitened NMSE   : {initial_metrics['nmse_input_db']:+.3f} dB")
    print(f"GT whitened NMSE   : {initial_metrics['nmse_refined_db']:+.3f} dB")
    print(f"Initial ΔH RMS      : {initial_metrics['relative_delta_rms']:.6e}")
    print("-" * 120)

    if abs(initial_metrics["ber_gt_ep5"] - initial_metrics["ber_base_ep5"]) > 1e-10:
        raise RuntimeError("Zero-initialized GT does not exactly reproduce the EP5 baseline.")

    save_checkpoint(init_path, model, optimizer, 0, initial_metrics["ber_gt_ep5"], initial_metrics, args)

    best_ber = initial_metrics["ber_gt_ep5"]
    best_step = 0

    history = [{
        "step": 0,
        "lr": args.lr,
        **initial_metrics,
    }]

    running_loss = 0.0
    running_parts = {
        "loss_h": 0.0,
        "loss_z": 0.0,
        "loss_rho": 0.0,
        "loss_diag": 0.0,
    }

    for step in range(1, args.steps + 1):
        model.train()

        sample = sample_selected_re(dataset, args.re_per_step)

        y_white = sample["y_white"]
        h_ls_white = sample["h_ls_white"]
        h_true_white = sample["h_true_white"]

        optimizer.zero_grad(set_to_none=True)

        output = model(y_white, h_ls_white)
        loss, parts = training_loss(output, y_white, h_true_white)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.detach().item()

        for key in running_parts:
            running_parts[key] += parts[key]

        if step % 25 == 0:
            denom = 25.0

            print(
                f"step={step:5d} | "
                f"loss={running_loss / denom:.5f} | "
                f"H={running_parts['loss_h'] / denom:.5f} | "
                f"z={running_parts['loss_z'] / denom:.5f} | "
                f"rho={running_parts['loss_rho'] / denom:.5f} | "
                f"diag={running_parts['loss_diag'] / denom:.5f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

            running_loss = 0.0

            for key in running_parts:
                running_parts[key] = 0.0

        if step % args.val_every == 0:
            metrics = validate(model, ep, validation, device)

            print(
                f"VAL step={step:5d} | "
                f"EP5={metrics['ber_base_ep5']:.6e} | "
                f"GT={metrics['ber_gt_ep5']:.6e} | "
                f"TrueH={metrics['ber_trueH_ep5']:.6e} | "
                f"gain={100.0 * metrics['ber_relative_gain']:+.3f}% | "
                f"recover={100.0 * metrics['oracle_gap_recovered']:+.2f}% | "
                f"NMSE {metrics['nmse_input_db']:+.2f}->{metrics['nmse_refined_db']:+.2f} dB | "
                f"ΔH={metrics['relative_delta_rms']:.3f}"
            )

            history.append({
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                **metrics,
            })

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if metrics["ber_gt_ep5"] < best_ber:
                best_ber = metrics["ber_gt_ep5"]
                best_step = step

                save_checkpoint(best_path, model, optimizer, step, best_ber, metrics, args)

                print(f"  NEW BEST: BER={best_ber:.8e} at step {best_step}")

            if step - best_step >= args.patience:
                print()
                print(f"Early stopping: no BER improvement for {args.patience} steps.")
                break

    print()
    print("=" * 120)
    print("TRAINING FINISHED")
    print("=" * 120)
    print(f"Initial EP5 BER : {initial_metrics['ber_base_ep5']:.8e}")
    print(f"Best GT BER     : {best_ber:.8e}")
    print(f"Best step       : {best_step}")
    print(f"Relative gain   : {100.0 * (initial_metrics['ber_base_ep5'] - best_ber) / initial_metrics['ber_base_ep5']:+.3f}%")
    print(f"Checkpoint      : {best_path}")
    print(f"History         : {history_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()