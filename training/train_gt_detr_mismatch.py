#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from training.train_gt_detr_lmmse import (
    load_yaml,
    make_model,
    resolve_snr_range,
    build_snr_schedule,
)


def whiten(y, h, ruu):
    chol = torch.linalg.cholesky(ruu)

    if chol.shape[0] == 1 and y.shape[0] > 1:
        chol = chol.expand(
            y.shape[0],
            -1,
            -1,
        )

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


def hard_errors(llr, bits):
    return (
        (llr > 0)
        != (bits > 0.5)
    ).sum().item()


def make_sample(
    dataset,
    re_per_channel,
    ruu_mode,
    include_references=False,
):
    batch = dataset.sample(1)

    data_idx = list(
        dataset.channel
        .resource_grid
        .data_symbols
    )

    num_streams = int(
        batch["metadata"]["num_streams"]
    )

    bits_per_symbol = int(
        batch["metadata"]["bits_per_symbol"]
    )

    y_raw = batch["y"][:, data_idx]

    h_raw = (
        batch["h_hat_lmmse"][:, data_idx]
    )

    bits_raw = batch["bits"]

    num_rx = int(
        y_raw.shape[-1]
    )

    y = y_raw.reshape(
        -1,
        num_rx,
    )

    h = h_raw.reshape(
        -1,
        num_rx,
        num_streams,
    )

    bits = bits_raw.reshape(
        -1,
        num_streams,
        bits_per_symbol,
    ).float()

    count = min(
        int(re_per_channel),
        y.shape[0],
    )

    idx = torch.randperm(
        y.shape[0],
        device=y.device,
    )[:count]

    y = y[idx]
    h = h[idx]
    bits = bits[idx]

    ruu_hat = batch["ruu_hat"]
    ruu_true = batch["ruu_true"]

    if ruu_mode == "hat":
        ruu_selected = ruu_hat
    elif ruu_mode == "true":
        ruu_selected = ruu_true
    else:
        raise ValueError(
            f"Unknown ruu_mode: {ruu_mode}"
        )

    y_white, h_white = whiten(
        y,
        h,
        ruu_selected,
    )

    z, gram = sufficient_statistics(
        y_white,
        h_white,
    )

    result = {
        "z": z,
        "gram": gram,
        "bits": bits,
    }

    if include_references:
        y_hat, h_hat = whiten(
            y,
            h,
            ruu_hat,
        )

        z_hat, gram_hat = (
            sufficient_statistics(
                y_hat,
                h_hat,
            )
        )

        y_true, h_true = whiten(
            y,
            h,
            ruu_true,
        )

        z_true, gram_true = (
            sufficient_statistics(
                y_true,
                h_true,
            )
        )

        result.update({
            "z_hatR": z_hat,
            "gram_hatR": gram_hat,
            "z_trueR": z_true,
            "gram_trueR": gram_true,
        })

    return result


def build_validation_set(
    dataset,
    num_channels,
    re_per_channel,
    ruu_mode,
):
    storage = {
        "z": [],
        "gram": [],
        "z_hatR": [],
        "gram_hatR": [],
        "z_trueR": [],
        "gram_trueR": [],
        "bits": [],
    }

    print(
        f"Building fixed validation set: "
        f"{num_channels} channels × "
        f"{re_per_channel} RE"
    )

    with torch.no_grad():
        for i in range(num_channels):
            sample = make_sample(
                dataset,
                re_per_channel,
                ruu_mode,
                include_references=True,
            )

            for key in storage:
                storage[key].append(
                    sample[key].cpu()
                )

            if (
                (i + 1) % 8 == 0
                or i + 1 == num_channels
            ):
                print(
                    f"  validation channel "
                    f"{i + 1}/{num_channels}"
                )

    return {
        key: torch.cat(value, dim=0)
        for key, value in storage.items()
    }


@torch.no_grad()
def validate(
    model,
    ep5,
    validation,
    device,
    chunk_size=256,
):
    model.eval()

    total_bits = 0

    errors_selected_ep = 0
    errors_neural = 0
    errors_hatR_ep = 0
    errors_trueR_ep = 0

    total_re = validation["bits"].shape[0]

    correction_sum = torch.zeros(
        5,
        dtype=torch.float64,
    )

    validity_sum = torch.zeros(
        5,
        dtype=torch.float64,
    )

    num_chunks = 0

    for start in range(
        0,
        total_re,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            total_re,
        )

        z = validation["z"][
            start:end
        ].to(device)

        gram = validation["gram"][
            start:end
        ].to(device)

        z_hatR = validation["z_hatR"][
            start:end
        ].to(device)

        gram_hatR = validation[
            "gram_hatR"
        ][start:end].to(device)

        z_trueR = validation["z_trueR"][
            start:end
        ].to(device)

        gram_trueR = validation[
            "gram_trueR"
        ][start:end].to(device)

        bits = validation["bits"][
            start:end
        ].to(device)

        selected_out = ep5(
            z,
            gram,
            return_iterations=(5,),
        )

        hatR_out = ep5(
            z_hatR,
            gram_hatR,
            return_iterations=(5,),
        )

        trueR_out = ep5(
            z_trueR,
            gram_trueR,
            return_iterations=(5,),
        )

        neural_out = model(
            z,
            gram,
            return_iterations=(5,),
        )

        errors_selected_ep += (
            hard_errors(
                selected_out["llr"].float(),
                bits,
            )
        )

        errors_hatR_ep += hard_errors(
            hatR_out["llr"].float(),
            bits,
        )

        errors_trueR_ep += hard_errors(
            trueR_out["llr"].float(),
            bits,
        )

        errors_neural += hard_errors(
            neural_out["llr"].float(),
            bits,
        )

        total_bits += bits.numel()

        correction_sum += (
            neural_out["correction_rms"]
            .double()
            .cpu()
        )

        validity_sum += (
            neural_out[
                "valid_update_fraction"
            ]
            .double()
            .cpu()
        )

        num_chunks += 1

    ber_ep = (
        errors_selected_ep
        / total_bits
    )

    ber_neural = (
        errors_neural
        / total_bits
    )

    return {
        "ber_ep5": ber_ep,
        "ber_neural": ber_neural,
        "ber_hatR_ep5": (
            errors_hatR_ep
            / total_bits
        ),
        "ber_trueR_ep5": (
            errors_trueR_ep
            / total_bits
        ),
        "relative_gain_vs_ep5": (
            (ber_ep - ber_neural)
            / max(ber_ep, 1e-12)
        ),
        "correction_rms": (
            correction_sum
            / max(num_chunks, 1)
        ).tolist(),
        "valid_update_fraction": (
            validity_sum
            / max(num_chunks, 1)
        ).tolist(),
    }


@torch.no_grad()
def check_anchor(
    model,
    ep5,
    validation,
    device,
):
    z = validation["z"][
        :256
    ].to(device)

    gram = validation["gram"][
        :256
    ].to(device)

    ep_llr = ep5(
        z,
        gram,
        return_iterations=(5,),
    )["llr"]

    neural_llr = model(
        z,
        gram,
        return_iterations=(5,),
    )["llr"]

    diff = (
        ep_llr - neural_llr
    ).abs().max().item()

    print(
        "Initial max "
        "|Neural-EP - EP5| "
        f"LLR difference: {diff:.6e}"
    )

    if diff > 1e-5:
        raise RuntimeError(
            "Initial neural model does "
            "not reproduce EP5."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/training/"
            "sgt_5db.yaml"
        ),
    )

    parser.add_argument(
        "--arch",
        choices=[
            "gt_ep",
            "detr_ep",
        ],
        required=True,
    )

    parser.add_argument(
        "--ruu-mode",
        choices=[
            "hat",
            "true",
        ],
        required=True,
    )

    parser.add_argument(
        "--bits-per-symbol",
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
        "--snr-min-db",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--snr-max-db",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--val-snr-db",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--scheduler-steps",
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
        default=25,
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--history",
        required=True,
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
    )

    args = parser.parse_args()

    snr_min, snr_max, val_snr = (
        resolve_snr_range(args)
    )

    snr_schedule = (
        build_snr_schedule(
            args.steps,
            snr_min,
            snr_max,
            args.seed,
        )
    )

    train_cfg = load_yaml(
        args.config
    )

    cfg = load_yaml(
        train_cfg["system_config"]
    )

    cfg["modulation"][
        "bits_per_symbol"
    ] = int(args.bits_per_symbol)

    cfg["link"]["rx_snr_db"] = (
        float(val_snr)
    )

    cfg["general"]["seed"] = (
        int(args.seed)
    )

    device = cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = (
        cfg["general"]["precision"]
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    output_dir = Path(
        args.output_dir
    )

    history_path = Path(
        args.history
    )

    if args.fresh:
        if output_dir.exists():
            shutil.rmtree(
                output_dir
            )

        if history_path.exists():
            history_path.unlink()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best.pth"
    )

    sionna_config.seed = (
        int(args.seed)
    )

    torch.manual_seed(
        int(args.seed)
    )

    dataset = UplinkUMADataset(
        cfg
    )

    torch.manual_seed(
        int(args.seed) + 100
    )

    model = make_model(
        cfg,
        args.arch,
    ).to(device)

    ep5 = (
        ExpectationPropagationDetector(
            cfg,
            num_iterations=5,
            damping=0.5,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler_steps,
            eta_min=args.lr * 0.05,
        )
    )

    print("=" * 132)
    print(
        "LMMSE-H COVARIANCE-MISMATCH "
        "TRAINING"
    )
    print("=" * 132)
    print(
        f"Architecture        : "
        f"{args.arch}"
    )
    print(
        "Channel             : "
        "LMMSE H"
    )
    print(
        f"Ruu mode            : "
        f"{args.ruu_mode}"
    )
    print(
        f"Training SNR range  : "
        f"[{snr_min:.2f}, "
        f"{snr_max:.2f}] dB"
    )
    print(
        f"Validation SNR      : "
        f"{val_snr:.2f} dB"
    )
    print(
        f"Steps               : "
        f"{args.steps}"
    )
    print(
        f"Validation          : "
        f"{args.val_channels} ch × "
        f"{args.val_re_per_channel} RE"
    )
    print("=" * 132)

    dataset.link.rx_snr_db = (
        float(val_snr)
    )

    sionna_config.seed = (
        int(args.seed) + 2000
    )

    torch.manual_seed(
        int(args.seed) + 2000
    )

    validation = (
        build_validation_set(
            dataset,
            args.val_channels,
            args.val_re_per_channel,
            args.ruu_mode,
        )
    )

    check_anchor(
        model,
        ep5,
        validation,
        device,
    )

    initial = validate(
        model,
        ep5,
        validation,
        device,
    )

    print(
        f"Initial selected EP5 : "
        f"{initial['ber_ep5']:.8e}"
    )

    print(
        f"EP5 with hat Ruu     : "
        f"{initial['ber_hatR_ep5']:.8e}"
    )

    print(
        f"EP5 with true Ruu    : "
        f"{initial['ber_trueR_ep5']:.8e}"
    )

    best_ber = (
        initial["ber_neural"]
    )

    best_step = 0

    history = [{
        "step": 0,
        "arch": args.arch,
        "ruu_mode": args.ruu_mode,
        "snr_min_db": snr_min,
        "snr_max_db": snr_max,
        "val_snr_db": val_snr,
        **initial,
    }]

    torch.save(
        {
            "model_state": (
                model.state_dict()
            ),
            "step": 0,
            "best_ber": best_ber,
            "arch": args.arch,
            "ruu_mode": args.ruu_mode,
            "metrics": initial,
        },
        checkpoint_path,
    )

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

    running_loss = 0.0
    running_ber = 0.0
    running_count = 0

    for step in range(
        1,
        args.steps + 1,
    ):
        model.train()

        sampled_snr = float(
            snr_schedule[
                step - 1
            ]
        )

        dataset.link.rx_snr_db = (
            sampled_snr
        )

        physical_seed = (
            int(args.seed)
            * 100000
            + step
        )

        sionna_config.seed = (
            physical_seed
        )

        torch.manual_seed(
            physical_seed
        )

        sample = make_sample(
            dataset,
            args.re_per_step,
            args.ruu_mode,
            include_references=False,
        )

        z = sample["z"]
        gram = sample["gram"]
        bits = sample["bits"]

        optimizer.zero_grad(
            set_to_none=True
        )

        out = model(
            z,
            gram,
            return_iterations=(5,),
        )

        llr = out["llr"]

        loss = (
            F.binary_cross_entropy_with_logits(
                llr,
                bits,
            )
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss "
                f"at step {step}"
            )

        loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
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

        running_loss += (
            loss.item()
        )

        running_ber += (
            batch_ber
        )

        running_count += 1

        if (
            step % 25 == 0
            or step == args.steps
        ):
            print(
                f"step={step:4d} | "
                f"SNR={sampled_snr:+.2f} | "
                f"BCE="
                f"{running_loss / running_count:.5f} | "
                f"BER="
                f"{running_ber / running_count:.5f} | "
                f"grad="
                f"{float(grad_norm):.3f}"
            )

            running_loss = 0.0
            running_ber = 0.0
            running_count = 0

        if (
            step % args.val_every == 0
            or step == args.steps
        ):
            dataset.link.rx_snr_db = (
                float(val_snr)
            )

            metrics = validate(
                model,
                ep5,
                validation,
                device,
            )

            print(
                f"VAL step={step:4d} | "
                f"R={args.ruu_mode:<4s} | "
                f"EP5="
                f"{metrics['ber_ep5']:.6e} | "
                f"Neural="
                f"{metrics['ber_neural']:.6e} | "
                f"gain="
                f"{100.0 * metrics['relative_gain_vs_ep5']:+.3f}% | "
                f"hatR EP5="
                f"{metrics['ber_hatR_ep5']:.6e} | "
                f"trueR EP5="
                f"{metrics['ber_trueR_ep5']:.6e}"
            )

            history.append({
                "step": step,
                "arch": args.arch,
                "ruu_mode": args.ruu_mode,
                "snr_min_db": snr_min,
                "snr_max_db": snr_max,
                "val_snr_db": val_snr,
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

            if (
                metrics["ber_neural"]
                < best_ber
            ):
                best_ber = (
                    metrics[
                        "ber_neural"
                    ]
                )

                best_step = step

                torch.save(
                    {
                        "model_state": (
                            model
                            .state_dict()
                        ),
                        "step": step,
                        "best_ber": (
                            best_ber
                        ),
                        "arch": args.arch,
                        "ruu_mode": (
                            args.ruu_mode
                        ),
                        "metrics": (
                            metrics
                        ),
                    },
                    checkpoint_path,
                )

                print(
                    f"  NEW BEST | "
                    f"{best_ber:.8e} "
                    f"@ step {best_step}"
                )

    print()
    print("=" * 132)
    print("TRAINING FINISHED")
    print("=" * 132)
    print(
        f"Architecture        : "
        f"{args.arch}"
    )
    print(
        f"Channel             : "
        f"LMMSE H"
    )
    print(
        f"Ruu mode            : "
        f"{args.ruu_mode}"
    )
    print(
        f"Initial EP5 BER     : "
        f"{initial['ber_ep5']:.8e}"
    )
    print(
        f"Best neural BER     : "
        f"{best_ber:.8e}"
    )
    print(
        f"Best step           : "
        f"{best_step}"
    )
    print(
        f"Relative gain       : "
        f"{100.0 * (initial['ber_ep5'] - best_ber) / max(initial['ber_ep5'], 1e-12):+.3f}%"
    )
    print(
        f"EP5 hat-R reference : "
        f"{initial['ber_hatR_ep5']:.8e}"
    )
    print(
        f"EP5 true-R reference: "
        f"{initial['ber_trueR_ep5']:.8e}"
    )
    print(
        f"Checkpoint          : "
        f"{checkpoint_path}"
    )
    print("=" * 132)


if __name__ == "__main__":
    main()
