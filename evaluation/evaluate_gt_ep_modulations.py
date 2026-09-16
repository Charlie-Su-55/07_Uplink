#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.lmmse import LMMSESoftDetector
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.gt_ep_detector import GraphTransformerEPDetector


MODULATIONS = {
    2: {
        "name": "QPSK",
        "key": "qpsk",
        "checkpoint": "ckp/gt_ep_256rx_16ue_qpsk_lsH_estR_5db/best.pth",
    },
    4: {
        "name": "16-QAM",
        "key": "16qam",
        "checkpoint": "ckp/gt_ep_256rx_16ue_lsH_estR_5db/best.pth",
    },
    6: {
        "name": "64-QAM",
        "key": "64qam",
        "checkpoint": "ckp/gt_ep_256rx_16ue_64qam_lsH_estR_5db/best.pth",
    },
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hard_errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


def extract_state(checkpoint):
    for key in ("model_state", "state", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        raise KeyError(
            "Checkpoint does not contain model_state/state/state_dict."
        )

    cleaned = {}

    for key, value in state.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]

        cleaned[new_key] = value

    return cleaned


def make_model(cfg):
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
        edge_mode="full",
        message_mode="cross_user",
    )


def load_model(cfg, checkpoint_path, device):
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}"
        )

    model = make_model(cfg)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_state(checkpoint)

    model.load_state_dict(
        state,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    step = int(checkpoint.get("step", -1))

    return model, step


def select_re(batch, dataset, re_per_channel):
    data_idx = list(
        dataset.channel.resource_grid.data_symbols
    )

    y = batch["y"][:, data_idx]
    h_ls = batch["h_hat_ls"][:, data_idx]
    h_true = batch["h_true"][:, data_idx]
    bits = batch["bits"]

    num_streams = int(
        batch["metadata"]["num_streams"]
    )

    bits_per_symbol = int(
        batch["metadata"]["bits_per_symbol"]
    )

    num_rx = int(y.shape[-1])

    y = y.reshape(
        -1,
        num_rx,
    )

    h_ls = h_ls.reshape(
        -1,
        num_rx,
        num_streams,
    )

    h_true = h_true.reshape(
        -1,
        num_rx,
        num_streams,
    )

    bits = bits.reshape(
        -1,
        num_streams,
        bits_per_symbol,
    ).float()

    if not (
        y.shape[0]
        == h_ls.shape[0]
        == h_true.shape[0]
        == bits.shape[0]
    ):
        raise RuntimeError(
            f"RE mismatch: y={y.shape[0]}, "
            f"h_ls={h_ls.shape[0]}, "
            f"h_true={h_true.shape[0]}, "
            f"bits={bits.shape[0]}"
        )

    count = min(
        int(re_per_channel),
        y.shape[0],
    )

    idx = torch.randperm(
        y.shape[0],
        device=y.device,
    )[:count]

    y = y[idx]
    h_ls = h_ls[idx]
    h_true = h_true[idx]
    bits = bits[idx]

    return {
        "y": y.reshape(1, 1, count, num_rx),
        "h_ls": h_ls.reshape(1, 1, count, num_rx, num_streams),
        "h_true": h_true.reshape(1, 1, count, num_rx, num_streams),
        "bits": bits,
        "ruu_hat": batch["ruu_hat"],
    }


def paired_bootstrap(
    errors_a,
    errors_b,
    bits_per_channel,
    repetitions,
    seed,
):
    a = np.asarray(
        errors_a,
        dtype=np.float64,
    )

    b = np.asarray(
        errors_b,
        dtype=np.float64,
    )

    if a.shape != b.shape:
        raise ValueError(
            "Paired bootstrap requires matching arrays."
        )

    rng = np.random.default_rng(seed)

    per_channel_delta = (
        a - b
    ) / float(bits_per_channel)

    n = len(per_channel_delta)

    observed = float(
        per_channel_delta.mean()
    )

    samples = np.empty(
        repetitions,
        dtype=np.float64,
    )

    for i in range(repetitions):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        samples[i] = float(
            per_channel_delta[idx].mean()
        )

    lo, hi = np.percentile(
        samples,
        [2.5, 97.5],
    )

    return {
        "delta_ber": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


@torch.no_grad()
def evaluate_modulation(
    base_cfg,
    bits_per_symbol,
    checkpoint_path,
    channels,
    re_per_channel,
    snr_db,
    eval_seed,
    bootstrap_repetitions,
):
    cfg = copy.deepcopy(base_cfg)

    cfg["modulation"]["bits_per_symbol"] = int(
        bits_per_symbol
    )

    cfg["link"]["rx_snr_db"] = float(
        snr_db
    )

    cfg["general"]["seed"] = int(
        eval_seed
    )

    device = cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(eval_seed)

    torch.manual_seed(
        int(eval_seed)
    )

    dataset = UplinkUMADataset(cfg)

    lmmse = LMMSESoftDetector(cfg)

    ep = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    model, checkpoint_step = load_model(
        cfg,
        checkpoint_path,
        device,
    )

    mod_name = MODULATIONS[
        bits_per_symbol
    ]["name"]

    print()
    print("=" * 128)
    print(
        f"INDEPENDENT EVALUATION | {mod_name}"
    )
    print("=" * 128)
    print(
        f"Checkpoint        : {checkpoint_path}"
    )
    print(
        f"Checkpoint step   : {checkpoint_step}"
    )
    print(
        f"SNR               : {snr_db:.1f} dB"
    )
    print(
        f"Channels          : {channels}"
    )
    print(
        f"RE / channel      : {re_per_channel}"
    )
    print(
        f"Bits / symbol     : {bits_per_symbol}"
    )
    print(
        f"Evaluation seed   : {eval_seed}"
    )
    print("=" * 128)

    errors = {
        "lmmse": [],
        "ep5": [],
        "gt": [],
        "trueH_lmmse": [],
        "trueH_ep5": [],
    }

    correction_sum = torch.zeros(
        5,
        dtype=torch.float64,
    )

    validity_sum = torch.zeros(
        5,
        dtype=torch.float64,
    )

    total_bits = 0

    for channel_idx in range(
        1,
        channels + 1,
    ):
        batch = dataset.sample(1)

        sample = select_re(
            batch,
            dataset,
            re_per_channel,
        )

        y = sample["y"]
        h_ls = sample["h_ls"]
        h_true = sample["h_true"]
        bits = sample["bits"]
        ruu_hat = sample["ruu_hat"]

        if y.ndim != 4:
            raise RuntimeError(f"Expected selected y [B,S,F,M], got {tuple(y.shape)}")

        if h_ls.ndim != 5 or h_true.ndim != 5:
            raise RuntimeError(
                f"Expected selected h [B,S,F,M,K], got "
                f"h_ls={tuple(h_ls.shape)}, h_true={tuple(h_true.shape)}"
            )
        
        physical_ls = lmmse(
            y,
            h_ls,
            ruu_hat,
        )

        physical_true = lmmse(
            y,
            h_true,
            ruu_hat,
        )

        z_ls = physical_ls[
            "z"
        ].reshape(
            -1,
            16,
        )

        gram_ls = physical_ls[
            "gram"
        ].reshape(
            -1,
            16,
            16,
        )

        llr_lmmse = physical_ls[
            "llr"
        ].reshape(
            -1,
            16,
            bits_per_symbol,
        ).float()

        z_true = physical_true[
            "z"
        ].reshape(
            -1,
            16,
        )

        gram_true = physical_true[
            "gram"
        ].reshape(
            -1,
            16,
            16,
        )

        llr_true_lmmse = physical_true[
            "llr"
        ].reshape(
            -1,
            16,
            bits_per_symbol,
        ).float()

        ep_ls = ep(
            z_ls,
            gram_ls,
            return_iterations=(5,),
        )

        ep_true = ep(
            z_true,
            gram_true,
            return_iterations=(5,),
        )

        gt = model(
            z_ls,
            gram_ls,
            return_iterations=(5,),
        )

        llr_ep5 = ep_ls[
            "llr"
        ].float()

        llr_true_ep5 = ep_true[
            "llr"
        ].float()

        llr_gt = gt[
            "llr"
        ].float()

        bits = bits.to(
            llr_gt.device
        )

        n_bits = bits.numel()

        errors["lmmse"].append(
            hard_errors(
                llr_lmmse,
                bits,
            )
        )

        errors["ep5"].append(
            hard_errors(
                llr_ep5,
                bits,
            )
        )

        errors["gt"].append(
            hard_errors(
                llr_gt,
                bits,
            )
        )

        errors["trueH_lmmse"].append(
            hard_errors(
                llr_true_lmmse,
                bits,
            )
        )

        errors["trueH_ep5"].append(
            hard_errors(
                llr_true_ep5,
                bits,
            )
        )

        total_bits += n_bits

        correction_sum += (
            gt["correction_rms"]
            .double()
            .cpu()
        )

        validity_sum += (
            gt["valid_update_fraction"]
            .double()
            .cpu()
        )

        if (
            channel_idx % 25 == 0
            or channel_idx == channels
        ):
            ber_lmmse = (
                sum(errors["lmmse"])
                / total_bits
            )

            ber_ep5 = (
                sum(errors["ep5"])
                / total_bits
            )

            ber_gt = (
                sum(errors["gt"])
                / total_bits
            )

            ber_true = (
                sum(errors["trueH_ep5"])
                / total_bits
            )

            print(
                f"{channel_idx:4d}/{channels} | "
                f"LMMSE={ber_lmmse:.6e} | "
                f"EP5={ber_ep5:.6e} | "
                f"GT={ber_gt:.6e} | "
                f"TrueH={ber_true:.6e}"
            )

        del batch
        del sample
        del physical_ls
        del physical_true
        del ep_ls
        del ep_true
        del gt

    ber = {
        key: (
            sum(value)
            / total_bits
        )
        for key, value in errors.items()
    }

    bits_per_channel = (
        re_per_channel
        * 16
        * bits_per_symbol
    )

    bootstrap_gt_vs_lmmse = paired_bootstrap(
        errors["lmmse"],
        errors["gt"],
        bits_per_channel,
        bootstrap_repetitions,
        eval_seed + 1001,
    )

    bootstrap_gt_vs_ep5 = paired_bootstrap(
        errors["ep5"],
        errors["gt"],
        bits_per_channel,
        bootstrap_repetitions,
        eval_seed + 1002,
    )

    bootstrap_ep5_vs_lmmse = paired_bootstrap(
        errors["lmmse"],
        errors["ep5"],
        bits_per_channel,
        bootstrap_repetitions,
        eval_seed + 1003,
    )

    relative_gt_vs_lmmse = (
        ber["lmmse"]
        - ber["gt"]
    ) / max(
        ber["lmmse"],
        1e-12,
    )

    relative_gt_vs_ep5 = (
        ber["ep5"]
        - ber["gt"]
    ) / max(
        ber["ep5"],
        1e-12,
    )

    relative_ep5_vs_lmmse = (
        ber["lmmse"]
        - ber["ep5"]
    ) / max(
        ber["lmmse"],
        1e-12,
    )

    oracle_gap = (
        ber["ep5"]
        - ber["trueH_ep5"]
    )

    recovered = (
        0.0
        if oracle_gap <= 0
        else (
            ber["ep5"]
            - ber["gt"]
        ) / oracle_gap
    )

    correction_mean = (
        correction_sum
        / channels
    ).tolist()

    validity_mean = (
        validity_sum
        / channels
    ).tolist()

    result = {
        "modulation": mod_name,
        "bits_per_symbol": bits_per_symbol,
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "snr_db": snr_db,
        "channels": channels,
        "re_per_channel": re_per_channel,
        "total_bits": total_bits,
        "eval_seed": eval_seed,
        "ber": ber,
        "relative_gain": {
            "gt_vs_lmmse": relative_gt_vs_lmmse,
            "gt_vs_ep5": relative_gt_vs_ep5,
            "ep5_vs_lmmse": relative_ep5_vs_lmmse,
            "oracle_gap_recovered": recovered,
        },
        "bootstrap": {
            "gt_vs_lmmse": bootstrap_gt_vs_lmmse,
            "gt_vs_ep5": bootstrap_gt_vs_ep5,
            "ep5_vs_lmmse": bootstrap_ep5_vs_lmmse,
        },
        "correction_rms": correction_mean,
        "valid_update_fraction": validity_mean,
        "per_channel_errors": errors,
    }

    print()
    print("-" * 128)
    print(
        f"{mod_name} FINAL"
    )
    print("-" * 128)

    print(
        f"LMMSE           : "
        f"{ber['lmmse']:.8e}"
    )

    print(
        f"EP5             : "
        f"{ber['ep5']:.8e}"
    )

    print(
        f"GT-EP v1        : "
        f"{ber['gt']:.8e}"
    )

    print(
        f"TrueH LMMSE     : "
        f"{ber['trueH_lmmse']:.8e}"
    )

    print(
        f"TrueH EP5       : "
        f"{ber['trueH_ep5']:.8e}"
    )

    print()

    print(
        f"GT vs LMMSE     : "
        f"{100.0 * relative_gt_vs_lmmse:+.3f}%"
    )

    print(
        f"GT vs EP5       : "
        f"{100.0 * relative_gt_vs_ep5:+.3f}%"
    )

    print(
        f"EP5 vs LMMSE    : "
        f"{100.0 * relative_ep5_vs_lmmse:+.3f}%"
    )

    print(
        f"Oracle recovered: "
        f"{100.0 * recovered:+.2f}%"
    )

    print()

    b = bootstrap_gt_vs_lmmse

    print(
        "GT vs LMMSE paired ΔBER: "
        f"{b['delta_ber']:+.6e} "
        f"95%CI=["
        f"{b['ci95_low']:+.6e}, "
        f"{b['ci95_high']:+.6e}]"
    )

    b = bootstrap_gt_vs_ep5

    print(
        "GT vs EP5 paired ΔBER  : "
        f"{b['delta_ber']:+.6e} "
        f"95%CI=["
        f"{b['ci95_low']:+.6e}, "
        f"{b['ci95_high']:+.6e}]"
    )

    print("-" * 128)

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--re-per-channel",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--snr-db",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--modulations",
        default="2,4,6",
        help="Comma-separated bits/symbol, e.g. 2,4,6",
    )

    parser.add_argument(
        "--output",
        default=(
            "results/gt_ep_modulation_eval/"
            "independent_500ch_128re_5db.json"
        ),
    )

    args = parser.parse_args()

    train_cfg = load_yaml(
        args.config
    )

    base_cfg = load_yaml(
        train_cfg["system_config"]
    )

    requested = [
        int(x.strip())
        for x in args.modulations.split(",")
        if x.strip()
    ]

    for bps in requested:
        if bps not in MODULATIONS:
            raise ValueError(
                f"Unsupported bits/symbol: {bps}"
            )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    for bps in requested:
        meta = MODULATIONS[bps]

        result = evaluate_modulation(
            base_cfg=base_cfg,
            bits_per_symbol=bps,
            checkpoint_path=meta["checkpoint"],
            channels=args.channels,
            re_per_channel=args.re_per_channel,
            snr_db=args.snr_db,
            eval_seed=args.seed + bps * 10000,
            bootstrap_repetitions=args.bootstrap,
        )

        all_results[
            meta["key"]
        ] = result

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                all_results,
                f,
                indent=2,
            )

    print()
    print("=" * 128)
    print(
        "FINAL MODULATION COMPARISON"
    )
    print("=" * 128)

    print(
        f"{'Modulation':<12}"
        f"{'LMMSE':>14}"
        f"{'EP5':>14}"
        f"{'GT-v1':>14}"
        f"{'GT/LMMSE':>14}"
        f"{'GT/EP5':>12}"
    )

    print("-" * 128)

    for bps in requested:
        meta = MODULATIONS[bps]
        r = all_results[
            meta["key"]
        ]

        ber = r["ber"]
        gain = r["relative_gain"]

        print(
            f"{meta['name']:<12}"
            f"{ber['lmmse']:>14.6e}"
            f"{ber['ep5']:>14.6e}"
            f"{ber['gt']:>14.6e}"
            f"{100.0 * gain['gt_vs_lmmse']:>+13.3f}%"
            f"{100.0 * gain['gt_vs_ep5']:>+11.3f}%"
        )

    print("-" * 128)

    print(
        f"Saved: {output_path}"
    )

    print("=" * 128)


if __name__ == "__main__":
    main()