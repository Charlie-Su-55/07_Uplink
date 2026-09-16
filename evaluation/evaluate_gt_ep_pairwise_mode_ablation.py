#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from models.graph.gt_ep_detector import GraphTransformerEPDetector
from models.graph.gt_ep_pairwise import PairwiseGraphTransformerEPDetector
from training.train_gt_ep_detector import hard_errors


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_v2(cfg, pairwise_mode):
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


def bootstrap(reference, method, n_boot=10000, seed=12345):
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
        "low": torch.quantile(
            samples,
            0.025,
        ).item(),
        "high": torch.quantile(
            samples,
            0.975,
        ).item(),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--v2-checkpoint",
        default="ckp/gt_ep_pairwise_v2_seed42/best.pth",
    )

    parser.add_argument(
        "--full-v1-checkpoint",
        default="ckp/gt_ep_edge_ablation/full_seed42/best.pth",
    )

    parser.add_argument(
        "--self-only-checkpoint",
        default="ckp/gt_ep_message_ablation/self_only_seed42/best.pth",
    )

    parser.add_argument(
        "--snr-db",
        type=float,
        default=5.0,
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
        "--seed",
        type=int,
        default=20260914,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output",
        default="results/gt_ep_pairwise_v2/pairwise_mode_ablation_5db.json",
    )

    args = parser.parse_args()

    train_cfg = load_yaml(
        args.config
    )

    cfg = load_yaml(
        train_cfg["system_config"]
    )

    cfg["link"]["rx_snr_db"] = float(
        args.snr_db
    )

    device = cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]

    torch.set_float32_matmul_precision(
        "high"
    )

    sionna_config.seed = int(
        args.seed
    )

    torch.manual_seed(
        int(args.seed)
    )

    dataset = UplinkUMADataset(
        cfg
    )

    ep = ExpectationPropagationDetector(
        cfg,
        num_iterations=5,
        damping=0.5,
    )

    v2_checkpoint = torch.load(
        args.v2_checkpoint,
        map_location=device,
        weights_only=False,
    )

    pairwise_models = {}

    for mode in [
        "full",
        "no_pairwise",
        "shuffled",
    ]:
        model = build_v2(
            cfg,
            pairwise_mode=mode,
        )

        model.load_state_dict(
            v2_checkpoint["model_state"],
            strict=True,
        )

        model.eval()

        pairwise_models[mode] = model

    full_v1 = build_v1(
        cfg,
        edge_mode="full",
        message_mode="cross_user",
    )

    full_v1_checkpoint = torch.load(
        args.full_v1_checkpoint,
        map_location=device,
        weights_only=False,
    )

    full_v1.load_state_dict(
        full_v1_checkpoint["model_state"],
        strict=True,
    )

    full_v1.eval()

    self_only = build_v1(
        cfg,
        edge_mode="no_edge",
        message_mode="self_only",
    )

    self_checkpoint = torch.load(
        args.self_only_checkpoint,
        map_location=device,
        weights_only=False,
    )

    self_only.load_state_dict(
        self_checkpoint["model_state"],
        strict=True,
    )

    self_only.eval()

    print("=" * 140)
    print("GT-EP v2 SAME-CHECKPOINT PAIRWISE MODE ABLATION")
    print("=" * 140)
    print(f"SNR               : {args.snr_db:.1f} dB")
    print(f"Channels          : {args.channels}")
    print(f"RE/channel        : {args.re_per_channel}")
    print(
        f"Bits              : "
        f"{args.channels * args.re_per_channel * 16 * 4}"
    )
    print(f"Evaluation seed   : {args.seed}")
    print(
        f"v2 checkpoint step: "
        f"{v2_checkpoint.get('step', -1)}"
    )
    print(
        f"v1 checkpoint step: "
        f"{full_v1_checkpoint.get('step', -1)}"
    )
    print(
        f"Self checkpoint   : "
        f"{self_checkpoint.get('step', -1)}"
    )
    print("=" * 140)

    names = [
        "ep5",
        "self_only",
        "full_v1",
        "v2_full",
        "v2_no_pairwise",
        "v2_shuffled",
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

    for channel_idx in range(
        1,
        args.channels + 1,
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
            args.re_per_channel,
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

        llr_self = self_only(
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_v1 = full_v1(
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_v2_full = pairwise_models["full"](
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_v2_no_pairwise = pairwise_models[
            "no_pairwise"
        ](
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        llr_v2_shuffled = pairwise_models[
            "shuffled"
        ](
            z,
            gram,
            return_iterations=(5,),
        )["llr"].float()

        outputs = {
            "ep5": llr_ep5,
            "self_only": llr_self,
            "full_v1": llr_v1,
            "v2_full": llr_v2_full,
            "v2_no_pairwise": llr_v2_no_pairwise,
            "v2_shuffled": llr_v2_shuffled,
        }

        nbits = bits.numel()
        total_bits += nbits

        for name, llr in outputs.items():
            errors = hard_errors(
                llr,
                bits,
            )

            total_errors[name] += errors

            per_channel[name].append(
                errors / nbits
            )

        if (
            channel_idx == 1
            or channel_idx % 50 == 0
            or channel_idx == args.channels
        ):
            current = {
                name: total_errors[name] / total_bits
                for name in names
            }

            print(
                f"{channel_idx:4d}/{args.channels} | "
                f"EP5={current['ep5']:.6e} | "
                f"Self={current['self_only']:.6e} | "
                f"V1={current['full_v1']:.6e} | "
                f"V2Full={current['v2_full']:.6e} | "
                f"NoPair={current['v2_no_pairwise']:.6e} | "
                f"Shuffle={current['v2_shuffled']:.6e}"
            )

    ber = {
        name: total_errors[name] / total_bits
        for name in names
    }

    comparisons = {
        "v2_full_vs_no_pairwise": bootstrap(
            per_channel["v2_no_pairwise"],
            per_channel["v2_full"],
            n_boot=args.bootstrap,
            seed=args.seed + 1,
        ),
        "v2_full_vs_shuffled": bootstrap(
            per_channel["v2_shuffled"],
            per_channel["v2_full"],
            n_boot=args.bootstrap,
            seed=args.seed + 2,
        ),
        "v2_full_vs_v1": bootstrap(
            per_channel["full_v1"],
            per_channel["v2_full"],
            n_boot=args.bootstrap,
            seed=args.seed + 3,
        ),
        "v1_vs_self_only": bootstrap(
            per_channel["self_only"],
            per_channel["full_v1"],
            n_boot=args.bootstrap,
            seed=args.seed + 4,
        ),
        "v2_full_vs_self_only": bootstrap(
            per_channel["self_only"],
            per_channel["v2_full"],
            n_boot=args.bootstrap,
            seed=args.seed + 5,
        ),
    }

    print()
    print("=" * 140)
    print("FINAL SAME-CHECKPOINT PAIRWISE ABLATION")
    print("=" * 140)

    for name in names:
        gain = (
            (ber["ep5"] - ber[name])
            / ber["ep5"]
            if name != "ep5"
            else 0.0
        )

        print(
            f"{name:16s}: "
            f"{ber[name]:.8e}"
            + (
                f" | gain vs EP5={100.0 * gain:+.3f}%"
                if name != "ep5"
                else ""
            )
        )

    print()
    print("PAIRWISE BRANCH CONTRIBUTION")

    for name in [
        "v2_full_vs_no_pairwise",
        "v2_full_vs_shuffled",
    ]:
        item = comparisons[name]

        print(
            f"{name:28s}: "
            f"ΔBER={item['delta']:+.6e} "
            f"95%CI=[{item['low']:+.6e}, "
            f"{item['high']:+.6e}]"
        )

    print()
    print("REFERENCE COMPARISONS")

    for name in [
        "v2_full_vs_v1",
        "v1_vs_self_only",
        "v2_full_vs_self_only",
    ]:
        item = comparisons[name]

        print(
            f"{name:28s}: "
            f"ΔBER={item['delta']:+.6e} "
            f"95%CI=[{item['low']:+.6e}, "
            f"{item['high']:+.6e}]"
        )

    result = {
        "snr_db": args.snr_db,
        "channels": args.channels,
        "re_per_channel": args.re_per_channel,
        "total_bits": total_bits,
        "evaluation_seed": args.seed,
        "v2_checkpoint_step": v2_checkpoint.get(
            "step",
            -1,
        ),
        "ber": ber,
        "comparisons": comparisons,
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
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
        f"Saved: {output_path}"
    )
    print("=" * 140)


if __name__ == "__main__":
    main()