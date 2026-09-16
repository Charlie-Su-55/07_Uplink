#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_platform_constellation(batch, qm):
    """Recover the exact platform symbol/bit mapping from generated data."""
    if "x_data" not in batch:
        raise KeyError(
            "batch['x_data'] is required to recover the platform constellation."
        )

    x = batch["x_data"].reshape(-1).to(torch.complex64)
    bits = batch["bits"].reshape(-1, qm).round().long()

    if x.numel() != bits.shape[0]:
        raise RuntimeError(
            f"x_data and bits are not aligned: {x.numel()} symbols vs "
            f"{bits.shape[0]} bit labels."
        )

    weights = (
        2 ** torch.arange(
            qm - 1,
            -1,
            -1,
            device=bits.device,
        )
    ).long()

    labels = (bits * weights).sum(-1)

    points = []
    bit_table = []

    for index in range(2 ** qm):
        mask = labels == index

        if not bool(mask.any()):
            raise RuntimeError(
                f"Constellation label {index} did not appear in the generated slot."
            )

        symbols = x[mask]
        reference = symbols[0]

        max_error = (symbols - reference).abs().max().item()

        if max_error > 1e-6:
            raise RuntimeError(
                f"Bit label {index} maps to multiple constellation points; "
                f"max deviation={max_error:.3e}"
            )

        points.append(reference)
        bit_table.append(bits[mask][0])

    points = torch.stack(points).to(torch.complex64)
    bit_table = torch.stack(bit_table).to(torch.float32)

    return points, bit_table


def hard_ber(llr, bits):
    return (
        ((llr > 0) != (bits > 0.5))
        .float()
        .mean()
        .item()
    )


def save_fallback_reproducer(
    path,
    y,
    h,
    ruu,
    bits,
    points,
    bit_table,
    report,
    max_re,
    metadata,
):
    """Fallback reproducer if the partner wrapper has no save_reproducer()."""
    n = min(
        int(max_re),
        int(y.shape[1] * y.shape[2]),
    )

    streams = int(h.shape[-1])
    qm = int(bits.shape[-1])

    y_small = (
        y[:1]
        .reshape(1, -1, 256)[:, :n]
        .reshape(1, 1, n, 256)
        .detach()
        .cpu()
    )

    h_small = (
        h[:1]
        .reshape(1, -1, 256, streams)[:, :n]
        .reshape(1, 1, n, 256, streams)
        .detach()
        .cpu()
    )

    bits_small = (
        bits[:1]
        .reshape(1, -1, streams, qm)[:, :n]
        .reshape(1, 1, n, streams, qm)
        .detach()
        .cpu()
    )

    torch.save(
        {
            "schema": "three-input-flow-reproducer-local-v1",
            "y": y_small,
            "h": h_small,
            "ruu": ruu[:1].detach().cpu(),
            "bits": bits_small,
            "constellation_points": points.detach().cpu(),
            "constellation_bits": bit_table.detach().cpu(),
            "report": report,
            "metadata": metadata,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--bundle",
        default="flow_three_input_partner",
    )

    parser.add_argument(
        "--qm",
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
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-re",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output-dir",
        default="results/partner_flow_diagnostic_v2",
    )

    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()

    if not bundle.is_dir():
        raise FileNotFoundError(
            f"Flow bundle not found: {bundle}"
        )

    diagnostics_path = bundle / "partner_diagnostics.py"

    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"partner_diagnostics.py not found: {diagnostics_path}"
        )

    sys.path.insert(0, str(bundle))

    from example_minimal_integration import PartnerFlowDetector
    from evaluation.partner_diagnostics_fixed import diagnose

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])

    # Runtime overrides only. YAML files are not modified.
    cfg["modulation"]["bits_per_symbol"] = int(args.qm)
    cfg["link"]["rx_snr_db"] = float(args.snr_db)
    cfg["general"]["seed"] = int(args.seed)
    cfg["general"]["device"] = args.device

    sionna_config.device = args.device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = int(args.seed)

    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    print("=" * 128)
    print("PARTNER FLOW DEPLOYMENT DIAGNOSTIC V2")
    print("=" * 128)
    print(f"Bundle              : {bundle}")
    print(f"Qm                  : {args.qm}")
    print(f"SNR                 : {args.snr_db:.1f} dB")
    print(f"Device              : {args.device}")
    print(f"Seed                : {args.seed}")
    print(f"Deep diagnostic RE  : {args.max_re}")
    print("Channel input       : LMMSE H")
    print("Covariance input    : estimated Ruu")
    print("=" * 128)

    dataset = UplinkUMADataset(cfg)
    batch = dataset.sample(1)

    data_idx = list(
        dataset.channel.resource_grid.data_symbols
    )

    y = (
        batch["y"][:, data_idx]
        .to(
            device=args.device,
            dtype=torch.complex64,
        )
        .contiguous()
    )

    h = (
        batch["h_hat_lmmse"][:, data_idx]
        .to(
            device=args.device,
            dtype=torch.complex64,
        )
        .contiguous()
    )

    ruu = (
        batch["ruu_hat"]
        .to(
            device=args.device,
            dtype=torch.complex64,
        )
        .contiguous()
    )

    bits = (
        batch["bits"]
        .to(
            device=args.device,
            dtype=torch.float32,
        )
        .contiguous()
    )

    expected_bits = (
        y.shape[0],
        y.shape[1],
        y.shape[2],
        h.shape[-1],
        args.qm,
    )

    if tuple(bits.shape) != expected_bits:
        raise RuntimeError(
            f"bits shape mismatch: got {tuple(bits.shape)}, "
            f"expected {expected_bits}"
        )

    points, bit_table = infer_platform_constellation(
        batch,
        args.qm,
    )

    print()
    print("PHYSICAL INPUT")
    print("-" * 128)
    print(f"y                   : {tuple(y.shape)} {y.dtype}")
    print(f"h_hat_lmmse         : {tuple(h.shape)} {h.dtype}")
    print(f"ruu_hat             : {tuple(ruu.shape)} {ruu.dtype}")
    print(f"bits                : {tuple(bits.shape)} {bits.dtype}")
    print(f"constellation points: {tuple(points.shape)}")
    print(f"constellation bits  : {tuple(bit_table.shape)}")
    print(
        f"constellation power : "
        f"{points.abs().square().mean().item():.6f}"
    )

    receiver = PartnerFlowDetector(
        Qm=args.qm,
        bundle_dir=str(bundle),
        device=args.device,
        constellation_points=points,
        constellation_bits=bit_table,
    )

    # Current wrapper stores the actual FlowDetector here.
    detector = getattr(
        receiver,
        "_detector",
        None,
    )

    if detector is None:
        raise AttributeError(
            "PartnerFlowDetector has no '_detector'. "
            "Please inspect the new example_minimal_integration.py API."
        )

    print()
    print("Running ordinary PartnerFlowDetector on the full slot...")

    with torch.inference_mode():
        llr = receiver(
            y,
            h,
            ruu,
        )

    if tuple(llr.shape) != expected_bits:
        raise RuntimeError(
            f"Flow LLR shape mismatch: {tuple(llr.shape)} "
            f"vs expected {expected_bits}"
        )

    if not bool(torch.isfinite(llr).all()):
        raise RuntimeError(
            "Flow output contains NaN or Inf."
        )

    ordinary_ber = hard_ber(
        llr,
        bits,
    )

    print(
        f"Full-slot Flow BER   : "
        f"{ordinary_ber:.8e}"
    )

    print()
    print("Running partner_diagnostics.py v2...")
    print(
        "Scale scan          : "
        "1, 1e2, 1e4, 1e5, 3e5, 1e6"
    )
    print(
        "Candidate sampling  : "
        f"first {args.max_re} RE, K=256"
    )
    print()

    report = diagnose(
        detector,
        y,
        h,
        ruu,
        bits=bits,
        max_re=args.max_re,
        deep_candidates=True,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "partner_diagnostic_v2.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    reproducer_path = (
        output_dir
        / "flow_reproducer.pt"
    )

    metadata = {
        "Qm": int(args.qm),
        "snr_db": float(args.snr_db),
        "seed": int(args.seed),
        "max_re": int(args.max_re),
        "channel_input": "h_hat_lmmse",
        "covariance_input": "ruu_hat",
        "full_slot_flow_ber": float(ordinary_ber),
    }

    # If the colleague's updated wrapper provides the official saver, use it.
    if hasattr(
        receiver,
        "save_reproducer",
    ):
        print(
            "Using partner-provided "
            "receiver.save_reproducer()."
        )

        receiver.save_reproducer(
            str(reproducer_path),
            y,
            h,
            ruu,
            bits=bits,
            report=report,
            max_re=args.max_re,
        )

    else:
        print(
            "Partner wrapper has no save_reproducer(); "
            "using local fallback saver."
        )

        save_fallback_reproducer(
            reproducer_path,
            y,
            h,
            ruu,
            bits,
            points,
            bit_table,
            report,
            args.max_re,
            metadata,
        )

    print()
    print("=" * 128)
    print("SUMMARY")
    print("=" * 128)

    if "ber" in report:
        for name, result in report["ber"].items():
            print(
                f"{name:<20}: "
                f"BER={result['overall']:.8e} | "
                f"errors={result['errors']}/{result['total_bits']}"
            )

    if "scale_rescue" in report:
        rescue = report["scale_rescue"]

        print()
        print("SCALE RESCUE")
        print("-" * 128)
        print(
            f"status              : "
            f"{rescue.get('status')}"
        )
        print(
            f"recommended scale   : "
            f"{rescue.get('recommended_common_scale')}"
        )
        print(
            f"recommended output  : "
            f"{rescue.get('recommended_output')}"
        )
        print(
            f"bounded LMMSE BER   : "
            f"{rescue.get('bounded_lmmse_ber')}"
        )
        print(
            f"best final BER      : "
            f"{rescue.get('bounded_best_final_ber')}"
        )
        print(
            f"best raw BER        : "
            f"{rescue.get('bounded_best_raw_ber')}"
        )

    if "verdicts" in report:
        print()
        print("VERDICTS")
        print("-" * 128)

        for verdict in report["verdicts"]:
            print(f"  {verdict}")

    if "common_scale_scan" in report:
        print()
        print("COMMON SCALE SCAN")
        print("-" * 128)
        print(
            f"{'scale':>12} | "
            f"{'LMMSE BER':>12} | "
            f"{'Flow BER':>12} | "
            f"{'joint truth':>12} | "
            f"{'oracle BER':>12} | "
            f"{'ESS frac':>12} | "
            f"{'alpha':>10}"
        )
        print("-" * 128)

        for scale, row in report[
            "common_scale_scan"
        ].items():
            ber = row.get(
                "ber",
                {},
            )

            candidate = row.get(
                "candidate",
                {},
            )

            lmmse_ber = (
                ber.get(
                    "lmmse_llr",
                    {},
                )
                .get(
                    "overall",
                    float("nan"),
                )
            )

            flow_ber = (
                ber.get(
                    "llr",
                    {},
                )
                .get(
                    "overall",
                    float("nan"),
                )
            )

            joint = candidate.get(
                "joint_truth_in_K",
                float("nan"),
            )

            oracle = candidate.get(
                "oracle_candidate_ber",
                float("nan"),
            )

            ess = (
                candidate.get(
                    "importance_ess_fraction",
                    {},
                )
                .get(
                    "mean",
                    float("nan"),
                )
            )

            alpha = (
                candidate.get(
                    "alpha",
                    {},
                )
                .get(
                    "mean",
                    float("nan"),
                )
            )

            print(
                f"{scale:>12} | "
                f"{lmmse_ber:>12.6e} | "
                f"{flow_ber:>12.6e} | "
                f"{joint:>12.6f} | "
                f"{oracle:>12.6e} | "
                f"{ess:>12.6e} | "
                f"{alpha:>10.6f}"
            )

    print()
    print("=" * 128)
    print("FILES SAVED")
    print("=" * 128)
    print(
        f"Diagnostic JSON     : "
        f"{report_path}"
    )
    print(
        f"Reproducer          : "
        f"{reproducer_path}"
    )
    print("=" * 128)


if __name__ == "__main__":
    main()
