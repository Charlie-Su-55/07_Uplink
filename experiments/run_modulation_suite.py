#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_stage(name, args):
    print()
    print("=" * 140)
    print(f"RUNNING: {name}")
    print("=" * 140)
    print(" ".join(args))
    print("=" * 140)

    subprocess.run(
        args,
        check=True,
    )


def check_diagnostic(history_path):
    path = Path(history_path)

    if not path.exists():
        raise RuntimeError(
            f"Diagnostic history does not exist: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if len(history) < 2:
        raise RuntimeError(
            "Diagnostic history contains no trained validation point."
        )

    trained = [
        item
        for item in history
        if int(item.get("step", 0)) > 0
    ]

    max_corr = 0.0

    for item in trained:
        values = item.get("correction_rms", [])

        if values:
            max_corr = max(
                max_corr,
                max(float(x) for x in values),
            )

    if max_corr <= 1e-6:
        raise RuntimeError(
            "64-QAM diagnostic failed: Graph correction remained zero."
        )

    best = min(
        trained,
        key=lambda x: float(x["ber_gt"]),
    )

    print()
    print("=" * 140)
    print("64-QAM DIAGNOSTIC PASSED")
    print("=" * 140)
    print(f"Max correction RMS : {max_corr:.6f}")
    print(f"Best diagnostic step: {best['step']}")
    print(f"EP5 BER             : {float(best['ber_ep5']):.8e}")
    print(f"GT BER              : {float(best['ber_gt']):.8e}")
    print(
        f"Gain                 : "
        f"{100.0 * float(best['relative_gain_vs_ep5']):+.3f}%"
    )
    print("=" * 140)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )

    parser.add_argument(
        "--keep-diagnostics",
        action="store_true",
    )

    parser.add_argument(
        "--skip-qpsk",
        action="store_true",
    )

    parser.add_argument(
        "--skip-64qam",
        action="store_true",
    )

    args = parser.parse_args()

    py = sys.executable

    qpsk_output = "ckp/gt_ep_256rx_16ue_qpsk_lsH_estR_5db"
    qpsk_history = "results/raw/gt_ep_256rx_16ue_qpsk_lsH_estR_5db_history.json"

    qam64_smoke_output = "ckp/_tmp_gt_ep_64qam_smoke"
    qam64_smoke_history = "results/raw/_tmp_gt_ep_64qam_smoke.json"

    qam64_diag_output = "ckp/_tmp_gt_ep_64qam_diag"
    qam64_diag_history = "results/raw/_tmp_gt_ep_64qam_diag.json"

    qam64_output = "ckp/gt_ep_256rx_16ue_64qam_lsH_estR_5db"
    qam64_history = "results/raw/gt_ep_256rx_16ue_64qam_lsH_estR_5db_history.json"

    qam16_checkpoint = Path(
        "ckp/gt_ep_256rx_16ue_lsH_estR_5db/best.pth"
    )

    if qam16_checkpoint.exists():
        print(
            f"Existing 16-QAM v1 checkpoint found: "
            f"{qam16_checkpoint}"
        )
    else:
        print(
            "WARNING: existing 16-QAM checkpoint was not found. "
            "The suite will not retrain 16-QAM."
        )

    if not args.skip_qpsk:
        run_stage(
            "QPSK FORMAL TRAINING",
            [
                py,
                "-m",
                "training.train_gt_ep_modulation",
                "--config",
                args.config,
                "--bits-per-symbol",
                "2",
                "--steps",
                "1000",
                "--scheduler-steps",
                "1000",
                "--re-per-step",
                "128",
                "--val-channels",
                "32",
                "--val-re-per-channel",
                "64",
                "--val-every",
                "25",
                "--output-dir",
                qpsk_output,
                "--history",
                qpsk_history,
                "--fresh",
            ],
        )

    if not args.skip_64qam:
        run_stage(
            "64-QAM SMOKE TEST",
            [
                py,
                "-m",
                "training.train_gt_ep_modulation",
                "--config",
                args.config,
                "--bits-per-symbol",
                "6",
                "--steps",
                "1",
                "--scheduler-steps",
                "1000",
                "--re-per-step",
                "128",
                "--val-channels",
                "1",
                "--val-re-per-channel",
                "8",
                "--val-every",
                "1",
                "--output-dir",
                qam64_smoke_output,
                "--history",
                qam64_smoke_history,
                "--fresh",
            ],
        )

        run_stage(
            "64-QAM 100-STEP DIAGNOSTIC",
            [
                py,
                "-m",
                "training.train_gt_ep_modulation",
                "--config",
                args.config,
                "--bits-per-symbol",
                "6",
                "--steps",
                "100",
                "--scheduler-steps",
                "1000",
                "--re-per-step",
                "128",
                "--val-channels",
                "32",
                "--val-re-per-channel",
                "64",
                "--val-every",
                "25",
                "--output-dir",
                qam64_diag_output,
                "--history",
                qam64_diag_history,
                "--fresh",
            ],
        )

        check_diagnostic(
            qam64_diag_history
        )

        run_stage(
            "64-QAM FORMAL TRAINING",
            [
                py,
                "-m",
                "training.train_gt_ep_modulation",
                "--config",
                args.config,
                "--bits-per-symbol",
                "6",
                "--steps",
                "1000",
                "--scheduler-steps",
                "1000",
                "--re-per-step",
                "128",
                "--val-channels",
                "32",
                "--val-re-per-channel",
                "64",
                "--val-every",
                "25",
                "--output-dir",
                qam64_output,
                "--history",
                qam64_history,
                "--fresh",
            ],
        )

    if not args.keep_diagnostics:
        for path in [
            Path(qam64_smoke_output),
            Path(qam64_diag_output),
        ]:
            if path.exists():
                shutil.rmtree(path)

        for path in [
            Path(qam64_smoke_history),
            Path(qam64_diag_history),
        ]:
            if path.exists():
                path.unlink()

    print()
    print("=" * 140)
    print("MODULATION TRAINING SUITE FINISHED")
    print("=" * 140)

    checkpoints = {
        "QPSK": Path(qpsk_output) / "best.pth",
        "16-QAM": qam16_checkpoint,
        "64-QAM": Path(qam64_output) / "best.pth",
    }

    for name, path in checkpoints.items():
        status = "OK" if path.exists() else "MISSING"
        print(
            f"{name:8s}: {status:7s} | {path}"
        )

    print("=" * 140)


if __name__ == "__main__":
    main()