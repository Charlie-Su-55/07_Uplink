#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import subprocess
import sys
from pathlib import Path


MOD_NAMES = {
    2: "qpsk",
    4: "16qam",
    6: "64qam",
}


def snr_tag(x):
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def is_enabled(value):
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="results/mcs_specialists/mcs_train_plan.csv",
    )
    parser.add_argument(
        "--config",
        default="configs/training/sgt_5db.yaml",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--scheduler-steps", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--val-channels", type=int, default=32)
    parser.add_argument("--val-re-per-channel", type=int, default=64)
    parser.add_argument("--re-per-step", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arches", default="gt_ep,detr_ep")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    arches = [x.strip() for x in args.arches.split(",") if x.strip()]

    for arch in arches:
        if arch not in ("gt_ep", "detr_ep"):
            raise ValueError(f"Unsupported architecture: {arch}")

    plan_path = Path(args.plan)

    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    with open(plan_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {
        "table",
        "mcs",
        "qm",
        "snr_min_db",
        "snr_max_db",
        "val_snr_db",
    }

    if not rows or not required.issubset(rows[0].keys()):
        raise RuntimeError(
            "Plan must contain columns: "
            + ", ".join(sorted(required))
        )

    active_rows = [
        row
        for row in rows
        if is_enabled(row.get("enabled", "1"))
    ]

    print("=" * 124)
    print("MCS SPECIALIST TRAINING | GT-EP vs DETR-EP")
    print("=" * 124)
    print(f"Plan               : {plan_path}")
    print(f"Active MCS         : {len(active_rows)}")
    print(f"Architectures      : {arches}")
    print(f"Total runs         : {len(active_rows) * len(arches)}")
    print(f"Steps / run        : {args.steps}")
    print("=" * 124)

    for row in active_rows:
        table = int(row["table"])
        mcs = int(row["mcs"])
        qm = int(row["qm"])
        snr_min = float(row["snr_min_db"])
        snr_max = float(row["snr_max_db"])
        val_snr = float(row["val_snr_db"])
        mod = MOD_NAMES[qm]

        for arch in arches:
            run_name = (
                f"t{table}_mcs{mcs}_{mod}_{arch}_"
                f"lmmseH_estR_"
                f"snr{snr_tag(snr_min)}to{snr_tag(snr_max)}db"
            )

            output_dir = f"ckp/mcs_specialists/{run_name}"
            history = f"results/mcs_specialists/history/{run_name}.json"

            cmd = [
                sys.executable,
                "-m",
                "training.train_gt_detr_lmmse",
                "--config",
                args.config,
                "--arch",
                arch,
                "--bits-per-symbol",
                str(qm),
                "--snr-min-db",
                str(snr_min),
                "--snr-max-db",
                str(snr_max),
                "--val-snr-db",
                str(val_snr),
                "--steps",
                str(args.steps),
                "--scheduler-steps",
                str(args.scheduler_steps),
                "--re-per-step",
                str(args.re_per_step),
                "--val-channels",
                str(args.val_channels),
                "--val-re-per-channel",
                str(args.val_re_per_channel),
                "--val-every",
                str(args.val_every),
                "--seed",
                str(args.seed),
                "--output-dir",
                output_dir,
                "--history",
                history,
            ]

            if args.fresh:
                cmd.append("--fresh")

            print()
            print("-" * 124)
            print(
                f"T{table} MCS{mcs} | "
                f"{mod.upper()} | "
                f"train SNR=[{snr_min:+.2f},{snr_max:+.2f}] dB | "
                f"val={val_snr:+.2f} dB | "
                f"{arch}"
            )
            print(" ".join(cmd))
            print("-" * 124)

            subprocess.run(cmd, check=True)

    print()
    print("=" * 124)
    print("ALL MCS SPECIALISTS FINISHED")
    print("=" * 124)


if __name__ == "__main__":
    main()
