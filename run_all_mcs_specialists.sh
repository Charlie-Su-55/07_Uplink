#!/usr/bin/env bash
set -euo pipefail

python -m training.train_mcs_specialists \
  --plan results/mcs_specialists/mcs_train_plan.csv \
  --config configs/training/sgt_5db.yaml \
  --steps 300 \
  --scheduler-steps 1000 \
  --val-every 25 \
  --val-channels 32 \
  --val-re-per-channel 64 \
  --re-per-step 128 \
  --arches gt_ep,detr_ep \
  --seed 42 \
  --fresh
