#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  ckp/ruu_mismatch \
  results/ruu_mismatch \
  logs/ruu_mismatch

for RUU in hat true
do
    for ARCH in gt_ep detr_ep
    do
        NAME="16qam_8db_lmmseH_${RUU}R_${ARCH}"

        echo
        echo "======================================================================"
        echo "${NAME}"
        echo "======================================================================"

        PYTHONUNBUFFERED=1 python -u \
          -m training.train_gt_detr_mismatch \
          --config configs/training/sgt_5db.yaml \
          --arch ${ARCH} \
          --ruu-mode ${RUU} \
          --bits-per-symbol 4 \
          --snr-min-db 7 \
          --snr-max-db 9 \
          --val-snr-db 8 \
          --steps 500 \
          --scheduler-steps 1000 \
          --re-per-step 128 \
          --val-channels 32 \
          --val-re-per-channel 64 \
          --val-every 25 \
          --seed 42 \
          --output-dir ckp/ruu_mismatch/${NAME} \
          --history results/ruu_mismatch/${NAME}.json \
          --fresh \
          2>&1 | tee logs/ruu_mismatch/${NAME}.log
    done
done

echo
echo "======================================================================"
echo "Ruu mismatch diagnostic finished"
echo "======================================================================"
