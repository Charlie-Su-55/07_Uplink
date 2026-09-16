#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

remove_path() {
  local p="$1"
  [[ -e "$p" ]] || return 0
  if [[ "$APPLY" -eq 1 ]]; then
    echo "[DELETE] $p"
    rm -rf -- "$p"
  else
    echo "[DRY-RUN] would delete: $p"
  fi
}

remove_glob() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  for p in "${matches[@]}"; do
    remove_path "$p"
  done
}

echo "======================================================================"
echo "07_Uplink cleanup | mode=$([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1) Diagnostics: all old one-off diagnostics are obsolete.
# ----------------------------------------------------------------------
remove_path "diagnostic"
remove_path "diagnostics"

# ----------------------------------------------------------------------
# 2) Evaluation: remove old LS / residual / generic refiner evaluators.
# Keep:
#   evaluation/sanity_lmmse_channel_estimation.py
#   evaluation/evaluate_gt_detr_lmmse.py
#   evaluation/evaluate_mcs_bler.py
# ----------------------------------------------------------------------
remove_path "evaluation/evaluate_ep_refiners.py"
remove_glob "evaluation/*residual*"
remove_glob "evaluation/*gepnet*"
remove_glob "evaluation/*edge_ablation*"
remove_glob "evaluation/*diagnos*"

# ----------------------------------------------------------------------
# 3) Models: keep only current GT and DETR proposal/control plus classical.
# Keep:
#   models/graph/gt_ep_detector.py
#   models/baselines/detr_ep_detector.py
# Delete failed/closed graph variants.
# ----------------------------------------------------------------------
remove_path "models/graph/gt_ep_residual.py"
remove_glob "models/graph/*gepnet*.py"
remove_glob "models/graph/*trustregion*.py"
remove_glob "models/graph/*decoupled*.py"
remove_glob "models/graph/*sparse*.py"
remove_glob "models/graph/*uncertainty_gate*.py"

# ----------------------------------------------------------------------
# 4) Checkpoints: delete obsolete LS-H, residual, GEPNet, diagnostics,
# edge-ablation and temporary runs. Keep current LMMSE-H GT/DETR runs.
# ----------------------------------------------------------------------
remove_glob "ckp/*lsH*"
remove_glob "ckp/*gepnet*"
remove_glob "ckp/*GEPNet*"
remove_glob "ckp/*residual*"
remove_glob "ckp/*trustregion*"
remove_glob "ckp/*decoupled*"
remove_glob "ckp/*edge_ablation*"
remove_glob "ckp/_tmp*"
remove_glob "ckp/*smoke*"
remove_glob "ckp/*diag*"

# ----------------------------------------------------------------------
# 5) Results: remove obsolete LS-H / old MCS / diagnostics / ablations.
# Keep current LMMSE-H histories and new lmmseH GT-vs-DETR result.
# ----------------------------------------------------------------------
remove_glob "results/raw/*lsH*"
remove_glob "results/raw/*gepnet*"
remove_glob "results/raw/*residual*"
remove_glob "results/raw/_tmp*"
remove_glob "results/raw/*edge_ablation*"
remove_path "results/mcs_bler"
remove_path "results/mcs_operating_point_100ch"
remove_glob "results/*diagnos*"
remove_glob "results/*edge_ablation*"

# Keep only new LMMSE-H comparison outputs in ep_refiners.
if [[ -d "results/ep_refiners" ]]; then
  shopt -s nullglob
  for p in results/ep_refiners/*; do
    base="$(basename "$p")"
    case "$base" in
      *lmmseH*) ;;
      *) remove_path "$p" ;;
    esac
  done
  shopt -u nullglob
fi

# ----------------------------------------------------------------------
# 6) Python caches.
# ----------------------------------------------------------------------
while IFS= read -r -d '' p; do
  remove_path "$p"
done < <(find . -type d -name __pycache__ -print0 2>/dev/null)

while IFS= read -r -d '' p; do
  remove_path "$p"
done < <(find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -print0 2>/dev/null)

echo "======================================================================"
if [[ "$APPLY" -eq 0 ]]; then
  echo "Dry-run only. If the list is correct, run:"
  echo "  bash tools/cleanup_uplink.sh --apply"
else
  echo "Cleanup finished."
fi
echo "======================================================================"
