# 07_Uplink

Practical mismatch-robust soft MU-MIMO detection for the ICC submission.
The receiver uses DMRS -> conventional LMMSE channel estimation/interpolation ->
`h_hat_lmmse + ruu_hat` -> covariance whitening -> LMMSE / EP5 / GT-EP / DETR-EP -> LLR -> coded BLER.
GT-EP is the proposal; DETR-EP is a control. Report measured differences without assuming GT must win.
An optional [Flow-Matching control](docs/FLOW_MATCHING.md) adds a discrete masked posterior
flow with the same whitened z/G interface, a dedicated trainer, and paired BER/BLER integration.

Detailed findings, baseline line references and limitations: [ICC audit](docs/ICC_AUDIT_20260918.md).

## Fixed experiment

- 256 BS Rx antennas, 16 UEs with one stream each; 6.7 GHz.
- 192 subcarriers, 14 OFDM symbols; silent 0/1, DMRS 2/13, data 3..12.
- UMa forced NLOS, outdoor/O2I mixture; four external interferers, I/N = 10 dB; no channel normalization.
- `Ruu = E[(i+n)(i+n)^H]`; `ruu_hat` already includes thermal noise. Shrinkage lambda = 0.10.
- LLR = log P(bit=1)/P(bit=0); hard bit = `llr > 0`.
- EP5: five iterations. GT: four layers, d=128, eight heads, edge=32, FFN=256, dropout=.05.
  DETR: three layers, d=128, eight heads, FFN=256, dropout=.05. Both refine five EP iterations.
- True-H is restricted to oracle diagnostics and simulator forward signal generation.
- Current DMRS is an interference-protected, power-boosted interleaved comb. CE assumes known thermal-noise variance and an offline channel covariance cache. State these assumptions in the paper.
- The historical `configs/training/sgt_5db.yaml` path is retained because active commands read its
  `system_config` field. Its old SGT model/receiver/training fields are not used by the canonical scripts.
  Architecture and SNR come from the canonical builders and CLI. Do not edit YAML for temporary experiments.

## Active entrypoints

| Purpose | Command/module |
|---|---|
| Train ordinary GT/DETR | `python -m training.train_gt_detr_lmmse` |
| Train Flow control | `python -m training.train_flow_matching` |
| Run MCS training plan | `python -m training.train_mcs_specialists` |
| Paired uncoded BER and channel bootstrap | `python -m evaluation.evaluate_gt_detr_lmmse` |
| Coded BLER and fixed-BLER crossing | `python -m evaluation.evaluate_mcs_bler` |
| LS/LMMSE/True-H CE diagnostic | `python -m evaluation.sanity_lmmse_channel_estimation` |
| Build independent CE covariance cache | `python -m tools.build_uma_lmmse_ft_cov` |
| Inspect NR MCS | `python -m link_level.nr_mcs` |

The scalar UA-GT experiment is retired. Normal GT tensor names and shapes are preserved.
The dataset's ambiguous `h_hat` alias and unused covariance split views were removed;
use `h_hat_lmmse` for practical detection and `h_true` explicitly for diagnostics.
LS estimation remains available to CE sanity checks.

## Local CPU checks

No 256Rx simulation, UMa generation or neural training on the laptop.
The audit tests use Python's standard library and synthetic evaluator callbacks.
Flow tests also use small synthetic PyTorch tensors; its Sionna 2 mapper/APP check
skips when Sionna 2 is unavailable. No test generates UMa channels.

```bash
python -m compileall -q data detectors models training evaluation link_level tools tests
python -m unittest discover -s tests -v
```

The audited laptop has torch 2.6.0 / Sionna 0.15.1 and lacks PyYAML; its environment was not changed.
Real imports and the commands below require the existing GPU-server environment:
Python 3.12, torch 2.11 + CUDA 13, Sionna 2.0.1, PyYAML.

## Server validation: smoke before formal

Run from the repository root after reviewing the audit branch diff and merging on the server.
Keep the server's existing covariance cache and checkpoints. Do not overwrite or regenerate them for this smoke.
These commands use the T1/MCS11 paths already declared in the baseline repository.

```bash
python -c "import torch, sionna, yaml; print('torch', torch.__version__, 'sionna', sionna.__version__); assert torch.cuda.is_available()"
python -m py_compile models/graph/gt_ep_detector.py models/baselines/detr_ep_detector.py training/train_gt_detr_lmmse.py training/train_mcs_specialists.py evaluation/evaluate_gt_detr_lmmse.py evaluation/evaluate_mcs_bler.py
python -m unittest discover -s tests -v
python -m training.train_gt_detr_lmmse --help
python -m evaluation.evaluate_gt_detr_lmmse --help
python -m evaluation.evaluate_mcs_bler --help

test -s data/cache/uma_lmmse_ft_cov.pt
test -s ckp/mcs_specialists/t1_mcs11_16qam_gt_ep_lmmseH_estR_snr7to9db/best.pth
test -s ckp/mcs_specialists/t1_mcs11_16qam_detr_ep_lmmseH_estR_snr7to9db/best.pth
sha256sum data/cache/uma_lmmse_ft_cov.pt

python -m evaluation.evaluate_gt_detr_lmmse \
  --bits-per-symbol 4 --snr-db 8 --channels 2 --re-per-channel 8 \
  --chunk-size 8 --bootstrap 100 --print-every 1 --seed 20260918 \
  --gt-checkpoint ckp/mcs_specialists/t1_mcs11_16qam_gt_ep_lmmseH_estR_snr7to9db/best.pth \
  --detr-checkpoint ckp/mcs_specialists/t1_mcs11_16qam_detr_ep_lmmseH_estR_snr7to9db/best.pth \
  --output results/icc_audit_smoke/ber.json

python -m evaluation.evaluate_mcs_bler \
  --tables 1 --mcs 11 --operating-point --formal-channels 2 \
  --gt-chunk 128 --seed 20260918 --output-dir results/icc_audit_smoke/bler_point

python -m evaluation.evaluate_mcs_bler \
  --tables 1 --mcs 11 --coarse-snrs 7,8,9 \
  --coarse-channels 1 --formal-channels 2 --formal-step-db 0.5 --formal-margin-db 0 \
  --bracket-step-db 0.5 --max-bracket-extensions 2 --gt-chunk 128 \
  --seed 20260918 --output-dir results/icc_audit_smoke/bler_grid
```

BLER smoke must pass the codec identity check, `y_clean = H*x` reconstruction check and detector LLR shape checks.
Legacy specialist checkpoints may lack `mcs_table/mcs_index`; the loader records and warns about those unverified fields.
Check them against the original training history before using a formal result. Contradictory metadata is rejected.
A two-channel smoke can be nonmonotone or fail to bracket; that is not a performance finding.
Its purpose is to validate execution, bounded extension and saved schema.

Check ordinary training after UA cleanup with a single step per model (GPU server only):

```bash
for arch in gt_ep detr_ep; do
  python -m training.train_gt_detr_lmmse \
    --arch "$arch" --bits-per-symbol 4 --snr-db 8 \
    --steps 1 --re-per-step 8 --val-channels 1 --val-re-per-channel 8 --val-every 1 \
    --seed 20260918 --output-dir "ckp/icc_audit_smoke/$arch" \
    --history "results/icc_audit_smoke/${arch}_history.json"
done
```

This checks the exact EP initialization assertion, forward/backward and checkpoint save; it is not formal retraining.
For a real-data pairing check, run the operating-point smoke again with the same seed and a new output directory,
and compare `results` in the two JSON files. Do not infer significance from smoke BER/BLER.
The CE sanity command defaults to a larger sample and has statistical performance assertions;
a one-channel BER improvement is not a reliable pass/fail criterion.

## Results and repeatability

BER requires explicit GT/DETR checkpoint paths. Its default result name uses runtime modulation,
SNR, requested channel/RE counts and seed; JSON also records actual RE count.
Evaluators reject pre-existing outputs unless `--overwrite` is supplied.
Training rejects pre-existing `best.pth` or history unless `--fresh` is supplied.
`--fresh` overwrites only those two artifacts, leaving other files in the output directory intact; it does not resume training.

Coded BLER defaults to T1/MCS11. Other MCS must be explicitly selected. Missing mappings fail before sampling.
The built-in map also contains T2/MCS5; both use 16QAM but distinct specialist checkpoints.
To extend it, pass `--checkpoint-map path/to/map.json` with entries such as:

```json
{"1:19": {"gt": "ckp/your_verified_64qam_gt/best.pth", "detr": "ckp/your_verified_64qam_detr/best.pth"}}
```

Those example paths are placeholders, not provided weights. Map by `(table,mcs)`, never only by Qm.
New specialist training records the table and index in the checkpoint and validates their modulation.
`run_all_mcs_specialists.sh` expects a server-side plan under `results/mcs_specialists/mcs_train_plan.csv`.
Required columns: `table,mcs,qm,snr_min_db,snr_max_db,val_snr_db`; optional `enabled`.
Archive that plan with the run. The repository does not invent unverified 64QAM operating regions.

Formal bracketing extends the missing side by `--bracket-step-db` (default .5dB), adding at most
`--max-bracket-extensions` total SNR points (default 8). Every added point evaluates all detectors,
with the same formal channel budget and common seed. An unresolved crossing is JSON `null`,
status `unresolved`, and a warning. Interpolation stays within measured support, including after zero-count flooring.
`--skip-formal` is a coarse diagnostic and does not trigger additional simulations.

`mcs_bler_results.json` is atomically saved after every completed SNR point; an interrupted MCS retains
status `running` and its completed points. Automatic resume is not implemented.
The JSON includes CLI/system configuration, Git commit/dirty state, versions, cache and checkpoint hashes, effective seed,
block/bit counts, channel-level counts, crossings and warnings. CSV summaries are derived after a completed MCS.
Use channel-level paired resampling for uncertainty; 16 blocks from one channel are not 16 independent channels.

All historical gains in the handoff remain user-supplied results pending artifact verification.
Preserve their exact checkpoints, cache hash, seeds and settings when locking the formal 16QAM anchor.
Next priorities are 64QAM BER, 64QAM BLER, QPSK verification, complexity, then compact mismatch/oracle diagnostics.

## Git workflow

Work on `codex/icc-audit-20260918`, review and merge on the GPU server, smoke, then formal evaluation.
No force push or merge into main from the laptop. Checkpoints, caches, generated results/logs,
partner code and partner weights remain local and ignored by Git.
