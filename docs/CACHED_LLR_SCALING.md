# Cached Mode-B conditional LLR scaling

This batch tests whether small conditional positive scales improve EP5 soft outputs using an existing, immutable cache. It trains new scale readouts, then re-decodes complete cached codewords with the original TB decoder. It does not regenerate channels, repeat GT/DETR inference, or run A1. A missing, partial, mismatched or corrupt source cache stops the batch.

## Evidence and fixed protocol

The source is `data/cache/paper_soft_output_dev/softdev_20261008_dev1/`, with the read-only archive in `docs/experiment_snapshots/softdev_20261008_dev1/`. The archived cache identity is `c1a4f87a0ead529adf0435ba7e8ab72b4d3bbf06445441a65466acfa1ddd33bf`; its manifest SHA256 is `4b6fde85c44644cb5d56f8139cdfe42094cc94068033f1507e3a5eddaa590371`. These identify the historical data. The new experiment records its own source-code/configuration/training identities separately.

Archived `summary.csv` reports these aggregate development counts across 64 independent channel seeds, three Eb/N0 points and 3072 TBs:

| Frozen receiver | Raw block errors | Global scale, old 32 calibration channels | Per-bit scale, old 32 calibration channels |
|---|---:|---:|---:|
| EP5 | 285 | 267 | 265 |
| GT best | 266 | 257 | 256 |
| GT last | 274 | 275 | 272 |
| DETR best | 270 | 254 | 251 |

The GT-last global result illustrates that lower BCE need not imply lower BLER. These historical constants and metrics are labelled as `archived_32cal` references, not used to initialize or select new scales. New `global_train24` and `per_bit_train24` EP5 constants are fitted on the same 24 channels used by the conditional readouts.

`configs/evaluation/cached_llr_scaling.yaml` fixes the following before evaluation:

| Item | Value |
|---|---|
| Scale-training channels | First 24 sorted old calibration seeds: `81000000`–`81000023` |
| Validation channels | Remaining 8 old calibration seeds: `81000024`–`81000031` |
| Development channels | `82000000`–`82000063`, evaluation only |
| Eb/N0 | `[-12, -10.5, -9]` dB for every channel |
| Payload/grid | Cached Mode-B: 16 UEs, 256 Rx, Qm=4, 2304 data RE, G=9216, k=3104 |
| Receiver front end | Cached practical CSI with the original CE uncertainty treatment |
| New training seeds | `42, 43, 44` |
| Optimization | 1000 AdamW steps, 256 RE/step, lr `1e-3`, weight decay `1e-4`, gradient clip `1.0` |
| Selection | Full validation coded-bit BCE every 100 steps; retain best and last, including the initial alpha=1 reference |
| Scale bounds | `[0.05, 8.0]`, fixed for every model, bit, seed and Eb/N0 |
| Runtime | Existing CUDA environment, Sionna `2.0.1`, archived torch `2.11.0`, float32/complex64, deterministic operations, TF32 disabled |
| Bootstrap | 2000 channel-cluster resamples, seed `84000000`, confidence `0.95` |

The same channel seed at three Eb/N0 points remains one independent channel. Train, validation and development lists are persisted and checked for overlap. Validation selects checkpoints using BCE only; development labels never fit normalization, constants, scale weights, stopping conditions or hyperparameters. All three seeds' best and last readouts are reported after training; the development set is not a final independent paper test set.

## Inputs and candidates

For each RE and UE, the six shared node features are EP5 posterior mean real/imaginary parts, log posterior variance, log Gram diagonal, log posterior residual and CE feature eta. They come only from cached inference statistics. Normalization uses the 24 training channels; labels are used for BCE and metrics only. Feature preprocessing uses log epsilon `1e-8`, log clipping `[-18,18]`, raw-feature clipping `[-20,20]`, normalization epsilon `1e-6` and standardized clipping `[-8,8]`. These are feature limits; original LLRs are not clipped.

All outputs have shape `[RE, UE, Qm=4]` with weights shared across UE identities. Each candidate outputs log scales and applies `alpha = exp(clamp(log_scale, log(0.05), log(8)))`. A zero final head starts every scale at exactly one. Scaling adds no bias and preserves the `log P1/P0` sign convention and hard decisions.

| Candidate | Input and architecture |
|---|---|
| `affine` | Shared affine map from standardized log residual and eta to four log scales |
| `mlp` | Six node features; two width-32 SiLU hidden layers |
| `graph` | Same six node features; width 32, two message layers, off-diagonal mean aggregation; normalized complex Gram real/imaginary/absolute edge features |
| `mlp_no_residual_eta` | Matched two-layer width-32 MLP with residual and eta removed |

Constants use bounded derivative bisection on training coded-bit BCE in float64, the same `[0.05,8]` range, at most 40 iterations, alpha tolerance `1e-5` and gradient tolerance `1e-7`. Boundary saturation is reported. Evaluation reads cached complete LLRs, applies scales, restores the original decoder-input layout and invokes the original codec; no RE subset substitutes for a full-codeword BLER measurement.

## Server commands

Run from the repository root in the existing server environment. The runner uses its active `python`; the `PYTHON` environment variable can select another existing interpreter. No dependency installation is performed. Do not create fake cache or checkpoint files on Windows.

One smoke command:

```bash
bash evaluation/run_cached_llr_scaling_batch.sh smoke --run-id cachedscale_20261008_smoke1
```

Smoke uses all four candidates, two optimizer steps and training seed 42. It reads channel `81000000` for training and validation channel `81000024` at -10.5 dB, then verifies full-codeword validation decoding. Its normalization and constants fit only that reduced training case. It does not evaluate development channels. The saved protocol keeps the full 24/8/64 split explicit and records the smoke subset separately.

One complete background batch command; the launcher log refuses overwrite:

```bash
mkdir -p logs/cached_llr_scaling && (set -o noclobber; exec nohup bash evaluation/run_cached_llr_scaling_batch.sh development --run-id cachedscale_20261008_dev1 < /dev/null > logs/cached_llr_scaling/cachedscale_20261008_dev1.launcher.log 2>&1) &
```

Development runs preflight, the new lightweight tests, a separate `cachedscale_20261008_dev1_smoke` batch and its completion checks, all full training jobs, development decoding and completion verification. Any failed stage stops subsequent stages. Progress, start time, elapsed time, PID and actual command/tee failure exit codes are retained.

```bash
tail -f logs/cached_llr_scaling/cachedscale_20261008_dev1/runner.log
cat logs/cached_llr_scaling/cachedscale_20261008_dev1/runner_status.txt
cat logs/cached_llr_scaling/cachedscale_20261008_dev1/stages.tsv
python -m evaluation.evaluate_cached_llr_scaling --run-id cachedscale_20261008_dev1 --mode development --check-complete
```

Completion requires `state=complete`, `exit_code=0`, a complete evaluation summary with `all_raw_redecode_checks_passed=true` and `all_positive_scale_hard_decisions_unchanged=true`, and successful `--check-complete`; run that check in the original server environment; it performs no GPU inference or decoding, but still verifies Sionna/runtime identity. The lock prevents two runners using one run ID concurrently. Default execution rejects existing output directories. Explicit `--reuse-complete` validates matching identities and hashes before reusing complete artifacts; it is not permission to resume or overwrite partial training/results. After a failed partial batch, use a new run ID. The source cache remains reusable and unchanged.

## Outputs and interpretation

- `ckp/cached_llr_scaling/<run-id>/manifest.json`: configuration, source-cache/new-code identities, explicit splits, training normalization, new constants and artifact records.
- `ckp/cached_llr_scaling/<run-id>/<candidate>/seed_<seed>/best.pth` and `last.pth`: new scale weights, actual training step and provenance; validation BER and scale statistics are logged alongside the BCE selection metric. Original detector checkpoints remain untouched.
- `results/cached_llr_scaling/<run-id>/summary.json` and `summary.csv`: coded BER/BCE, fixed-scale GMI proxy, full-TB BLER/payload BER/CRC, raw/constant/candidate comparisons, saturation and seed variation.
- `results/cached_llr_scaling/<run-id>/decision_summary.json`: Each candidate versus both train24 constants, Graph versus MLP, MLP versus affine, and MLP versus the feature ablation, including paired BLER/BCE intervals and consistency across training seeds. These are development findings, not automatic paper claims.
- `results/cached_llr_scaling/<run-id>/channels.jsonl` and `decoded_channels/`: per-channel and per-UE decoded outcomes for paired rescue/harm and bootstrap calculations.
- `logs/cached_llr_scaling/<run-id>/`: `runner.log`, `runner.pid`, `runner_status.txt` and `stages.tsv`; the development smoke has separate training/result directories ending in `_smoke`.

Compare conditional readouts to the new train24 constants and feature ablation, and report every training seed and best/last pair. Paired bootstrap resamples channel clusters across all associated Eb/N0 points; its interval describes this small development sample. BCE reduction, individual rescue cases or a single favorable seed are insufficient evidence of a reliable BLER gain. Report saturation and harm as well as rescue; a negative feasibility result is valid.

Create a new local report archive after completion. This command includes manifests, configuration, statistics and logs, excludes all `.pt`/`.pth` weights and decoded tensors, and neither changes the frozen archive nor uploads anything:

```bash
archive="results/cached_llr_scaling/cachedscale_20261008_dev1_reports_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
(set -o noclobber; tar -czf - --exclude='*.pt' --exclude='*.pth' configs/evaluation/cached_llr_scaling.yaml data/cache/paper_soft_output_dev/softdev_20261008_dev1/manifest.json ckp/cached_llr_scaling/cachedscale_20261008_dev1 results/cached_llr_scaling/cachedscale_20261008_dev1 logs/cached_llr_scaling/cachedscale_20261008_dev1 > "$archive")
tar -tzf "$archive"
```

Local validation uses `python -m unittest discover -s tests -p 'test_cached_llr_*.py' -v` or individual `tests.test_cached_llr_*` modules, plus Bash syntax and CLI-help checks. Native codec tests may explicitly skip if Sionna 2.x is unavailable. These CPU tests do not establish server training/BLER results or a GPU runtime estimate.
