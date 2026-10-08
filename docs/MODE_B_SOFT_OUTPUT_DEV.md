# Mode-B soft-output development

This batch compares `sionna_lmmse`, `ep5`, `gt_best`, `gt_last`,
`detr_best`, and `detr_last` on new practical-CSI channels. It caches the
complete receiver outputs once, fits positive LLR scales using only the
calibration split, and evaluates full transport blocks on the development
split. Neither split is the final independent paper test set. Temperature
scaling is a calibration baseline, not a claimed new paper contribution.

No training or A1 inference runs. The existing PHY, pilot grid, CE estimator,
covariance builder, Eb/N0 conversion, codec, receiver mathematics, architectures,
trainers, comparison scripts, archives, weights, and historical results remain
unchanged. Neural loading reuses the existing strict loader in the A1
integration module; the external A1 bundle is not read, imported, or
instantiated.

## Audited sources and provenance

The inventory was checked against the actual files, not checkpoint filenames:

- `docs/experiment_snapshots/mode_b_v1/README.md`, `snapshot_index.json`,
  `training_overview.json`, all three `training_history/*.json`, and
  `a1_comparison_overview.json`; complete archived evaluation JSON files supply
  the evaluation-channel seed exclusions.
- `configs/system/uma_16ue_256rx_paper_reference.yaml`.
- `training/train_paper_reference_detector.py`.
- `link_level/sionna_reference.py`, `sionna_ce.py`, `sionna_ebno.py`,
  `nr_codec.py`, `detector_adapter.py`, and `partner_a1_adapter.py`.
- `detectors/classical/ep.py`, `evaluation/evaluate_paper_reference.py`, and
  `evaluation/compare_paper_reference_a1.py`.

These identities describe different events and must not be conflated:

| Event | Commit |
| --- | --- |
| Frozen `paper-reference-classical-v1` tag / archived CE prior generation | `f7741e53a9cad31cf62eda44b4607dfd1e3c9147` |
| Archived GT/DETR training runs | `52c367d8f45677cdc42861ff3f270f8395ae0323` |
| Archived 100-channel A1 comparison run | `4700b5dac6e84794e3615c24cc8ee8f0616f3926` |
| Snapshot collection code, `snapshot_index.snapshot_code_commit` | `fff7e4aeed69343ace2272a8d9df8bad12d84d00` |
| Repository HEAD at the start of this development task | `eaf735bf30974cf05371a9ca49386012a863cb3f` |

The branch is `refactor/sionna-standard-ebno`. New manifests record their own
runtime commit, branch, dirty state, source/config hashes and numerical
environment. They do not relabel historical runs with the current commit.
Snapshot text files may have CRLF in a Windows checkout; indexed archive hashes
describe the original server bytes. Binary checkpoint/prior SHA256 comparisons
are exact. The archived JSON source copies preserve parsed values even where
formatting whitespace was compacted.

## Frozen configuration and entrypoint overrides

| Parameter | Frozen YAML / source default | Actual batch behavior and evidence |
| --- | --- | --- |
| Mode | `paper_reference` | Same frozen Mode-B distribution |
| CSI | YAML `perfect` | Entrypoint overrides to `practical`; same override as archived training/comparison |
| Execution seed | YAML `20260925`; trainer default `42`; comparison `20260929` | Explicit independent per-channel split lists; a base seed is not a training-channel list |
| Carrier / OFDM | 6.7 GHz; 30 kHz SCS; 192 subcarriers; 14 symbols | Unchanged |
| UE / stream / receive antennas | 16 UEs, 1 stream/UE, 256 Rx | Unchanged |
| Physical UE array / stream mapping | 4 Tx antennas/UE, rank-one equal-gain unit-norm mapping | Unchanged; no normalization after channel projection |
| BS array | 8 rows x 16 columns, dual cross polarization | 256 physical receive elements |
| UT array | 2 rows x 1 column, dual cross polarization | 4 physical transmit elements |
| Channel | Normalized UMa, forced NLOS, low O2I, indoor probability 0.5 | Frozen topology; max outdoor/indoor speeds 30/3 km/h; pathloss and shadow fading enabled before channel normalization |
| Power / interference | No fractional power control; external interference disabled | Unit expected symbol energy per UE; aggregate energy 16 |
| CP | 14 samples | Included by native `ebnodb2no` on the actual transmitted ResourceGrid |
| Pilots | Kronecker, OFDM symbols `[2, 11]`, seed `314159` | Physically transmitted; 384 reserved RE/UE, 24 nonzero pilots/UE, active pilot energy 16 |
| Pilot identity | Recorded by the actual ResourceGrid | SHA256 `971ba250926c2e955e9e106703e53e9570b5840c4a642d84650e2ab2dbcd0e3a` in the archive |
| Data symbols | `[0,1,3,4,5,6,7,8,9,10,12,13]` | All 2304 data RE/UE; no RE subsampling |
| MCS / modulation | Table 1, MCS 10, Qm=4 | 16-QAM, original mapper bit positions |
| Nominal code rate | `340/1024 = 0.33203125` | Used by native TB size selection |
| Payload / coded length | `k=3104`, `G=9216`, TB size 3104, padding 0 | Eb/N0 uses actual `k/G = 0.3368055555555556`, not the nominal rate |
| Codec | Native NR TBEncoder / TBDecoder | PUSCH scrambling enabled; RNTI `[1,...,16]`, n_id `[1,...,1]`, one layer, 20 BP iterations |
| LLR | `log(P(bit=1)/P(bit=0))` | Hard decision `LLR > 0`; no bias or sign change |
| CE | Native LS + LMMSE interpolation `f-t`, Rx chunk 4 | Existing sample CSI cache; native `err_var` retained |
| CE prior | `data/cache/paper_reference_ft_cov.pt` | Existing independent 200-channel prior, seeds 4000000 through 4000199, shrinkage 0.001; no rebuild |
| Raw covariance | `Ruu = N0 I` | Thermal noise only; CE uncertainty remains separate |
| EP/neural CE policy | `sionna_diagonal` | Divide y/Hhat by `sqrt(N0 + sum_UE err_var)` before the shared frontend; effective whitened noise identity |
| EP5 | 5 iterations, damping 0.5, minimum variance/site precision 1e-6 | Existing EP implementation, original posterior mean and variance |
| Numerical precision | `single` | Stored real tensors float32 and complex tensors complex64; calibration reductions/optimization float64 |

The frozen distribution ID is
`c3732839f5cbfbfd9c4818bd36404c7d47a597e163a9d46b4ced18b6bfdfa25f`.
The CE prior SHA256 from `snapshot_index.local_only_artifacts` is
`76a11f88598eef9a22033d221dda9f00d415ece6b046af78d0c8de037601fbc4`.
The frozen YAML's archived LF-byte SHA256 is
`d6a9dd63b9b2cb0b5acf1127ba90d3e882119e1e7ce834d12483a371ab761803`.

This experiment does not use the legacy `rx_snr` platform or its CE-only grid.
No measured receive power determines N0. The actual N0 for every channel is
returned by the frozen `ebnodb2no` path and stored in the cache; complex AWGN
satisfies `E[|n|^2]=N0`, with real/imaginary component variance `N0/2`.
Archived grid accounting gives CP factor `1.0729166666666667` and grid energy
factor `1.2517361111111112`; the batch calls the native conversion rather than
reimplementing these factors.

## Checkpoints and historical training

| Alias | Server path | SHA256 in snapshot index | Archived step evidence |
| --- | --- | --- | --- |
| `gt_best` | `ckp/paper_reference_gt_ep_smoke300/best.pth` | `e762975f2a6ecf09f5f4143dd404dd23a45f433375934343e5d5a70962311cef` | Actual best metadata in comparison: 150 |
| `gt_last` | `ckp/paper_reference_gt_ep_smoke300/last.pth` | `b40cdb3b1a3ed570074a4f2641300eded3b0c8dd21525650e9b284c247ded70f` | History ends at 300; actual saved step read on server |
| `detr_best` | `ckp/paper_reference_detr_ep_smoke300/best.pth` | `244fb6cf492bd052f5eac057122dd993029affca4923d78d67b1fc411ffb8d15` | Actual best metadata in comparison: 275 |
| `detr_last` | `ckp/paper_reference_detr_ep_smoke300/last.pth` | `08e7b815871222c09540b5049d5f3ecb2a7fec148050be87542a0fed811be1ba` | History ends at 300; actual saved step read on server |

Both completed histories use 300 planned training channels, uniform sampled
Eb/N0 in [-13,-9] dB, 128 RE per step, learning rate 5e-5, weight decay 1e-4,
gradient clipping 1, and seed 42. Actual validation uses 8 channels x 64 RE
at -10.5 dB every 25 steps; the trainer CLI default is 32 validation channels.
`best` was selected by pre-decoder coded-bit BER on that small validation set,
not by TB BLER. GT best/last logged BER is 0.112518310546875 /
0.115142822265625; DETR best/last is 0.112274169921875 /
0.11260986328125. New development results never overwrite `best.pth`.

The lr2e5 history has status `running`, with records only at steps 0 and 25.
It declares all 300 planned training seeds, which are excluded conservatively.
Its best logged step is 25 and BER 0.113250732421875. It is an incomplete
historical experiment, excluded from the default comparison and never resumed.

The archived comparison used 100 channels per Eb/N0 and full 2304-RE
codewords. For reference, its raw GT/DETR BLER at -12, -10.5, -9 dB was
respectively `(0.143125,0.145)`, `(0.0875,0.0875)`, and
`(0.051875,0.053125)`. Those are historical metrics on different seeds and
are not targets or acceptance thresholds for this batch.

The server checks each of the four weight file hashes against the index,
then calls the existing strict Mode-B loader. Its checks cover kind,
architecture, MCS, CSI policy, prior hash, distribution ID, system config,
LLR convention, training/validation seeds, and a valid actual checkpoint
`step`. All four actual steps and hashes are recorded at runtime. Server-only
weights and covariance were not opened or fabricated on Windows.

## Independent experiment configuration and seeds

`configs/evaluation/paper_soft_output_dev.yaml` is JSON-formatted valid YAML,
so lightweight configuration tests do not need PyYAML. The frozen PHY YAML
is left unchanged. Default development settings are:

- Calibration: 32 independent channel seeds, candidate start 81000000.
- Development: 64 independent channel seeds, candidate start 82000000.
- Eb/N0: `[-12,-10.5,-9]` dB; batch size 1; chunk size 128 RE.
- CUDA device `cuda:0`, float32/complex64, two Torch CPU threads,
  deterministic algorithms enabled, TF32 disabled.
- One positive global alpha, or four positive mapper-bit-position alphas,
  fitted separately for every receiver.
- Bounded derivative bisection for coded-bit BCE in float64, alpha range
  `[0.05,8.0]`, alpha tolerance 1e-5, gradient tolerance 1e-7, maximum
  40 iterations. Boundary solutions are reported; development cannot widen
  the interval or select another temperature.
- Paired bootstrap: 2000 replicates, seed 83000000, confidence 0.95.
- Residual comparison tolerances: absolute 1e-5 and relative 5e-6;
  CE feature denominator epsilon 1e-12.

All archived JSON is inspected for training, validation, CE-calibration and
evaluation seed records. The initial inventory finds 300 unique training
seeds, 8 validation seeds, 200 CE-calibration seeds, and 104 historical
evaluation seeds (20260925 through 20261028). Conservatively including base
seed 42 gives 613 distinct exclusions. All three training histories declare
the same 300 planned training seeds. This check uses the lists themselves,
not just the base seed.

The checked candidate ranges do not intersect any exclusion or each other.
With the current archive the final lists are 81000000 through 81000031 and
82000000 through 82000063. The generated manifest records the full explicit
lists, exclusion values and sources; selection is rechecked on the server.
The rescue/harm-selected eight replay cases are historical exclusions and
never calibration data.

The same channel seeds are paired across the three Eb/N0 points. There are
32 and 64 independent channel identities, not 96 and 192 independent channels.
Cross-Eb/N0 summaries/bootstrap retain each seed as one cluster. Per-point
results retain 32 or 64 channels. The two-channel smoke uses one channel from
each split at -10.5 dB; smoke fits only exercise the pipeline and have no
statistical interpretation.

## Cache and inference features

For each `(split,Eb/N0,seed)`, the builder calls the existing `transmit` once,
performs practical CE once using the existing CSI cache, and constructs the
CE-aware frontend once per chunk. EP and all four neural checkpoints share
that chunk's z/G. Native Sionna LMMSE receives the same observation, Hhat,
err_var, N0, ResourceGrid and transmitted bits. No model resamples a channel.

Each channel file contains the complete coded bits and payload bits in their
original codeword order, all six full LLRs and their exact decoder inputs,
raw per-UE payload errors/block errors/CRC outcomes, data indices and ordinals,
actual N0, CE-whitened z/G, EP5 posterior mean/variance, received energy,
residual and per-UE CE feature. It stores tensors on CPU, preserving
float32/complex64, with explicit shapes/axes, provenance and file SHA256.
Large y/Hhat/h_true tensors and A1 candidates are not saved.

For whitened observations, `z=H_w^H y_w`, `G=H_w^H H_w` and
`s=||y_w||^2`. The EP posterior feature is

```text
r = (s - 2 Re(mu^H z) + Re(mu^H G mu) + sum_k G_kk nu_k) / 256
eta_k = sum_m err_var[m,k] /
        (sum_m |Hhat[m,k]|^2 + sum_m err_var[m,k] + 1e-12)
```

Here mu/nu come from EP5, not transmitted symbols. The residual is checked
against direct `||y_w-H_w mu||^2 + sum_mk |H_w[m,k]|^2 nu_k` calculations
with the configured tolerances. A materially negative residual raises an
error. eta is a candidate feature, not an exact posterior error probability.
Neither feature uses true H, true noise or labels; labels enter only loss,
metrics and consistency checks.

The evaluator representation `[B,Ndata,UE,Qm]` converts through the existing
`llrs_to_native_order` to `[B,UE,1,G]`; the codec removes only the singleton
stream axis. Flattening follows the original data-RE order with mapper bit
position as the innermost axis. Global scaling uses one alpha; per-bit scaling
broadcasts along the Qm axis, never the UE axis.

Files are written to temporary paths and atomically replaced. A channel is
complete only after its data, hash and reload/decode checks succeed. Explicit
reuse requires matching identity/schema/hash and a complete record. Interrupted
or mismatched artifacts cannot silently become completed cache entries.

## Calibration, decode and reporting

Fitting uses only calibration cache LLRs/coded bits, pooling all calibration
Eb/N0 points with the predeclared optimizer. Cache integrity validation may
read development channel files to verify tensor schemas and hashes, but their
labels never enter the fitting function or its optimization decisions.
Evaluation reads the fixed alphas
and development cache; it creates no UMa channels and executes no GT/DETR/A1
models. It uses the unchanged NR TB decoder on complete scaled codewords.

Raw, global-scale and per-bit-scale results include coded BER, BCE in nats,
the fixed-scale GMI proxy, full-TB BLER, payload BER, CRC, per-channel/per-UE
decoded outcomes, and calibrated-versus-raw rescue/harm counts. Alpha=1 must
reproduce raw LLRs, payload bits and CRC. Positive scales must preserve hard
bit decisions. GMI is a fixed-scale proxy, not an independently optimized GMI
claim; lower BCE does not guarantee lower BLER.

Paired channel bootstrap comparisons include each calibrated receiver against
its own raw output and best/last checkpoint contrasts. Small-sample intervals
are descriptive; UE blocks and repeated Eb/N0 observations are not independent
bootstrap samples. Smoke intervals have especially limited meaning.

## Server commands and completion

Run from the repository root in the existing Sionna 2.0.1 environment.
The archived environment is Python 3.12.13, Torch 2.11.0, NumPy 2.4.4,
PyYAML 6.0.3, CUDA 13.0, with an RTX 5090. These are historical facts, not
performance predictions. Preflight reports actual package/GPU/runtime details
and any differences. No dependency installation or upgrade occurs. Matching
versions alone do not establish bitwise reproduction of historical outputs.

One foreground smoke command:

```bash
bash evaluation/run_paper_soft_output_batch.sh smoke --run-id softdev_20261008_smoke1
```

One background development command (use a new run ID for a new experiment):

```bash
mkdir -p logs/paper_soft_output_dev && nohup bash evaluation/run_paper_soft_output_batch.sh development --run-id softdev_20261008_dev1 >> logs/paper_soft_output_dev/softdev_20261008_dev1.launch.log 2>&1 < /dev/null &
```

The append-only launch log captures startup/path/overwrite errors before the
runner creates its own log. It lives outside the per-run log directory, so
launching does not trigger the runner's existing-directory protection.

Development automatically runs preflight, lightweight tests, a separate
`softdev_20261008_dev1_smoke` two-channel full-codeword smoke and its
consistency checks, both full caches, calibration fitting, full BLER
evaluation and summary. Any smoke or subsequent stage failure stops the
pipeline. The runner uses `set -euo pipefail`, preserves errors through tee,
tracks PID/stage/exit status, prints per-stage/per-channel progress and elapsed
time, and never terminates other GPU jobs.

View progress and final status:

```bash
tail -n 80 logs/paper_soft_output_dev/softdev_20261008_dev1.launch.log
tail -f logs/paper_soft_output_dev/softdev_20261008_dev1/runner.log
cat logs/paper_soft_output_dev/softdev_20261008_dev1/runner.pid
cat logs/paper_soft_output_dev/softdev_20261008_dev1/runner_status.txt
cat logs/paper_soft_output_dev/softdev_20261008_dev1/stages.tsv
python -m json.tool results/paper_soft_output_dev/softdev_20261008_dev1/summary.json
python -c 'import json; s=json.load(open("results/paper_soft_output_dev/softdev_20261008_dev1/summary.json")); assert s["status"] == "complete" and s["all_raw_redecode_checks_passed"] and s["all_positive_scale_hard_decisions_unchanged"]'
```

Runner status must report `state=complete`, `stage=summary`, and `exit_code=0`,
with both summary consistency flags true. Failures report `state=failed`, the
failing stage and its exit code. Terminal success and a complete summary are
both required; an existing PID or cache directory does not mean success. Existing output
directories are rejected by default. Add `--reuse-cache` explicitly to reuse
validated completed artifacts with identical identities. It also enables
validated evaluator result reuse, never an implicit overwrite.

The outputs for each run ID are under:

| Root | Contents |
| --- | --- |
| `data/cache/paper_soft_output_dev/<run-id>/` | Cache manifest, separate per-channel files, explicit split seeds, identity/schema, file hashes, status, bytes and timing |
| `results/paper_soft_output_calibration/<run-id>/temperatures.json` | Fitted alpha values, optimizer records/boundary flags, `fit_sources` hashes and `fit_identity` |
| `results/paper_soft_output_dev/<run-id>/summary.json` and `summary.csv` | Per-method/variant/Eb/N0 metrics, pooled metrics, calibrated-versus-raw and best-versus-last paired comparisons |
| `results/paper_soft_output_dev/<run-id>/channels.jsonl` | Per-channel/per-UE counts and decoded artifact links |
| `results/paper_soft_output_dev/<run-id>/decoded_channels/` | Exact CPU decoded payload bits, CRC and per-UE outcomes for each method/variant |
| `logs/paper_soft_output_dev/<run-id>/` | Runner log, PID, stage records and exit status |

Summary `metrics` records include `method`, `variant`, `ebno_db` (null for
pooled points), `coded_ber`, `bce_nats`, `fixed_scale_gmi_proxy`, `bler`,
`payload_ber`, CRC counts, `rescue_vs_raw`, `harm_vs_raw`,
`independent_channels` and `channel_ebno_observations`.
`calibration_vs_raw` and `best_vs_last` contain paired bootstrap comparisons;
the latter reports last minus best. Artifact hashes link the detailed outputs.
The per-file byte counts and elapsed times come from the actual run; no GPU
runtime or storage measurement was made on the Windows laptop.

Individual entrypoints are available for inspection and controlled reuse:

```bash
python -m evaluation.build_paper_soft_cache --help
python -m evaluation.evaluate_paper_soft_output --help
```

## Local validation limits

Windows validation is limited to lightweight unit tests, syntax and CLI help.
Tests needing native Sionna 2.x explicitly skip in the laptop's older Sionna
environment; such skips are not GPU/native-link acceptance. No local 256Rx
UMa run, checkpoint/covariance load, training, or synthetic replacement artifact
is required. Server smoke must succeed before any development result is used.

The checked source/archives establish the frozen settings, expected binary
hashes, best checkpoint steps and historical seed exclusions. Actual last
checkpoint steps, current server artifact existence/hashes, runtime device and
versions, complete cache/redecode consistency, calibration outcomes, BLER,
storage usage and elapsed times remain runtime checks until the server batch
is executed.
