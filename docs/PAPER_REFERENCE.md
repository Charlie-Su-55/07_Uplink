# Mode B paper reference

The Level-0 all-data reference remains available. Mode B is a separate, fixed input
distribution implemented by the same `SionnaReferenceLink`, selected by
`mode: paper_reference` in `configs/system/uma_16ue_256rx_paper_reference.yaml`.
No neural training or architecture changes are part of this stage.

## Frozen waveform and energy accounting

| Quantity | Mode B value |
|---|---|
| UEs / streams | 16 transmitters, one stream each |
| Physical antennas | 256 Rx; four Tx elements per UE |
| Mapping | Unit-norm equal-gain rank-one mapping |
| Channel | Normalized UMa, same topology distribution as Level-0 |
| Power control / interference | None / none |
| OFDM | 14 symbols, FFT 192, 30 kHz SCS, CP 14 |
| Pilot symbols (zero-based) | 2 and 11 |
| Pilot pattern | Native Sionna orthogonal Kronecker QPSK, fixed seed 314159 |
| Data symbols | 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13 |
| Data RE / UE | 2304 |
| MCS | Table 1, index 10: Qm=4, target rate 340/1024 |
| G / payload TB per UE | 9216 coded bits / 3104 information bits |
| Payload rate used for Eb/N0 | 3104/9216 |

This is an actual transmitted pilot waveform with native NR TB coding. The pilot
pattern is an orthogonal research reference, **not a claim of full NR PUSCH DMRS
port/CDM compliance for 16 UEs**. No legacy CE-only resource grid is involved.
The exact same native ResourceGrid object is passed to ResourceGridMapper,
GenerateOFDMChannel, LSChannelEstimator and LMMSEEqualizer.

There are 384 reserved pilot REs per UE, of which 24 carry nonzero pilots. UE k
uses subcarriers `k, k+16, ...` on the two pilot symbols. Native pilot normalization
gives each nonzero pilot energy 16; averaging over all 384 reserved positions,
including the zeros, gives energy 1. This keeps average transmit energy per UE
equal to 1 across both data and pilot symbols. There is no additional pilot boost
or `1/sqrt(K)` factor in our code. The transmitted pilot SHA256 is saved.

For this actual grid, the native `ebnodb2no()` result is:

```text
N0 = [(14/12) * (1 + 14/192)] / [10^(EbNo_dB/10) * 4 * (3104/9216)]
```

Pilot and CP overhead are included once. The code checks the frozen G/TB against
the native encoder. CSI mode, detector selection and Eb/N0 cannot change the
pilot sequence, data mask, coded length or TB size. The distribution fingerprint
excludes CSI mode, Monte Carlo seed and device but includes physical/grid/codec
configuration. Pilot SHA256 additionally records the actual generated sequence.

## Channel estimation and independent calibration

`--csi perfect` still sends the real pilots. It uses true effective H with zero
`err_var`. `--csi practical` uses only received y, known pilots, N0 and a frozen
independent covariance prior: native LSChannelEstimator followed by native
LMMSEInterpolator in frequency-then-time (`f-t`) order, without spatial smoothing.
The returned `h_hat` and `err_var` are both passed to native LMMSEEqualizer.

`tools.build_sionna_reference_covariance` estimates frequency and time uncentered
second moments of the **effective** channel on separate UMa realizations. It pools
UEs/Rx antennas/time or frequency, preserves the actual projected channel power,
and applies trace-preserving 0.001 diagonal shrinkage. It does not normalize the
effective channel again. This is a separable pooled prior; `err_var` is a model
prediction, not an empirical or exact guarantee of estimation MSE. Diagnostics
save both predicted uncertainty and actual estimation error against simulator truth.

The artifact has a distinct schema, distribution hash, calibration seeds, version
and Git provenance. Legacy/mismatched caches and overlapping evaluation seeds are
rejected. Evaluation never fits the prior from its own H_true. Keep the same prior
and its SHA256 for comparisons. More calibration channels can assess prior
convergence; changing the prior is a new receiver configuration, not a new grid.

Native interpolation is evaluated in chunks of four Rx antennas to bound its
temporary matrices. These chunks share the same realization and prior; no spatial
smoothing is requested. The equalizer still sees the full 256Rx system.

## Adapter and checkpoint policy

`link_level/detector_adapter.py` exposes:

| Field | Layout / meaning |
|---|---|
| `y` | `[B,T,F,M]`, full actual received grid |
| `h_true`, `h_hat`, `err_var` | `[B,T,F,M,K]`; err_var is real and nonnegative |
| `n0` | Scalar complex AWGN variance |
| `Ruu` | `[B,M,M]`, exactly `N0 * I` |
| `bits`, `coded_bits` | `[B,K,k]`, `[B,K,G]` |
| `data_indices` | Common flattened grid positions in native mapper order |
| `metadata` | CSI mode, seed, Eb/N0, grid, codec, distribution and checkpoint policy |

Each batch calls `transmit()` exactly once. CSI estimates are cached on that
sample. All requested receivers use that same y/Hhat/N0/grid/codeword. Pilots are
excluded from custom detection with native grid data indices, and each returned
LLR is restored to `[B,K,1,G]` before the shared TB decoder.

The first adapters enable only native LMMSE, existing custom LMMSE, and existing
EP5. Practical custom detection fails unless
`--custom-ce-policy sionna_diagonal` is explicit. That optional validation policy
matches Sionna's diagonal CE approximation on unit-energy data REs:

```text
variance[m,RE] = N0 + sum_k err_var[m,k,RE]
y_white = y / sqrt(variance)
H_white = Hhat / sqrt(variance)
```

The existing custom core receives these prewhitened values with identity noise
covariance. EP5 uses the resulting existing frontend z/Gram statistics. The
adapter's exported Ruu remains `N0 I`; err_var remains a separate field. There is
no hidden CE drop or second addition of N0. This policy is a named classical
comparison baseline, not a decision about future GT/DETR uncertainty inputs.

All existing checkpoints are classified as **legacy-distribution** by the Mode B
config and output metadata; their files are preserved unchanged. This entrypoint
has no checkpoint-loading or training path. GT/DETR names are rejected pending
classical validation; Flow is excluded. No files under `models/graph/`, DETR or
Flow architecture files are modified.

## Server commands

Run from the repository root in the existing `precoder` environment (Sionna 2.0.1).
First run `python -m unittest discover -s tests -v`; all native Sionna tests must
execute on the target environment rather than be skipped for a missing/old version.
These tests use small synthetic channels, never the full UMa simulation.

1. Mode B perfect-CSI T1/MCS10 smoke (no covariance artifact needed):

```bash
python -m evaluation.evaluate_paper_reference --csi perfect --table 1 --mcs 10 --channels 2 --batch-size 1 --seed 20260925 --ebno-dbs=-22,-18,-14 --output results/paper_reference/perfect_smoke.json
```

2. Mode B perfect-CSI 100-channel waterfall:

```bash
python -m evaluation.evaluate_paper_reference --csi perfect --table 1 --mcs 10 --channels 100 --batch-size 1 --seed 20260925 --ebno-dbs=-24,-22,-20,-18,-16,-14,-12,-10,-8 --output results/paper_reference/perfect_100.json
```

Before any practical run, build the independent prior **once**:

```bash
python -m tools.build_sionna_reference_covariance --channels 200 --batch-size 1 --seed 4000000 --output data/cache/paper_reference_ft_cov.pt
```

3. Mode B practical-CSI 100-channel waterfall:

```bash
python -m evaluation.evaluate_paper_reference --csi practical --table 1 --mcs 10 --channels 100 --batch-size 1 --seed 20260925 --ce-covariance data/cache/paper_reference_ft_cov.pt --ebno-dbs=-24,-22,-20,-18,-16,-14,-12,-10,-8 --output results/paper_reference/practical_100.json
```

Practical runs automatically also save the same-realization perfect-CSI control.
`--csi both` is an explicit alias for this paired evaluation. Matching seed, batch
size and grid also make the perfect curve in (3) comparable to (2).

4. Same-realization comparison of native LMMSE / custom LMMSE / EP5, for both CSI modes:

```bash
python -m evaluation.evaluate_paper_reference --csi both --channels 2 --batch-size 1 --seed 20260925 --ebno-dbs=-22,-18,-14 --ce-covariance data/cache/paper_reference_ft_cov.pt --detectors sionna_lmmse,custom_lmmse,ep5 --custom-ce-policy sionna_diagonal --re-chunk 128 --output results/paper_reference/classical_paired.json
```

5. Practical H NMSE / err_var diagnostic, including paired perfect/practical BLER:

```bash
python -m evaluation.evaluate_paper_reference --csi practical --channels 10 --batch-size 1 --seed 20260925 --ebno-dbs=-18,-12,-6,0 --ce-covariance data/cache/paper_reference_ft_cov.pt --output results/paper_reference/ce_diagnostic.json
```

Start with batch size 1 on the GPU server. Output files are protected from silent
overwrite and saved atomically after each batch; failed/interrupted runs retain
their partial counts. Use new output names for reruns, or explicit `--overwrite`.
No resume is implemented and no command trains a model or deletes checkpoints.

## Reading results

- `reference.grid`, `reference.codec` and `distribution_id` identify the fixed
  waveform. `reference.covariance` records the practical prior and file SHA256.
- `points[].receivers["perfect/sionna_lmmse"]` and
  `["practical/sionna_lmmse"]` contain exact BLER/BER, CRC and per-channel/per-UE counts.
- `points[].diagnostics[].csi.practical` contains full-grid/data-only H NMSE,
  per-channel/UE data NMSE, mean/median err_var, and measured data H MSE.
  Medians are per batch, using the lower middle value for even sample counts.
- `lmmse_comparison` reports relative RMS differences for soft symbols, effective
  noise and LLRs. For the explicit matching CE policy these should remain small;
  EP5 need not produce the same LLRs or outperform LMMSE at every point.
- `first_batch_power_diagnostic` checks actual waveform reconstruction and power.
  `status: complete` means execution completed. `screening` is the inherited
  conservative statistical report; 100-channel curves can remain `unresolved`.
  Curves are not smoothed and these runs do not automatically authorize neural work.

Mode B pilots change energy overhead and TB length relative to Level-0, so its
waterfall need not cross at Level-0's Eb/N0. The proposed scan range is a starting
range; widen it if necessary while holding the grid/TB/seed/prior fixed.

API/accounting references pinned to Sionna 2.0.1:
[pilot pattern](https://github.com/NVlabs/sionna/blob/v2.0.1/src/sionna/phy/ofdm/pilot_pattern.py),
[LS/LMMSE interpolation](https://github.com/NVlabs/sionna/blob/v2.0.1/src/sionna/phy/ofdm/channel_estimation.py),
[Eb/N0 conversion](https://github.com/NVlabs/sionna/blob/v2.0.1/src/sionna/phy/utils/misc.py),
[NR TB calculation](https://github.com/NVlabs/sionna/blob/v2.0.1/src/sionna/phy/nr/utils.py).
