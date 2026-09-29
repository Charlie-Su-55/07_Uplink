# Frozen A1 on the Mode-B paper reference

The external package is expected beside this repository:

```text
Projects/
  07_Uplink/
  A1_PARTNER_MINIMAL/
```

Its `MODEL.json` identifies combined / step 6000 / 16QAM, with 4,247,644 model
parameters. The supplied weights and source remain external and unchanged.
`--a1-root` accepts another location. The original bundle must also be present on
the GPU server; pushing this repository alone does not transfer it.

The bundled `00_RUN_ONE_CLICK.py` targets the legacy `UplinkUMADataset`, receive
SNR 8 dB, and uncoded BER. Use the new independent entry point for Mode B:
`python -m evaluation.compare_paper_reference_a1`.

## Scientific comparison

The physical link is the frozen T1/MCS10 Mode B: 16 UE/streams, 256 Rx, four
physical Tx elements per UE, normalized UMa, native pilot symbols 2 and 11,
2304 data REs, G=9216 and payload TB=3104. No power control or external
interference. Practical CSI and covariance are produced by the existing
LS + LMMSE f-t path. The x axis is its existing per-UE payload Eb/N0.

Each channel/Eb/N0 has one `transmit()` and one cached practical estimate. Native
Sionna LMMSE, custom LMMSE, EP5, optional GT/DETR, and both A1 candidate budgets
share this realization and its transmitted coded bits. GT/DETR use the training
entry point's existing CE-whitened z/G statistics. The original classical
evaluator, receiver mathematics, grid, coding and all model architectures are
unchanged.

The default A1 input policy is explicit `sionna_diagonal`:

```text
v[m, RE] = N0 + sum_UE(err_var[m, UE, RE])
A1 inputs = y/sqrt(v), Hhat/sqrt(v), I
```

Thus A1 receives the same external CE treatment as GT/DETR/EP5. Only the three
input tensors reach A1; bits and true H are used for metrics/diagnostics. Its
own LS residual normalization and learned structured covariance remain intact.
This is a receiver containing the external CE front end plus the original A1;
it is not a claim that A1's internal sufficient statistics equal GT/DETR's.
The exported batch's original `Ruu=N0 I` is never changed.

`--a1-ce-policy both` adds a clearly named auxiliary `a1_thermal_*` receiver
using the original y/Hhat/N0I. This auxiliary receiver has no explicit external
CE uncertainty input; it retains A1's internal uncertainty handling. Its
package-native LMMSE control also omits explicit CE uncertainty. It must not be
confused with the common-front-end comparison.

Production inference calls the package's original `portable.runtime.load_model`
and `predict`, with K256, RE64/sample_chunk64, original Direct-RB readout and LLR
clip 20. K64 is the exact prefix snapshot from the same K256 population.
The package's deterministic numeric setup is applied before evaluation to all
receivers, including disabled TF32 and the 70% CUDA allocator cap. Its original
conditional CUDA BF16 proposal arithmetic remains unchanged. Runtime versions
are recorded; the qualified Sionna version is 2.0.1. Do not replace the server's
PyTorch by installing the package requirements wholesale.

**The frozen A1 checkpoint does not certify Mode-B training provenance.** This
experiment is an external frozen-model transfer comparison, not a controlled
comparison of architectures trained on identical data/budgets. A1 training-seed
overlap cannot be established from this artifact. Do not claim otherwise in the
paper. A1 retraining would require the colleague's training implementation and
a separate agreed protocol; this integration neither trains nor selects A1 weights.

GT/DETR loading requires `paper_reference_detector_v1`, matching architecture,
Qm/MCS, distribution hash, covariance SHA256, practical CSI, CE policy and LLR
sign. Evaluation seeds must be disjoint from their training/validation seeds and
the CE calibration seeds. Step-zero checkpoints are explicitly reported as
`untrained_EP_anchor`. Use the actual trained checkpoints for paper results.

## Server commands

Run from `07_Uplink` in the existing validated environment. First:

```bash
python -m unittest discover -s tests -v
```

The external-bundle CPU tests verify the real weights and original sampler on
one synthetic RE. Native Sionna tests must execute on the server. No test
generates UMa or runs model training.

Start with a two-channel, 128-RE **BER-only** execution check. This command needs
only the bundle and the existing covariance prior:

```bash
python -u -m evaluation.compare_paper_reference_a1 \
  --a1-root ../A1_PARTNER_MINIMAL \
  --ce-covariance data/cache/paper_reference_ft_cov.pt \
  --channels 2 --ebno-dbs=-10.5 --re-per-channel 128 \
  --output results/paper_reference/a1_ber_smoke.json
```

To include the neural detectors, append their actual Mode-B checkpoint paths:

```bash
  --gt-checkpoint /path/to/ModeB_gt/best.pth \
  --detr-checkpoint /path/to/ModeB_detr/best.pth
```

Without these options the output includes only the classical and A1 controls;
it does not constitute an A1-versus-GT comparison. The paths above are placeholders.

Then run a complete-codeword smoke, again adding the actual GT/DETR paths:

```bash
python -u -m evaluation.compare_paper_reference_a1 \
  --a1-root ../A1_PARTNER_MINIMAL \
  --ce-covariance data/cache/paper_reference_ft_cov.pt \
  --channels 2 --ebno-dbs=-10.5 \
  --output results/paper_reference/a1_coded_smoke.json
```

After checking provenance, decoding and native/custom LMMSE agreement, use a
fixed 100-channel characterization around the operating point:

```bash
python -u -m evaluation.compare_paper_reference_a1 \
  --a1-root ../A1_PARTNER_MINIMAL \
  --ce-covariance data/cache/paper_reference_ft_cov.pt \
  --gt-checkpoint /path/to/ModeB_gt/best.pth \
  --detr-checkpoint /path/to/ModeB_detr/best.pth \
  --channels 100 --seed 20260929 --ebno-dbs=-13,-12,-11,-10.5,-10,-9 \
  --output results/paper_reference/a1_gt_detr_coded_100.json
```

Add `--a1-ce-policy both` with a new output name to measure the thermal-only
auxiliary control; this invokes A1 a second time for each channel. Every full
slot evaluates all 2304 data REs with K256, so use the smoke's measured duration
to budget the larger run. No runtime estimate is inferred from CPU tests.

## Outputs and interpretation

The JSON and companion CSV contain per-detector coded-bit BER, bit BCE in nats,
fixed-scale `1-BCE/ln(2)` GMI proxy, and (full-slot mode only) payload BER, TB BLER
and CRC counts. This GMI proxy is neither optimized nor clipped. LLR sign is
log(P1/P0); hard bits are `llr > 0`.

`--re-per-channel` always means BER-only, even if set to 2304. Omit the option
for full native codeword order and actual TB decoding; no incomplete-codeword
BLER is fabricated. Classical controls still run their frozen full-grid path,
then select the same diagnostic REs as A1/GT/DETR.

The JSON saves seeds, selected codeword ordinals, per-channel counts, CE NMSE /
err_var diagnostics, native/custom LMMSE agreement and the package-native LMMSE
cross-check. It also includes package/model/checkpoint hashes, the prior SHA256,
distribution identity and execution environment. Do not change LLR scaling or
sign to improve test results. A large native-baseline discrepancy should stop
the formal experiment for investigation.

`paired_comparisons` uses the package's channel-level paired bootstrap against
Sionna LMMSE, EP5 and available GT/DETR. Positive base-minus-candidate means
fewer candidate errors. Each UE TB is not treated as an independent channel.
One-channel intervals are unavailable, and two-channel intervals are only smoke
diagnostics. A 100-channel characterization is not automatically sufficient to
establish a small improvement.

`channels[].timing` records execution diagnostics with explicit scopes:
classical full-grid group including decode, shared GT/DETR frontend and their
individual forward times, and A1's joint K64/K256 inference including its
preprocessing/CPU readout transfer. There is no separately measured K64 latency.
The first channel includes cold-start effects; these groups are not a matched
standalone speed benchmark and must not be used to claim detector speedups.

Completed channel results are saved after every channel; failures retain the
partial JSON/CSV. New runs refuse existing output paths. This entry point uses
the requested fixed channel budget, without the legacy one-click performance
stopping gate. It has no resume/cache/ZIP upload workflow and does not delete or
modify any external package data or weights. No result is sent to the colleague.
