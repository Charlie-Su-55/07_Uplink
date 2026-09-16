07_Uplink — Codex Handoff

1. Purpose

This repository is a research prototype for uplink MU-MIMO soft detection. The immediate goal is to determine whether a true Graph Transformer / Soft Graph Transformer (SGT) can provide meaningful BER gain over strong model-based baselines.

This handoff is intentionally minimal. Old GEPNet experiments, tests, smoke scripts, checkpoints, logs, and result dumps should be removed before Codex scans the project.

2. Canonical target system

Use this as the reference system unless a diagnostic explicitly says otherwise:

16 simultaneous UEs / streams, 1 stream per UE

BS: 256 physical Rx antennas

UE: 4 Tx antenna elements, equal-gain rank-one stream mapping

Carrier: 6.7 GHz

OFDM: 192 subcarriers, 30 kHz SCS, 14 OFDM symbols

Silent / covariance-observation symbols: 0, 1

DMRS symbols: 2, 13

Data symbols: 3..12

Modulation: 16-QAM

Training channel: 3GPP UMa, forced NLOS, outdoor + O2I mixture

Channel normalization: disabled

Power control: fractional, alpha = 0.8

4 interferers

Total IoT = 10 dB

Interferer relative strengths [dB]: [5.229, 5.229, 5.229, 1.0914]

Receiver SNR used in the current diagnostic: 5 dB

Current detector assumption: oracle h_true, estimated interference-plus-noise covariance ruu_hat

Covariance estimator: shrinkage with lambda = 0.10. Treat lambda=0.10 as fixed; do not re-sweep it.

Canonical system config:

configs/system/uma_16ue_256rx.yaml

The BS array should resolve to 256 antennas, e.g. 8 x 16 x dual polarization.

3. Trusted parts

The following pipeline is considered useful and should be audited but not casually rewritten:

data/channels/topology.py

data/channels/uma.py

data/link/resource_grid.py

data/link/stream_mapping.py

data/link/modulation.py

data/link/uplink_link.py

data/link/power_control.py

data/link/interference.py

data/link/covariance.py

data/link/dmrs.py

data/link/channel_estimation.py

data/preprocessing/whitening.py

data/dataset.py

detectors/classical/lmmse.py

detectors/classical/ep.py

evaluation/metrics.py

evaluation/evaluate_snr.py

Classical receiver convention:

LLR sign is log P(bit=1) / P(bit=0).

Hard bit is llr > 0.

Whitened sufficient statistics are

z = H^H R^{-1} y

G = H^H R^{-1} H

Complex EP is the strongest trusted baseline so far.

Typical 256Rx / 16-stream / estimated-R / 5 dB results are approximately:

LMMSE BER: ~4.3e-2

EP3 BER: ~3.65e-2

EP5 BER: ~3.59e-2 to 3.62e-2 depending on the fixed validation set

Do not treat tiny sub-percent differences as meaningful research gains.

4. What has been abandoned

Old GEPNet line

Many EP-anchored GNN/GEPNet variants were tried, including real-node, complex-node, sparse Top-K, learned gates, covariance-aware gates, coupled/decoupled EP feedback, trust regions, and T5-only correction.

They produced only roughly 0.0x% to 0.x% relative BER gains over strong EP baselines. This line is abandoned. Old GEPNet code, training files, checkpoints, logs, tests, and result JSONs should not be restored unless needed only for historical comparison.

32 physical Rx diagnostic

A temporary 32Rx / 16-stream experiment was used only to test loading effects. It made the physical link much harder and produced ~0.33 BER. It is not the target system and should not become the main architecture. Do not introduce beam reduction unless explicitly requested later.

5. Current SGT prototype: IMPORTANT — DO NOT TRUST IT YET

The only neural detector worth keeping for Codex inspection is the current SGT prototype:

models/graph/sgt.py

training/train_sgt.py

configs/training/sgt_5db.yaml

Its intended structure is based on the Soft Graph Transformer paper:

Convert whitened complex MIMO to a real-valued model.

Create 2*Nr linear-constraint tokens from received observations and channel rows.

Create 2*Nt symbolic tokens from soft priors.

Use self-attention for contextual encoding inside each token set.

Use cross-attention for constraint-to-symbol message passing.

Produce bit-level soft outputs / LLRs.

Current 256Rx implementation uses:

512 linear tokens

32 symbolic tokens

8 SGT layers

8 attention heads

d_model = 128

dropout = 0.1

Current failure

The current SGT run does not learn detection:

training BCE stays around ~0.687-0.692, close to random BCE ln(2)=0.6931

training BER remains roughly 0.45-0.47

validation BER remains roughly 0.45-0.47

EP5 is roughly 0.036

Therefore the present SGT implementation/training pipeline is suspect. Do not optimize it blindly and do not use its result as evidence that Graph Transformers fail.

6. Codex priority tasks

Please work in this order.

Priority A — audit the SGT implementation against the paper

Inspect models/graph/sgt.py line by line and compare it with the Soft Graph Transformer architecture. Specifically verify:

graph-aware tokenization

real-valued MIMO conversion

exact meaning and scaling of linear-constraint tokens

symbolic token / prior representation

positional encodings

direction and recurrence of self-attention and cross-attention

whether both token streams are updated as intended across MP iterations

whether the implementation accidentally loses the weighted factor-graph structure

LLR ordering and QAM bit mapping

normalization of y, H, and noise variance

whether whitening plus normalize_channel: false creates pathological feature scaling

Do not assume the current implementation is faithful just because it uses MultiheadAttention.

Priority B — build a minimal reproduction before touching the UMa target

Before judging SGT on 256Rx UMa, reproduce a small setting close to the paper:

i.i.d. Rayleigh

perfect CSI

QPSK

8x8 first

then 16x16 if 8x8 works

Use a realistic effective batch size via gradient accumulation if needed. The previous micro-batch=8 256Rx run is not a convincing reproduction recipe.

The purpose is binary:

If the SGT cannot learn the small paper-like task, fix implementation/training.

Only after it works there should it be ported back to the 256Rx / 16-stream UMa system.

Priority C — return to the canonical target

Once the SGT reproduction works, evaluate on:

256 physical Rx

16 streams

16-QAM

oracle h_true

estimated ruu_hat

5 dB initially

Compare against the trusted complex EP5 baseline.

7. Hard constraints for the next iteration

Until the SGT reproduction is working, do not add:

beam reduction

learned covariance estimation

channel-estimation error

new GEPNet residual branches

gates

trust-region heuristics

EP feedback hybrids

additional auxiliary losses

multiple competing model variants

The immediate question is simply:

Can a correctly implemented Graph Transformer learn MIMO detection and then beat or materially complement complex EP in the target system?

8. Go / no-go criterion

After a credible SGT reproduction and a clean 256Rx evaluation:

If SGT remains far worse than EP5: stop this research direction.

If SGT only provides a few tenths of a percent relative BER gain: stop; that is not publication-level value.

Continue only if the gain is clearly material and persistent across SNR / channel test conditions.

Do not spend time polishing a 0.x% gain.

9. Environment

Known working environment:

Python 3.12

PyTorch 2.11.0 + CUDA 13.0 build

Sionna 2.0.1

NVIDIA RTX 5090 30 GB

Conda environment name used previously: precoder

10. Minimal expected repository after cleanup

07_Uplink/
├── README.md
├── configs/
│   ├── system/
│   │   └── uma_16ue_256rx.yaml
│   └── training/
│       └── sgt_5db.yaml
├── data/
│   ├── channels/
│   │   ├── topology.py
│   │   └── uma.py
│   ├── link/
│   │   ├── resource_grid.py
│   │   ├── stream_mapping.py
│   │   ├── modulation.py
│   │   ├── uplink_link.py
│   │   ├── power_control.py
│   │   ├── interference.py
│   │   ├── covariance.py
│   │   ├── dmrs.py
│   │   └── channel_estimation.py
│   ├── preprocessing/
│   │   └── whitening.py
│   └── dataset.py
├── detectors/
│   └── classical/
│       ├── lmmse.py
│       └── ep.py
├── evaluation/
│   ├── metrics.py
│   └── evaluate_snr.py
├── models/
│   └── graph/
│       └── sgt.py
└── training/
    └── train_sgt.py

__init__.py files may remain wherever required by Python packages.

11. Known dead / suspicious internal paths to inspect

Previous covariance-aware GEPNet work introduced split-view covariance outputs such as ruu_views / ruu_view_scm. The current SGT and classical baselines do not need them. If they are still referenced only by abandoned code, remove them after checking dependency references.

Likewise, the current SGT trainer contains its own whitening helper even though data/preprocessing/whitening.py exists. Consolidate this only after confirming numerical equivalence.

12. Reference papers

If available locally, give Codex these papers together with the repository:

SOFTGRAPHTRANSFORMERFORMIMODETECTION.pdf

Graph Neural Network Aided Expectation Propagation Detector for MU-MIMO Systems.pdf

Edge-augmented Graph Transformers Global Self-attention is Enough for Graphs.pdf

The Soft Graph Transformer paper is the primary architecture reference for the next step.