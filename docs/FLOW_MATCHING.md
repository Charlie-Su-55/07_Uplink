# Flow-Matching control for 07_Uplink

This is a new, trainable control derived from
`PARTNER_FLOW_LLR_RETRAIN_16QAM/posterior_set_receiver/{base_model,sampling,constellation}.py`.
The original package is untouched. Its checkpoints cannot be loaded into this
architecture; train from random initialization.

## Interfaces and dimensions

| Input/output | Shape | Meaning |
|---|---|---|
| `z` | `[N_RE,S]` or `[B,T,F,S]` | `H_hat^H Ruu_hat^-1 y` |
| `gram` | `[N_RE,S,S]` or `[B,T,F,S,S]` | `H_hat^H Ruu_hat^-1 H_hat` |
| `out["llr"]` | `[N_RE,S,Qm]` or `[B,T,F,S,Qm]` | Final soft detector output |
| `out["raw_flow_llr"]` | same as final LLR | Conditional-probability Flow readout |
| `out["lmmse_llr"]` | same as final LLR | APP LMMSE anchor |
| Training `bits` | same as final LLR | Binary labels, identical RE/user/bit order |

The default system is S=16, Qm=4. The model also supports active S=1..configured
capacity and Qm=2/4/6 with separate weights for each modulation. All leading
dimensions are preserved. It accepts `return_iterations=(5,)` for evaluator
compatibility; this does not request five Flow steps. Each trajectory has S
serial unmasking steps.

LLR is `log P(bit=1)/P(bit=0)`; hard decisions use `llr > 0`. QAM uses the
normalized Gray PAM construction and natural-binary row labels used by the
existing mapper. Symbols 3..12 are data. Select these before using the raw
adapter; the model does not implicitly remove silent symbols or DMRS.

```python
from models.baselines.flow_matching_detector import FlowMatchingDetector

model = FlowMatchingDetector(cfg).to(device)
out = model(z, gram)  # same inputs as GT-EP / DETR-EP
llr = out["llr"]

# Optional three-input adapter, useful for comparison with the partner API:
# y: [B,10,192,256], H: [B,10,192,256,16], Ruu: [B,256,256]
out = model.detect(y_data, h_hat_lmmse_data, ruu_hat)
assert out["llr"].shape == (y_data.shape[0], 10, 192, 16, 4)
```

The raw adapter uses the existing CovarianceAwareFrontEnd exactly once.
Ruu already contains interference AND thermal noise; do not add noise again.
The sufficient-statistics interface needs no antenna projection, covariance
eigenfeatures, observation energy, MCS or true-H input. Padded/inactive streams
must be sliced out upstream. Computation is complex64/float32, matching the
canonical single-precision experiment.

## Changes from the partner implementation

- Retains the discrete absorbing-MASK, random-order conditional categorical
  process and Gram-conditioned stream attention. This is discrete Flow, not
  a continuous ODE model.
- Uses 3 layers, width 128, 8 heads and FFN 256. There are no stream identity
  embeddings, and the conditional network is equivariant to stream reordering.
- Caches LMMSE, normalized Gram edges and context embeddings before expanding
  trajectories. The physical adapter accounts approximately for uncertainty
  in unresolved streams instead of treating their completion as exact.
- Streams the conditional bit probabilities and their second moments directly
  on the compute device. No full candidate population is copied to CPU.
  RE tiles and trajectory tiles independently bound peak inference memory.
- Default K=64, configurable at training time (`--num-samples 256` restores
  the partner's candidate budget). Fixed scrambled Sobol draws define orders
  and categorical uniforms. The same quadrature is shared across REs, so
  changing execution chunks does not change which draws an RE receives.
  Sampling does not advance the simulator RNG. This finite-K sampling rule
  is not exactly stream-permutation equivariant; the conditional network is.
- Uses APP log-sum-exp LLRs to match this repository's LMMSE baseline. The
  original partner anchor used clipped max-log LLRs. A zero-initialized,
  per-stream residual head starts at exactly the internal APP LMMSE output.
  Its correction is bounded to +/-20; the APP base/final output is not clipped.
- Proposal training uses symbol CE at a uniformly sampled prefix depth and
  stream order. Final bit BCE trains the residual head on sampled deployment
  features. Discrete sampling is detached; BCE does not backpropagate through
  token draws. There is no training-truth input in the inference path.

These are architectural/efficiency changes, not a measured BER/BLER improvement.
The z/G interface intentionally omits the partner's raw-antenna and covariance
shape features so all practical controls use the existing common input.
Report any resulting accuracy tradeoff empirically. Smaller K and width change
compute cost; report K, parameter count and latency with comparison results.

## Training and validation

Run from `07_Uplink` on the existing Sionna 2 GPU server. No dependency changes
are needed. The canonical dataset, practical H/Ruu, physical seed schedule and
fixed held-out validation sampler are reused. True-H is never a model input.
Best weights are selected by held-out final bit BCE, as appropriate for a soft
decoder input; GT/DETR currently select by BER, so disclose that difference.

First run a small integration smoke in its own directory:

```bash
python -m training.train_flow_matching \
  --bits-per-symbol 4 --snr-db 8 --mcs-table 1 --mcs-index 11 \
  --steps 1 --re-per-step 4 --val-channels 1 --val-re-per-channel 4 --val-every 1 \
  --num-samples 8 --sample-chunk 4 --re-chunk 4 \
  --output-dir ckp/flow_smoke
```

Example full 16QAM specialist (same T1/MCS11 SNR region as the existing controls):

```bash
python -m training.train_flow_matching \
  --bits-per-symbol 4 --mcs-table 1 --mcs-index 11 \
  --snr-min-db 7 --snr-max-db 9 --val-snr-db 8 \
  --steps 3000 --re-per-step 128 --val-channels 32 --val-re-per-channel 64 \
  --val-every 100 --num-samples 64 --sample-chunk 16 --re-chunk 16 --seed 42 \
  --output-dir ckp/mcs_specialists/t1_mcs11_16qam_flow_matching_lmmseH_estR_snr7to9db
```

Outputs: `best.pth`, `last.pth`, `history.json`. Checkpoints record model
configuration, K, sampling seed/draws, system settings, source revision, cache
hash, training arguments and optional MCS identity. Existing outputs are rejected
unless `--fresh` explicitly replaces those three artifacts. Automatic resume
is not implemented. `last.pth` contains optimizer/scheduler state for manual
recovery. A best checkpoint at step 0 is the untrained LMMSE anchor and must not
be presented as evidence of learned improvement.

For evaluation, model dimensions/K are restored from the checkpoint. Runtime
`--chunk-size` / `--gt-chunk` affect execution only. Load Flow weights using
`flow_from_checkpoint(cfg, checkpoint)`; partner/other-architecture, wrong
modulation, wrong CSI/covariance and mismatched stream capacities are rejected.
Coded evaluation also requires matching MCS table/index metadata.

## Paired BER and coded BLER

The existing BER evaluator gains an optional `--flow-checkpoint`. Without it,
the existing experiment is unchanged. Flow shares the channels, REs and bit
labels; the result includes its channel BER and paired bootstrap comparisons
against EP5, GT-EP and DETR-EP.

```bash
python -m evaluation.evaluate_gt_detr_lmmse \
  --bits-per-symbol 4 --snr-db 8 --channels 2 --re-per-channel 8 \
  --chunk-size 8 --bootstrap 100 --print-every 1 --seed 20260921 \
  --gt-checkpoint ckp/mcs_specialists/t1_mcs11_16qam_gt_ep_lmmseH_estR_snr7to9db/best.pth \
  --detr-checkpoint ckp/mcs_specialists/t1_mcs11_16qam_detr_ep_lmmseH_estR_snr7to9db/best.pth \
  --flow-checkpoint ckp/flow_smoke/best.pth \
  --output results/flow_smoke/ber.json
```

For coded evaluation, add optional `"flow": "path/to/best.pth"` alongside
`gt` and `detr` in the existing MCS checkpoint map. The provided
`configs/evaluation/flow_mcs11.example.json` points at the full training output
above; it contains paths only, not supplied weights. To use smoke weights,
copy the map and replace its Flow path with `ckp/flow_smoke/best.pth`.

```bash
python -m evaluation.evaluate_mcs_bler \
  --tables 1 --mcs 11 --operating-point --formal-channels 2 --gt-chunk 8 \
  --checkpoint-map configs/evaluation/flow_mcs11.example.json \
  --seed 20260921 --output-dir results/flow_bler_smoke
```

Flow is included in BLER/BER counts, fixed/broken-vs-EP counts, crossings, CSV
summaries and checkpoint provenance. The existing
`[B,T,F,S,Qm] -> [B,S,T*F*Qm]` decoder conversion is reused. The shared neural
chunk adapter now derives S from the input rather than hard-coding 16.

## Checks and limits

```bash
python -m compileall -q models training evaluation tests
python -m unittest discover -s tests -v
python -m training.train_flow_matching --help
```

Synthetic PyTorch CPU tests cover shape preservation, noncontiguous inputs,
S=16, all three modulations, known bit signs, exact scalar APP, raw front-end
equivalence, unit scaling, gradients, chunk/RNG invariance, checkpoint contracts
and coded-bit ordering. The existing standard-library audit suite remains.
A Sionna 2-specific mapper/APP equivalence check runs automatically on the
server and skips on the laptop's older Sionna version.

No full radio simulation, trained Flow checkpoint or measured performance
claim is supplied by this code change. Run the GPU smoke before formal training
and paired BER/BLER evaluation.
