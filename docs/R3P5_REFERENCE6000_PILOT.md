# R3P5 paired BLER pilot

Run from the repository root in the existing `precoder` environment on the GPU
server:

```bash
MPLCONFIGDIR=/tmp/precoder-mpl conda run --no-capture-output -n precoder \
  python -m evaluation.evaluate_r3p5_reference6000_pilot
```

The default pilot uses the eight fixed old validation channel seeds
`81000024`–`81000031` at all seven confirmation Eb/N0 points. It sends one
Mode-B realization per `(seed, Eb/N0)` and decodes complete 16-UE codewords.
Each realization is shared by native Sionna LMMSE, raw EP5, frozen
noise-conditioned per-bit EP5, the three frozen full MLP checkpoints, and the
external R3P5 reference6000 receiver. The ZIP package keeps its native K64
Direct-RB sampling and readout. The runner checks the bundle QAM mapping against
the actual link mapper before evaluating.

This is a diagnostic pilot, not the planned 512-channel evaluation. Its
validation seeds must not be used to tune or select models. R3P5 training seeds
are undisclosed, so training/test independence for that model is unverified.
Treat the point estimates as preliminary; they do not support a strong ranking.
The runner refuses to overwrite an existing output directory and atomically
updates `summary.json` and `summary.csv` after each paired case.
