# R3P5 reference6000 adapter

This is an external frozen transfer model. It is not part of the paper-reference
training distribution, and the supplied bundle does not disclose training seeds.
Its checkpoint is a 4,247,644-parameter adaptive6000 export at step 6000. The
archive SHA256 is `f7ef4e6d5fc74a1ae7d631ce4f4c7e4e5d9c1bb3e7c32c3e1966b41a629bcf98`;
the weight SHA256 recorded and independently checked by the bundle is
`889cf4abd6ed836e023fdd55174b5a4ece3ee03241d433147b1f490c88820466`.

The public API accepts complex64 `y[B,T,F,256]`, `Hhat[B,T,F,256,S]` and
`Ruu[B,256,256]`, with 1–16 streams. It returns float32
`log P(bit=1)/P(bit=0)` in `[B,T,F,S,4]`, using the bundle's normalized,
MSB-first Sionna 16QAM mapping. The frozen receiver applies its own adaptive
cross-RE covariance correction and K64 Direct-RB readout, with RE64 tiling and
its native clipping. It does not decode. This adapter passes every data RE in
time-major/frequency-minor order and refuses incomplete Mode-B grids.

For practical CSI, the adapter applies the existing receiver's diagonal
uncertainty model `v=N0+sum_UE(err_var)`: `y/sqrt(v)`, `Hhat/sqrt(v)`, and
`Ruu=I`. This is the documented common-scale transformation of the package's
input contract; the package performs no hidden second whitening. It does not
pass true channel, transmitted labels, true noise, or decoder outcomes. The
native package loader verifies bundled source and model files before inference.
Because the bundle protects its top-level Python package identity, use it in a
fresh process without another `r3p5_invariant` imported.

Current host preflight recorded PyTorch `2.11.0+cu130`, CUDA runtime `13.0`,
and Sionna `2.0.1`. The bundled synthetic CPU example and adapter contract
unit tests pass. The host's NVIDIA driver is unavailable (`nvidia-smi` cannot
communicate with it), so CUDA model qualification and formal BLER execution
remain pending. The bundled README's 1920-RE historical window differs from
Mode-B's 2304 data RE; its API supports full windows and requires at least eight
REs, but numerical transfer at 2304 RE has not yet been qualified.
