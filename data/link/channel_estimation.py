import torch
from sionna.phy.ofdm import LSChannelEstimator as SionnaLSChannelEstimator
from sionna.phy.ofdm import LMMSEInterpolator, PilotPattern, ResourceGrid


class LSLinearChannelEstimator:
    def __init__(self, cfg, resource_grid, dmrs):
        ccfg = cfg["channel_estimation"]
        self.mode = ccfg["mode"]
        self.frequency_interpolation = ccfg["frequency_interpolation"]
        self.time_interpolation = ccfg["time_interpolation"]
        self.grid = resource_grid
        self.dmrs = dmrs

        if self.mode != "ls_linear":
            raise ValueError(f"Unsupported channel estimation mode: {self.mode}")
        if self.frequency_interpolation != "linear" or self.time_interpolation != "linear":
            raise ValueError("Current LS estimator only supports linear frequency/time interpolation.")
        if len(self.grid.dmrs_symbols) != 2:
            raise ValueError("Current LS estimator expects exactly two DMRS symbols.")

    @staticmethod
    def _interpolate_frequency(values, pilot_indices, num_subcarriers):
        device = values.device
        pilot_idx = torch.tensor(pilot_indices, dtype=torch.long, device=device)
        full_idx = torch.arange(num_subcarriers, dtype=torch.long, device=device)

        right = torch.searchsorted(pilot_idx, full_idx)
        left = torch.clamp(right - 1, 0, pilot_idx.numel() - 1)
        right = torch.clamp(right, 0, pilot_idx.numel() - 1)

        x0, x1 = pilot_idx[left], pilot_idx[right]
        denom = (x1 - x0).clamp_min(1).to(values.real.dtype)
        weight = (full_idx - x0).to(values.real.dtype) / denom
        weight = torch.where(left == right, torch.zeros_like(weight), weight)

        v0, v1 = values[:, left], values[:, right]
        return v0 * (1.0 - weight)[None, :, None] + v1 * weight[None, :, None]

    def __call__(self, y):
        if y.ndim != 4:
            raise ValueError(f"Expected y [B,S,F,M], got {tuple(y.shape)}")

        b, num_symbols, num_subcarriers, num_rx = y.shape
        num_streams = self.dmrs.num_streams
        dmrs_symbols = self.grid.dmrs_symbols
        pilot = self.dmrs.pilot_amplitude

        h_dmrs = torch.zeros(
            b, len(dmrs_symbols), num_subcarriers, num_rx, num_streams,
            dtype=y.dtype, device=y.device
        )

        for d, symbol_idx in enumerate(dmrs_symbols):
            for k, pilot_indices in enumerate(self.dmrs.pilot_indices):
                idx = torch.tensor(pilot_indices, dtype=torch.long, device=y.device)
                h_ls = y[:, symbol_idx, idx, :] / pilot
                h_dmrs[:, d, :, :, k] = self._interpolate_frequency(
                    h_ls, pilot_indices, num_subcarriers
                )

        s0, s1 = dmrs_symbols
        time_idx = torch.arange(num_symbols, dtype=y.real.dtype, device=y.device)
        weight = ((time_idx - float(s0)) / float(s1 - s0)).clamp(0.0, 1.0)

        h0, h1 = h_dmrs[:, 0], h_dmrs[:, 1]
        h_hat = h0[:, None] * (1.0 - weight)[None, :, None, None, None]
        h_hat = h_hat + h1[:, None] * weight[None, :, None, None, None]

        return {
            "h_hat": h_hat,
            "h_dmrs_freq_interp": h_dmrs,
        }


class LMMSEChannelEstimator:
    """
    Sionna-2.0-compatible LS-at-pilots + LMMSE f/t interpolation.

    Two implementation details are essential for this project:

    1) The complete DMRS OFDM symbols are reserved in the PilotPattern, and
       each stream has non-zero pilots only on its interleaved comb positions.
       This matches Sionna's KroneckerPilotPattern semantics: the other DMRS REs
       are marked "unused" for that stream rather than "data".

    2) Physical UMa channels are not normalized (normalize_channel=False), so
       their covariance power can be far below 1e-12. Sionna's internal LMMSE
       implementation contains absolute numerical thresholds. We therefore
       multiply BOTH channel covariance matrices and AWGN variance by the same
       scalar. This leaves the LMMSE matrix mathematically unchanged but moves
       the computation to a numerically safe scale.

    Project I/O:
        y      [B,S,F,M]
        h_hat  [B,S,F,M,K]
    """
    def __init__(
        self,
        cfg,
        resource_grid,
        dmrs,
        covariance_path="data/cache/uma_lmmse_ft_cov.pt",
        order="f-t",
        rx_chunk_size=8,
    ):
        self.grid = resource_grid
        self.dmrs = dmrs
        self.device = cfg["general"]["device"]
        self.precision = cfg["general"]["precision"]
        self.mode = f"ls_plus_lmmse_{order}"
        self.order = order
        self.rx_chunk_size = int(rx_chunk_size)

        if self.rx_chunk_size <= 0:
            raise ValueError("rx_chunk_size must be positive.")
        if order not in ("f-t", "t-f"):
            raise ValueError("Use order='f-t' or order='t-f'.")
        if len(self.grid.dmrs_symbols) != 2:
            raise ValueError("Current LMMSE setup expects exactly two DMRS symbols.")

        state = torch.load(covariance_path, map_location="cpu", weights_only=False)
        cov_freq = state["freq_cov"].to(self.device)
        cov_time = state["time_cov"].to(self.device)

        s = int(self.grid.num_ofdm_symbols)
        f = int(self.grid.num_subcarriers)
        k = int(self.dmrs.num_streams)

        if cov_freq.shape != (f, f):
            raise ValueError(f"freq_cov shape {tuple(cov_freq.shape)} does not match {(f, f)}")
        if cov_time.shape != (s, s):
            raise ValueError(f"time_cov shape {tuple(cov_time.shape)} does not match {(s, s)}")

        p_freq = torch.diagonal(cov_freq).real.mean()
        p_time = torch.diagonal(cov_time).real.mean()
        if p_freq <= 0 or p_time <= 0:
            raise ValueError(f"Non-positive covariance power: freq={p_freq.item()}, time={p_time.item()}")

        # Both covariance matrices estimate the same E[|h|^2]. Use their
        # geometric mean as one common reference so that the exact same scale
        # factor is applied to R_f, R_t, and the pilot LS error variance.
        p_ref = torch.sqrt(p_freq * p_time)
        self.covariance_power = float(p_ref.item())
        self.stat_scale = float((1.0 / p_ref).item())

        cov_freq = cov_freq * self.stat_scale
        cov_time = cov_time * self.stat_scale

        # Reserve the full DMRS symbols, exactly as Sionna's Kronecker pilot
        # pattern does. Non-zero pilots identify the comb of each stream.
        mask = torch.zeros(k, 1, s, f, dtype=torch.bool, device=self.device)
        mask[:, :, list(self.grid.dmrs_symbols), :] = True

        num_reserved = len(self.grid.dmrs_symbols) * f
        cdtype = torch.complex64 if self.precision == "single" else torch.complex128
        pilots = torch.zeros(k, 1, num_reserved, dtype=cdtype, device=self.device)
        pilot_value = torch.tensor(self.dmrs.pilot_amplitude, dtype=cdtype, device=self.device)

        for stream_idx, pilot_indices in enumerate(self.dmrs.pilot_indices):
            idx = torch.tensor(pilot_indices, dtype=torch.long, device=self.device)
            for d in range(len(self.grid.dmrs_symbols)):
                pilots[stream_idx, 0, d * f + idx] = pilot_value

        pilot_pattern = PilotPattern(
            mask=mask,
            pilots=pilots,
            normalize=False,
            precision=self.precision,
            device=self.device,
        )

        self.sionna_grid = ResourceGrid(
            num_ofdm_symbols=s,
            fft_size=f,
            subcarrier_spacing=float(self.grid.subcarrier_spacing),
            num_tx=k,
            num_streams_per_tx=1,
            cyclic_prefix_length=int(self.grid.cyclic_prefix_length),
            num_guard_carriers=(0, 0),
            dc_null=False,
            pilot_pattern=pilot_pattern,
            precision=self.precision,
            device=self.device,
        )

        interpolator = LMMSEInterpolator(
            pilot_pattern=self.sionna_grid.pilot_pattern,
            cov_mat_time=cov_time,
            cov_mat_freq=cov_freq,
            order=order,
        )

        self.estimator = SionnaLSChannelEstimator(
            resource_grid=self.sionna_grid,
            interpolator=interpolator,
            precision=self.precision,
            device=self.device,
        )

        print(
            f"[LMMSE CE] order={order} | covariance power={self.covariance_power:.3e} | "
            f"internal scale={self.stat_scale:.3e} | Rx chunk={self.rx_chunk_size}"
        )

    @staticmethod
    def _format_noise_variance(noise_var, batch_size, num_rx_ant, device, dtype):
        no = torch.as_tensor(noise_var, device=device, dtype=dtype)

        if no.ndim == 0 or no.numel() == 1:
            return no.reshape(())

        if no.shape[0] == batch_size:
            if no.ndim == 1:
                return no

            if no.shape[-1] == num_rx_ant:
                reduce_dims = tuple(range(1, no.ndim - 1))
                if reduce_dims:
                    no = no.mean(dim=reduce_dims)
                return no[:, None, :]

            return no.reshape(batch_size, -1).mean(dim=-1)

        return no.mean()

    def __call__(self, y, noise_var):
        if y.ndim != 4:
            raise ValueError(f"Expected y [B,S,F,M], got {tuple(y.shape)}")

        b, s, f, m = y.shape
        if s != self.grid.num_ofdm_symbols or f != self.grid.num_subcarriers:
            raise ValueError(
                f"Grid mismatch: y has S,F={(s, f)}, expected "
                f"{(self.grid.num_ofdm_symbols, self.grid.num_subcarriers)}"
            )

        no_full = self._format_noise_variance(
            noise_var,
            batch_size=b,
            num_rx_ant=m,
            device=y.device,
            dtype=y.real.dtype,
        )

        # Numerical rescaling only. R and noise variance are multiplied by the
        # same scalar, so R(R+Sigma)^-1 is unchanged.
        no_full = no_full * self.stat_scale

        h_chunks = []
        e_chunks = []

        with torch.no_grad():
            for start in range(0, m, self.rx_chunk_size):
                end = min(start + self.rx_chunk_size, m)

                y_chunk = y[..., start:end]
                y_sionna = y_chunk.permute(0, 3, 1, 2).unsqueeze(1)

                if torch.is_tensor(no_full) and no_full.ndim >= 3 and no_full.shape[-1] == m:
                    no_chunk = no_full[..., start:end]
                else:
                    no_chunk = no_full

                h_hat, err_var = self.estimator(y_sionna, no_chunk)

                h_hat = h_hat[:, 0, :, :, 0, :, :].permute(0, 3, 4, 1, 2).contiguous()
                err_var = err_var[:, 0, :, :, 0, :, :].permute(0, 3, 4, 1, 2).contiguous()

                h_chunks.append(h_hat)
                # Convert reported error variance back to physical channel scale.
                e_chunks.append(err_var * self.covariance_power)

        return {
            "h_hat": torch.cat(h_chunks, dim=3),
            "err_var": torch.cat(e_chunks, dim=3),
        }
