"""Independent Mode B channel priors and native LS + LMMSE interpolation.

The prior is an uncentered second moment of the effective rank-one channel,
pooled over calibration channels, UEs, and Rx antennas. Its absolute power is
preserved. Evaluation truth is used only by diagnostics, never by the estimator.
"""

import hashlib
import json
from pathlib import Path


COVARIANCE_KIND = "paper_reference_ft_covariance_v1"


def distribution_spec(cfg):
    sections = ("ofdm", "bs_array", "ut_array", "topology", "channel", "stream_mapping", "link", "paper_contract")
    spec = {key: cfg[key] for key in sections}
    spec["general"] = {key: value for key, value in cfg["general"].items() if key not in ("seed", "device")}
    spec["nr"] = {key: cfg["nr"][key] for key in ("mcs_table", "mcs_index")}
    spec["mode"] = cfg.get("mode", "level0")
    return spec


def distribution_id(cfg):
    return hashlib.sha256(json.dumps(distribution_spec(cfg), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def channel_second_moments(h_eff):
    """h_eff [B,T,F,M,K] -> Rf[f,g]=E[h_f conj(h_g)], Rt likewise."""
    import torch

    if h_eff.ndim != 5 or not h_eff.is_complex() or not torch.isfinite(h_eff).all():
        raise ValueError("Expected finite complex effective channel [B,T,F,M,K].")
    b, t, f, m, k = h_eff.shape
    hf = h_eff.permute(0, 3, 4, 1, 2).reshape(-1, f)
    ht = h_eff.permute(0, 3, 4, 2, 1).reshape(-1, t)
    # Do not reverse the conjugation: that would reverse frequency phase slopes.
    rf = hf.mT @ hf.conj() / (b * m * k * t)
    rt = ht.mT @ ht.conj() / (b * m * k * f)
    return rf, rt


def regularize_covariance(matrix, shrinkage=1e-3):
    """Trace-preserving diagonal shrinkage; no unit-power renormalization."""
    import torch

    if not 0 < shrinkage < 1:
        raise ValueError("Covariance shrinkage must lie in (0,1).")
    matrix = (matrix + matrix.mH) * .5
    power = matrix.diagonal().real.mean()
    if not torch.isfinite(matrix).all() or power <= 0:
        raise ValueError("Invalid calibration covariance.")
    return (1 - shrinkage) * matrix + shrinkage * power * torch.eye(
        matrix.shape[0], dtype=matrix.dtype, device=matrix.device)


def validate_covariance_artifact(artifact, cfg):
    import torch

    if artifact.get("kind") != COVARIANCE_KIND or artifact.get("distribution_id") != distribution_id(cfg):
        raise ValueError("CE covariance is legacy or belongs to a different paper distribution; rebuild it.")
    seeds = artifact.get("calibration_seeds", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("CE covariance needs independent calibration seed provenance.")
    for key, size in (("cov_mat_freq", cfg["ofdm"]["fft_size"]),
                      ("cov_mat_time", cfg["ofdm"]["num_ofdm_symbols"])):
        matrix = artifact.get(key)
        if (not isinstance(matrix, torch.Tensor) or matrix.shape != (size, size)
                or not matrix.is_complex() or not torch.isfinite(matrix).all()
                or not torch.allclose(matrix, matrix.mH, atol=1e-6, rtol=1e-5)):
            raise ValueError(f"Invalid {key}.")
        if not (matrix.diagonal().real > 0).all() or torch.linalg.eigvalsh(matrix).min() < -1e-6:
            raise ValueError(f"Non-PSD {key}.")


def build_estimator(link):
    import torch
    from sionna.phy import config
    from sionna.phy.ofdm import LSChannelEstimator, LMMSEInterpolator

    path = Path(link.cfg["channel_estimation"]["covariance_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing paper CE covariance: {path}. Run tools.build_sionna_reference_covariance first.")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    validate_covariance_artifact(artifact, link.cfg)
    dtype = torch.complex64 if link.precision == "single" else torch.complex128
    rf = artifact["cov_mat_freq"].to(device=link.device, dtype=dtype)
    rt = artifact["cov_mat_time"].to(device=link.device, dtype=dtype)
    # Native v2.0.1 interpolators use global config at construction. Scope it
    # so CPU tests on a CUDA server also build every internal buffer on CPU.
    old_device, old_precision = config.device, config.precision
    try:
        config.device, config.precision = link.device, link.precision
        interpolator = LMMSEInterpolator(link.resource_grid.pilot_pattern, rt, rf, order="f-t")
    finally:
        config.device, config.precision = old_device, old_precision
    estimator = LSChannelEstimator(link.resource_grid, interpolator=interpolator, **link.options)
    metadata = {key: value for key, value in artifact.items() if key not in ("cov_mat_freq", "cov_mat_time")}
    metadata.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    estimator="Sionna LSChannelEstimator + LMMSEInterpolator(f-t)",
                    spatial_smoothing=False, rx_chunk_size=link.cfg["channel_estimation"]["rx_chunk_size"])
    return estimator, metadata


def csi_diagnostics(batch):
    """Data-RE and full-grid NMSE/uncertainty for one shared adapter batch."""
    import torch

    with torch.no_grad():
        truth, estimate, variance = batch["h_true"], batch["h_hat"], batch["err_var"]
        error = (estimate - truth).abs().square()
        power = truth.abs().square()
        b, t, f, m, k = truth.shape
        idx = batch["data_indices"]
        data_error = error.reshape(b, t * f, m, k)[:, idx]
        data_power = power.reshape(b, t * f, m, k)[:, idx]
        data_var = variance.reshape(b, t * f, m, k)[:, idx]
        return dict(
            h_nmse=float(error.sum().div(power.sum().clamp_min(1e-30)).item()),
            data_h_nmse=float(data_error.sum().div(data_power.sum().clamp_min(1e-30)).item()),
            data_h_nmse_per_channel_ue=(data_error.sum((1, 2)) / data_power.sum((1, 2)).clamp_min(1e-30)).cpu().tolist(),
            mean_err_var=float(variance.mean().item()), median_err_var=float(variance.median().item()),
            data_mean_err_var=float(data_var.mean().item()), data_median_err_var=float(data_var.median().item()),
            data_empirical_mse=float(data_error.mean().item()),
            median_definition="torch.median, lower middle for an even number of elements")
