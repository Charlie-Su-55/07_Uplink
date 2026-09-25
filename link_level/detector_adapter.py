"""One received realization, one CSI estimate, multiple existing detector cores.

Ruu always denotes thermal noise alone. An explicitly selected Sionna-diagonal
CE approximation is applied in this adapter, not hidden in Ruu or architecture.
Neural checkpoint loading and training are intentionally absent.
"""

import copy

from link_level.sionna_ce import distribution_id


CLASSICAL_DETECTORS = ("sionna_lmmse", "custom_lmmse", "ep5")


def validate_detector_request(detectors, csi, custom_ce_policy):
    if not detectors or len(detectors) != len(set(detectors)) or set(detectors) - set(CLASSICAL_DETECTORS):
        raise ValueError("Only sionna_lmmse, custom_lmmse and ep5 are enabled; GT/DETR await classical validation. Flow is excluded.")
    if custom_ce_policy not in ("require_explicit", "sionna_diagonal"):
        raise ValueError("Unsupported custom CE uncertainty policy.")
    if csi in ("practical", "both") and set(detectors) - {"sionna_lmmse"} and custom_ce_policy != "sionna_diagonal":
        raise ValueError("Practical custom detection requires explicit --custom-ce-policy sionna_diagonal; err_var cannot be dropped.")


def native_channel_to_grid(h):
    """[B,1,M,K,1,T,F] -> [B,T,F,M,K], without changing user/RE order."""
    if h.ndim != 7 or h.shape[1] != 1 or h.shape[4] != 1:
        raise ValueError("Expected one receiver and one stream per UE.")
    return h[:, 0, :, :, 0].permute(0, 3, 4, 1, 2)


def make_detector_batch(link, sample, csi):
    import torch

    h_hat, err_var = link.estimate_channel(sample, csi)
    b = sample["y"].shape[0]
    return dict(
        y=sample["y"][:, 0].permute(0, 2, 3, 1),
        h_true=sample["h_eff"], h_hat=native_channel_to_grid(h_hat),
        err_var=native_channel_to_grid(err_var.expand_as(h_hat)), n0=sample["no"],
        Ruu=sample["no"] * torch.eye(link.num_rx, dtype=h_hat.dtype, device=h_hat.device).expand(b, -1, -1),
        bits=sample["info"], coded_bits=sample["coded"], data_indices=link.data_indices,
        metadata=dict(csi=csi, seed=sample["seed"], ebno_db=sample["ebno_db"],
                      distribution_id=distribution_id(link.cfg), grid=copy.deepcopy(link.grid_metadata),
                      codec=link.codec.metadata(), Ruu_definition="N0 * I; CE uncertainty is separate",
                      existing_checkpoints_distribution="legacy-distribution",
                      neural_status="disabled_pending_classical_validation"))


def whiten_diagonal_uncertainty(y, h, err_var, n0):
    """Match native equalizer's N0 + sum_UE(err_var) diagonal approximation.

    This is a model-based variance, not empirical residual noise or true-H MSE.
    Data symbols have unit expected energy, and there is no external interference.
    """
    import torch

    if y.shape != h.shape[:-1] or err_var.shape != h.shape:
        raise ValueError("Incompatible y/Hhat/err_var shapes.")
    variance = n0 + err_var.sum(-1)
    if not torch.isfinite(variance).all() or (variance <= 0).any() or (err_var < 0).any():
        raise ValueError("Invalid CE/noise variance.")
    std = variance.sqrt()
    return y / std, h / std.unsqueeze(-1)


def llrs_to_native_order(llrs):
    """[B,Ndata,K,Qm] -> [B,K,1,G], preserving full codeword order."""
    if llrs.ndim != 4:
        raise ValueError("Expected LLRs [B,Ndata,K,Qm].")
    b, n, k, q = llrs.shape
    return llrs.permute(0, 2, 1, 3).reshape(b, k, 1, n * q)


class ClassicalDetectorAdapter:
    def __init__(self, link, detectors=CLASSICAL_DETECTORS, custom_ce_policy="require_explicit", re_chunk=128):
        validate_detector_request(detectors, link.csi, custom_ce_policy)
        if re_chunk < 1:
            raise ValueError("re_chunk must be positive.")
        self.link, self.detectors = link, tuple(detectors)
        self.custom_ce_policy, self.re_chunk = custom_ce_policy, re_chunk
        self.custom, self.ep = None, None
        if set(detectors) - {"sionna_lmmse"}:
            from detectors.classical.lmmse import LMMSESoftDetector
            cfg = copy.deepcopy(link.cfg)
            cfg["modulation"] = {"bits_per_symbol": link.codec.qm}
            self.custom = LMMSESoftDetector(cfg)
            if "ep5" in detectors:
                from detectors.classical.ep import ExpectationPropagationDetector
                self.ep = ExpectationPropagationDetector(cfg, num_iterations=5)

    def evaluate(self, sample, batch):
        """This method never calls transmit() or a channel provider."""
        import torch

        csi = batch["metadata"]["csi"]
        validate_detector_request(self.detectors, csi, self.custom_ce_policy)
        outputs = {}
        with torch.no_grad():
            if "sionna_lmmse" in self.detectors:
                outputs["sionna_lmmse"] = self.link.receive(sample, csi=csi)
            if self.custom is None:
                return outputs
            b, t, f, m = batch["y"].shape
            k = batch["h_hat"].shape[-1]
            indices = batch["data_indices"]
            llrs = {name: [] for name in self.detectors if name != "sionna_lmmse"}
            symbols, variances = [], []
            for start in range(0, len(indices), self.re_chunk):
                idx = indices[start:start + self.re_chunk]
                y = batch["y"].reshape(b, t * f, m)[:, idx].unsqueeze(1)
                h = batch["h_hat"].reshape(b, t * f, m, k)[:, idx].unsqueeze(1)
                ruu = batch["Ruu"]
                if csi == "practical":
                    error = batch["err_var"].reshape(b, t * f, m, k)[:, idx].unsqueeze(1)
                    y, h = whiten_diagonal_uncertainty(y, h, error, batch["n0"])
                    ruu = torch.eye(m, dtype=h.dtype, device=h.device).expand(b, -1, -1)
                custom = self.custom(y, h, ruu)
                if "custom_lmmse" in llrs:
                    llrs["custom_lmmse"].append(custom["llr"][:, 0])
                    symbols.append(custom["x_hat"][:, 0])
                    variances.append(custom["no_eff"][:, 0])
                if "ep5" in llrs:
                    ep = self.ep(custom["z"], custom["gram"], return_iterations=(5,))
                    llrs["ep5"].append(ep["llr"][:, 0])
            for name, pieces in llrs.items():
                llr = llrs_to_native_order(torch.cat(pieces, dim=1))
                if not torch.isfinite(llr).all():
                    raise RuntimeError(f"Non-finite {name} LLRs.")
                decoded, crc_ok = self.link.codec.decode(llr)
                outputs[name] = dict(llr=llr, decoded=decoded, crc_ok=crc_ok)
            if "custom_lmmse" in outputs:
                outputs["custom_lmmse"].update(
                    x_hat=torch.cat(symbols, 1).permute(0, 2, 1).unsqueeze(2),
                    no_eff=torch.cat(variances, 1).permute(0, 2, 1).unsqueeze(2))
        return outputs


def compare_lmmse_outputs(outputs):
    if not {"sionna_lmmse", "custom_lmmse"} <= outputs.keys():
        return None
    result = {}
    for key in ("x_hat", "no_eff", "llr"):
        reference, actual = outputs["sionna_lmmse"][key], outputs["custom_lmmse"][key]
        result[key + "_relative_rms"] = float(((actual - reference).abs().square().mean()
            / reference.abs().square().mean().clamp_min(1e-30)).sqrt().item())
    return result
