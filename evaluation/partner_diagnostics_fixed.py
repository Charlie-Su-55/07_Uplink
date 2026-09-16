"""Deployment diagnostics for the portable three-input Flow detector.

All checks use only y, h, Ruu and optional transmitted bits.  The report is
JSON-serializable and intentionally separates physical preprocessing, candidate
coverage, neural readout and bit semantics.
"""
from __future__ import annotations

from typing import Iterable
import torch

from three_input_receiver.constellation import lmmse_from_sufficient
from three_input_receiver.deployment_preprocess import covariance_factor, whiten
from three_input_receiver.preprocess import observable_features
from three_input_receiver.sampling import (
    _canonical_rng_for_re_span,
    generate_candidate_population,
)


COMPONENTS = ("lmmse_llr", "raw_flow_llr", "llr")


def _number(value):
    return float(torch.as_tensor(value).detach().cpu())


def _summary(value):
    x = value.detach().float()
    return {"mean": _number(x.mean()), "rms": _number(x.square().mean().sqrt()),
            "min": _number(x.amin()), "max": _number(x.amax())}


def _hard(detector, llr):
    return llr > 0 if detector.llr_sign == 1 else llr < 0


def _pair(detector, reference, actual):
    x, y = reference.detach().float().reshape(-1), actual.detach().float().reshape(-1)
    xc, yc = x-x.mean(), y-y.mean()
    corr = (xc*yc).sum()/(xc.square().sum()*yc.square().sum()).clamp_min(1e-20).sqrt()
    scale = (x*y).sum()/y.square().sum().clamp_min(1e-20)
    return {"correlation": _number(corr),
            "hard_disagreement": _number((_hard(detector, reference)!=_hard(detector, actual)).float().mean()),
            "best_scale_to_reference": _number(scale),
            "mae": _number((x-y).abs().mean()),
            "scaled_mae": _number((x-scale*y).abs().mean()),
            "max_abs": _number((x-y).abs().amax())}


def _validate_bits(bits, shape, qm, device):
    bits = torch.as_tensor(bits, device=device)
    if tuple(bits.shape) != (*shape, qm):
        raise ValueError(f"bits must be {(*shape, qm)}, got {tuple(bits.shape)}")
    if not bool(((bits == 0) | (bits == 1)).all()):
        raise ValueError("bits must contain only 0/1")
    return bits.bool()


def _ber(detector, llr, bits):
    error = (_hard(detector, llr) != bits)
    return {"overall": _number(error.float().mean()),
            "per_bit": [_number(x) for x in error.float().mean(tuple(range(error.ndim-1)))],
            "errors": int(error.sum().item()), "total_bits": int(error.numel())}


def _platform_to_internal_bits(detector, bits):
    internal = torch.empty_like(bits)
    for platform_bit, (model_bit, sign) in enumerate(zip(
            detector._bit_order.tolist(), detector._bit_sign.tolist())):
        value = bits[..., platform_bit]
        internal[..., model_bit] = value if sign == 1 else ~value
    return internal


def _map_llr(detector, value):
    return (value[...,detector._bit_order]*detector._bit_sign*detector.llr_sign).float()


@torch.inference_mode()
def lmmse_only(detector,y,h,ruu):
    """Compute the packaged LMMSE endpoint without running K256 Flow sampling."""
    w,_=covariance_factor(ruu);b,t,f,_=y.shape;streams=h.shape[-1]
    yf=y.reshape(-1,256);hf=h.reshape(-1,256,streams);parts=[]
    for start in range(0,len(yf),64):
        stop=min(start+64,len(yf));bi=torch.arange(start,stop,device=y.device)//(t*f)
        yw,hw=whiten(w[bi],yf[start:stop],hf[start:stop])
        gram=hw.mH@hw;matched=(hw.mH@yw[...,None]).squeeze(-1)
        endpoint=lmmse_from_sufficient(gram,matched,torch.ones(stop-start,device=y.device),
                                       detector.model.table).llrs
        parts.append(_map_llr(detector,endpoint))
    return torch.cat(parts).reshape(b,t,f,streams,detector.model.table.num_bits)


def _candidate_report(detector, y, h, ruu, bits, max_re):
    # Deep candidate inspection is deliberately bounded and uses sample 0 only.
    n = min(int(max_re), int(y.shape[1]*y.shape[2]))
    streams, qm = int(h.shape[-1]), int(detector.model.table.num_bits)
    yy = y[:1].reshape(1,-1,256)[:,:n].reshape(n,256)
    hh = h[:1].reshape(1,-1,256,streams)[:,:n].reshape(n,256,streams)
    w, cov = covariance_factor(ruu[:1])
    yw, hw = whiten(w.expand(n,-1,-1), yy, hh)
    u = observable_features(yw, hw, cov.expand(n,-1))
    gram=hw.mH@hw;matched=(hw.mH@yw[...,None]).squeeze(-1)
    noise=torch.ones(n,device=detector.device)
    lmmse=lmmse_from_sufficient(gram,matched,noise,detector.model.table)
    lmmse.observation_energy=yw.abs().square().sum(-1).real.float()
    lmmse.uncertainty_features=u
    prepared=detector.model.prepare_receiver_context(yw,hw,gram,matched)
    order,uniforms=_canonical_rng_for_re_span(start=0,stop=n,streams=streams,
        num_samples=256,seed_root=detector.seed,device=detector.device)
    population=generate_candidate_population(detector.model,gram,matched,noise,lmmse,
        num_samples=256,sample_chunk=128,seed=detector.seed,random_order=order,
        uniforms=uniforms,model_forward_kwargs=prepared)
    raw=detector.model.table.index_probabilities_to_llrs(
        population["rao_blackwell_probability"].to(detector.device),clip=20)
    fusion=detector.model.candidate_set_llrs(
        candidate_indices=population["candidate_tokens"].to(detector.device),
        physical_energy=population["physical_energy"].to(detector.device),
        path_log_q=population["path_log_q"].to(detector.device),flow_llrs=raw,
        lmmse_llrs=lmmse.llrs,gram=gram,post_variance=lmmse.post_variance,
        uncertainty_features=u,return_components=True)
    outputs={
        "lmmse_llr":_map_llr(detector,lmmse.llrs),
        "raw_flow_llr":_map_llr(detector,raw),
        "llr":_map_llr(detector,fusion["llrs"]),
    }
    candidates = population["candidate_tokens"].long()
    energy, logq = population["physical_energy"].float(), population["path_log_q"].float()
    table_bits = detector.model.table.bit_table.detach().cpu().bool()
    report = {
        "inspected_batch": 0, "inspected_re": n, "K": int(candidates.shape[1]),
        "alpha": _summary(fusion["alpha"]),
        "soft_alpha": _summary(fusion["soft_alpha"]),
        "importance_ess_fraction": _summary(fusion["importance_ess_fraction"]),
        "components": {name:_summary(value) for name,value in outputs.items()},
        "observable_features": {
            name: _summary(u[...,index]) for index,name in enumerate((
                "ruu_mean_eigenvalue", "ruu_mean_inverse_eigenvalue",
                "ruu_condition_number", "ruu_effective_rank_fraction",
                "whitened_channel_energy", "stream_coupling", "whitened_ls_residual_power"))},
    }
    if bits is None:
        return report,outputs
    truth_bits = bits[:1].reshape(1,-1,streams,qm)[:,:n].reshape(n,streams,qm)
    truth_internal = _platform_to_internal_bits(detector, truth_bits).cpu()
    matches = (truth_internal[:,:,None,:] == table_bits[None,None,:,:]).all(-1)
    if not bool((matches.sum(-1) == 1).all()):
        raise RuntimeError("Transmitted bits do not map uniquely to the packaged constellation")
    truth_index = matches.float().argmax(-1)
    per_stream_hit = (candidates == truth_index[:,None,:]).any(1)
    joint_hit = (candidates == truth_index[:,None,:]).all(-1).any(1)
    candidate_bits = table_bits[candidates]
    hamming = (candidate_bits != truth_internal[:,None,:,:]).sum((-1,-2))
    oracle_errors = hamming.amin(1)
    importance_top = (-energy-logq).argmax(1)
    physical_top = (-energy).argmax(1)
    rows = torch.arange(n)
    importance_bits = table_bits[candidates[rows,importance_top]]
    physical_bits = table_bits[candidates[rows,physical_top]]
    report.update({
        "joint_truth_in_K": _number(joint_hit.float().mean()),
        "per_stream_truth_in_K": _number(per_stream_hit.float().mean()),
        "oracle_candidate_ber": _number(oracle_errors.sum()/float(n*streams*qm)),
        "importance_top1_ber": _number((importance_bits!=truth_internal).float().mean()),
        "physical_top1_ber": _number((physical_bits!=truth_internal).float().mean()),
        "ber": {name:_ber(detector,value,truth_bits) for name,value in outputs.items()},
    })
    return report,outputs


@torch.inference_mode()
def diagnose(detector, y, h, ruu, *, bits=None,
             scales: Iterable[float]=(1.0,1e2,1e4,1e5,3e5,1e6),
             max_re=64, deep_candidates=True):
    """Return a bounded same-batch deployment diagnosis.

    ``scales`` applies the physically equivalent transform
    ``(y,H,Ruu)->(a*y,a*H,a**2*Ruu)`` on at most ``max_re`` REs.  A portable
    receiver should retain hard decisions under this coordinate change.
    """
    if y.ndim != 4 or h.ndim != 5 or h.shape[:-1] != y.shape:
        raise ValueError("Expected y[B,T,F,256], h[B,T,F,256,S]")
    qm = int(detector.model.table.num_bits)
    truth = None if bits is None else _validate_bits(bits, (*y.shape[:3],h.shape[-1]), qm, y.device)
    base = detector.detect(y,h,ruu)
    report = {"schema":"three-input-flow-diagnostic-v2",
        "shape":{"y":list(y.shape),"h":list(h.shape),"ruu":list(ruu.shape),"Qm":qm},
        "components":{name:_summary(base[name]) for name in COMPONENTS},
        "comparisons":{
            "raw_vs_lmmse":_pair(detector,base["lmmse_llr"],base["raw_flow_llr"]),
            "final_vs_lmmse":_pair(detector,base["lmmse_llr"],base["llr"]),
            "final_vs_raw":_pair(detector,base["raw_flow_llr"],base["llr"]),
        }}
    if truth is not None:
        report["ber"]={name:_ber(detector,base[name],truth) for name in COMPONENTS}

    n=min(int(max_re),int(y.shape[1]*y.shape[2]))
    ys=y[:1].reshape(1,-1,256)[:,:n].reshape(1,1,n,256)
    hs=h[:1].reshape(1,-1,256,h.shape[-1])[:,:n].reshape(1,1,n,256,h.shape[-1])
    rs=ruu[:1]
    truth_slice=None if truth is None else truth[:1].reshape(
        1,-1,h.shape[-1],qm)[:,:n].reshape(1,1,n,h.shape[-1],qm)
    scale_rows=[];reference=None
    for value in scales:
        a=float(value)
        if not torch.isfinite(torch.tensor(a)) or a<=0: raise ValueError("scales must be finite and positive")
        if deep_candidates:
            candidate,out=_candidate_report(detector,ys*a,hs*a,rs*(a*a),truth_slice,n)
        else:
            out=detector.detect(ys*a,hs*a,rs*(a*a));candidate=None
        if a==1.0: reference=out
        scale_rows.append((a,out,candidate))
    if reference is None:
        if deep_candidates: _,reference=_candidate_report(detector,ys,hs,rs,truth_slice,n)
        else: reference=detector.detect(ys,hs,rs)
    scale_report={}
    for a,out,candidate in scale_rows:
        row={"comparisons_to_scale_1":{name:_pair(detector,reference[name],out[name])
             for name in COMPONENTS}}
        if truth_slice is not None:
            row["ber"]={name:_ber(detector,out[name],truth_slice) for name in COMPONENTS}
        if candidate is not None:
            row["candidate"]={key:value for key,value in candidate.items()
                              if key not in ("components","ber","observable_features")}
            row["observable_features"]=candidate["observable_features"]
        scale_report[str(a)]=row
    report["common_scale_scan"]=scale_report

    _, covariance = covariance_factor(ruu)
    report["ruu_features"]={name:_summary(covariance[:,index]) for index,name in enumerate((
        "mean_eigenvalue","mean_inverse_eigenvalue","condition_number","effective_rank_fraction"))}
    if deep_candidates:
        base_key=next((str(a) for a,_,_ in scale_rows if a==1.0),None)
        if base_key is not None:
            report["candidate_diagnostic"]={**scale_report[base_key].get("candidate",{}),
                "observable_features":scale_report[base_key].get("observable_features",{}),
                "ber":scale_report[base_key].get("ber")}

    if truth_slice is not None:
        ranked=sorted(((row["ber"]["llr"]["overall"],a,row)
                       for a,_,_ in scale_rows for row in (scale_report[str(a)],)),
                      key=lambda item:(item[0],abs(torch.log10(torch.tensor(item[1])).item())))
        best_ber,best_scale,best_row=ranked[0]
        lmmse_ber=scale_report[next((str(a) for a,_,_ in scale_rows if a==1.0),str(scale_rows[0][0]))]["ber"]["lmmse_llr"]["overall"]
        use_flow=best_ber<=lmmse_ber
        report["scale_rescue"]={
            "status":"FLOW_SCALE_FOUND" if use_flow else "FALLBACK_LMMSE",
            "recommended_common_scale":float(best_scale) if use_flow else 1.0,
            "recommended_output":"flow" if use_flow else "lmmse",
            "bounded_lmmse_ber":lmmse_ber,
            "bounded_best_final_ber":best_ber,
            "bounded_best_raw_ber":best_row["ber"]["raw_flow_llr"]["overall"],
            "calibration_batch":0,"calibration_re":n,
            "rule":"use Flow only when best tested final BER is no worse than same-RE LMMSE",
        }

    verdicts=[]
    scale_disagreement=max(row["comparisons_to_scale_1"][name]["hard_disagreement"]
                           for row in report["common_scale_scan"].values()
                           for name in ("raw_flow_llr","llr"))
    if scale_disagreement>1e-3:
        verdicts.append("FAIL_COMMON_SCALE_INVARIANCE: learned inputs depend on the platform's absolute signal units")
    if truth is not None:
        b=report["ber"]
        if b["raw_flow_llr"]["overall"]-b["lmmse_llr"]["overall"]>0.02:
            verdicts.append("FLOW_STAGE_DEGRADES_HARD_BITS: inspect scale statistics and candidate diagnostics")
        if max(b["raw_flow_llr"]["per_bit"])-min(b["raw_flow_llr"]["per_bit"])>0.20:
            verdicts.append("BIT_ASYMMETRY: verify mapper bit order, complements and I/Q convention")
    report["verdicts"]=verdicts or ["NO_AUTOMATIC_FAILURE_FOUND"]
    return report
