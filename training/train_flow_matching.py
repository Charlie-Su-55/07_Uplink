"""Train the masked Flow control on practical uplink statistics (GPU server)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from models.baselines.flow_matching_detector import FlowMatchingDetector


@torch.no_grad()
def validate(model, ep, validation, device, chunk_size=32):
    model.eval()
    errors = dict(flow_matching=0, raw_flow=0, lmmse=0, ep5=0)
    bce = dict(flow_matching=0.0, raw_flow=0.0, lmmse=0.0, ep5=0.0)
    count = 0
    for start in range(0, len(validation["bits"]), chunk_size):
        z = validation["z_lmmse"][start:start + chunk_size].to(device)
        gram = validation["gram_lmmse"][start:start + chunk_size].to(device)
        bits = validation["bits"][start:start + chunk_size].to(device)
        out = model(z, gram)
        llrs = dict(flow_matching=out["llr"], raw_flow=out["raw_flow_llr"],
                    lmmse=out["lmmse_llr"], ep5=ep(z, gram, return_iterations=(5,))["llr"])
        for name, llr in llrs.items():
            if llr.shape != bits.shape or not torch.isfinite(llr).all():
                raise RuntimeError(f"Invalid {name} LLR shape or nonfinite output")
            errors[name] += int(((llr > 0) != (bits > .5)).sum())
            bce[name] += float(F.binary_cross_entropy_with_logits(llr, bits, reduction="sum"))
        count += bits.numel()
    return {"total_bits": count, "errors": errors,
            **{f"ber_{k}": v / count for k, v in errors.items()},
            **{f"bce_{k}": v / count for k, v in bce.items()}}


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--bits-per-symbol", type=int, choices=[2, 4, 6], default=4)
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--snr-min-db", type=float)
    parser.add_argument("--snr-max-db", type=float)
    parser.add_argument("--val-snr-db", type=float)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--re-per-step", type=int, default=128)
    parser.add_argument("--val-channels", type=int, default=32)
    parser.add_argument("--val-re-per-channel", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-chunk", type=int, default=16)
    parser.add_argument("--re-chunk", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=1729)
    parser.add_argument("--proposal-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcs-table", type=int, choices=[1, 2])
    parser.add_argument("--mcs-index", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    for name in ("steps", "re_per_step", "val_channels", "val_re_per_channel", "val_every",
                 "num_samples", "sample_chunk", "re_chunk", "lr", "grad_clip", "proposal_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative and finite")
    if (args.mcs_table is None) != (args.mcs_index is None):
        parser.error("Use --mcs-table and --mcs-index together")
    if args.mcs_table is not None:
        from link_level.nr_mcs import get_pusch_mcs
        if get_pusch_mcs(args.mcs_table, args.mcs_index).bits_per_symbol != args.bits_per_symbol:
            parser.error("MCS identity does not match --bits-per-symbol")
    output = Path(args.output_dir)
    paths = [output / name for name in ("best.pth", "last.pth", "history.json")]
    if not args.fresh and any(path.exists() for path in paths):
        raise FileExistsError("Flow output exists; choose a new --output-dir or explicitly use --fresh")

    # Deferred imports keep --help usable without the server's radio simulator.
    from sionna.phy import config as sionna_config
    from training.train_gt_detr_lmmse import (load_yaml, resolve_snr_range, build_snr_schedule,
                                             sample_re, build_validation_set)
    from data.dataset import UplinkUMADataset
    from detectors.classical.ep import ExpectationPropagationDetector
    snr_min, snr_max, val_snr = resolve_snr_range(args)
    if not all(math.isfinite(x) for x in (snr_min, snr_max, val_snr)):
        parser.error("SNR values must be finite")
    schedule = build_snr_schedule(args.steps, snr_min, snr_max, args.seed)
    cfg = load_yaml(load_yaml(args.config)["system_config"])
    cfg["modulation"]["bits_per_symbol"] = args.bits_per_symbol
    cfg["link"]["rx_snr_db"] = val_snr
    cfg["general"]["seed"] = args.seed
    device = cfg["general"]["device"]
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Run 256Rx generation and training on the GPU server")
    sionna_config.device, sionna_config.precision = device, cfg["general"]["precision"]
    if cfg["general"]["precision"] != "single":
        raise ValueError("This Flow control currently trains in single precision")
    torch.set_float32_matmul_precision("high")
    sionna_config.seed = args.seed
    torch.manual_seed(args.seed)
    dataset = UplinkUMADataset(cfg)
    torch.manual_seed(args.seed + 100)
    model = FlowMatchingDetector(cfg, num_samples=args.num_samples, sample_chunk=args.sample_chunk,
                                  re_chunk=args.re_chunk, sampling_seed=args.sampling_seed).to(device)
    ep = ExpectationPropagationDetector(cfg, num_iterations=5, damping=.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .05)
    sionna_config.seed = args.seed + 2000
    torch.manual_seed(args.seed + 2000)
    validation = build_validation_set(dataset, args.val_channels, args.val_re_per_channel)
    initial = validate(model, ep, validation, device)
    z, g = (validation[key][:8].to(device) for key in ("z_lmmse", "gram_lmmse"))
    with torch.no_grad():
        anchor = model(z, g)
        if not torch.equal(anchor["llr"], anchor["lmmse_llr"]):
            raise RuntimeError("Flow zero residual failed to reproduce APP LMMSE")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    cache = Path("data/cache/uma_lmmse_ft_cov.pt")
    metadata = dict(arch="flow_matching", implementation_id=model.implementation_id,
                    model_config=model.model_config, bits_per_symbol=args.bits_per_symbol,
                    csi="lmmseH", covariance="estimated_Ruu", system_config=cfg, args=vars(args),
                    snr_min_db=snr_min, snr_max_db=snr_max, val_snr_db=val_snr,
                    selection_metric="bce_flow_matching", source_commit=commit, source_dirty=dirty,
                    torch_version=str(torch.__version__), covariance_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    if args.mcs_table is not None:
        metadata.update(mcs_table=args.mcs_table, mcs_index=args.mcs_index)
    output.mkdir(parents=True, exist_ok=True)
    history = [{"step": 0, **initial}]
    best_bce, best_step = initial["bce_flow_matching"], 0

    def payload(step, metrics):
        return dict(metadata, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(), step=step, best_bce=best_bce,
                    best_step=best_step, metrics=metrics)

    def save_history():
        temporary = output / "history.tmp"
        temporary.write_text(json.dumps(dict(metadata, history=history), indent=2), encoding="utf-8")
        temporary.replace(output / "history.json")

    save_checkpoint(paths[0], payload(0, initial))
    save_history()
    print(f"Flow parameters={sum(p.numel() for p in model.parameters()):,}, K={args.num_samples}; initial={initial}", flush=True)
    for step, snr in enumerate(schedule, 1):
        dataset.link.rx_snr_db = snr
        physical_seed = args.seed * 100000 + step
        sionna_config.seed = physical_seed
        torch.manual_seed(physical_seed)
        with torch.no_grad():
            sample = sample_re(dataset, args.re_per_step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model.training_loss(sample["z_lmmse"], sample["gram_lmmse"], sample["bits"], args.proposal_weight)
        if not torch.isfinite(out["loss"]):
            raise RuntimeError(f"Nonfinite Flow loss at step {step}")
        out["loss"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 25 == 0 or step == 1 or step == args.steps:
            print(f"step={step} SNR={snr:.2f} BCE={out['bce'].item():.6f} CE={out['proposal_ce'].item():.6f} grad={float(grad):.3f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(model, ep, validation, device)
            history.append(dict(step=step, lr=optimizer.param_groups[0]["lr"], **metrics))
            if metrics["bce_flow_matching"] < best_bce:
                best_bce, best_step = metrics["bce_flow_matching"], step
                save_checkpoint(paths[0], payload(step, metrics))
            save_checkpoint(paths[1], payload(step, metrics))
            save_history()
            print(f"VAL step={step} best_step={best_step} {metrics}", flush=True)
    print(f"Saved {paths[0]} (held-out BCE={best_bce:.6f}, step={best_step})")


if __name__ == "__main__":
    main()
