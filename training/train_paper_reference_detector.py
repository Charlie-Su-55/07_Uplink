"""Fresh GT/DETR-EP training on the frozen Mode-B practical-CSI waveform.

Run from the repository root with python -m training.train_paper_reference_detector.
Validation measures pre-decoder coded-bit BER on fixed data REs, not TB BLER.
There is deliberately no checkpoint-loading path; legacy weights are never used.
"""

import argparse
import copy
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from data.preprocessing.whitening import CovarianceAwareFrontEnd
from evaluation.evaluate_sionna_bler import positive_int, provenance, save_json
from link_level.detector_adapter import make_detector_batch, whiten_diagonal_uncertainty
from link_level.sionna_ce import distribution_id
from link_level.sionna_reference import PAPER_CONFIG, SionnaReferenceLink, load_config, validate_reference_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gt_ep", "detr_ep"), required=True)
    parser.add_argument("--ce-covariance", required=True, help="Existing independent Mode-B covariance; never rebuilt here.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ebno-min-db", type=float, default=-13.)
    parser.add_argument("--ebno-max-db", type=float, default=-9.)
    parser.add_argument("--val-ebno-db", type=float, default=-10.5)
    parser.add_argument("--steps", type=positive_int, default=300)
    parser.add_argument("--re-per-step", type=positive_int, default=128)
    parser.add_argument("--val-channels", type=positive_int, default=32)
    parser.add_argument("--val-re-per-channel", type=positive_int, default=64)
    parser.add_argument("--val-every", type=positive_int, default=25)
    parser.add_argument("--val-chunk", type=positive_int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, help="New directory for best.pth, last.pth and history.json.")
    args = parser.parse_args(argv)
    for key in ("ebno_min_db", "ebno_max_db", "val_ebno_db", "lr", "weight_decay", "grad_clip"):
        if not math.isfinite(getattr(args, key)):
            parser.error(f"--{key.replace('_', '-')} must be finite")
    if args.ebno_min_db > args.ebno_max_db:
        parser.error("--ebno-min-db must not exceed --ebno-max-db")
    if args.seed < 0 or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("Require seed >= 0, lr > 0, weight-decay >= 0 and grad-clip > 0")
    if max(args.re_per_step, args.val_re_per_channel) > 2304:
        parser.error("Mode B has only 2304 data REs per channel")
    return args


def training_config(args):
    # No alternate grid/MCS/CSI CLI: only execution seed, device and prior path vary.
    cfg = copy.deepcopy(load_config(PAPER_CONFIG))
    cfg["general"].update(device=args.device, seed=args.seed)
    cfg["channel_estimation"].update(mode="practical", covariance_path=args.ce_covariance)
    grid = validate_reference_config(cfg)
    if (cfg.get("mode") != "paper_reference" or
            tuple(grid[key] for key in ("num_ues", "num_rx", "num_tx_antennas", "num_ofdm_symbols", "fft_size"))
            != (16, 256, 4, 14, 192)):
        raise ValueError("Training requires the frozen 16UE/256Rx Mode-B profile.")
    return cfg


def make_model(cfg, arch):
    # Identical constructor settings to train_gt_detr_lmmse; no legacy data import.
    common = dict(cfg=cfg, num_users=16, num_iterations=5, damping=.5, d_model=128,
                  num_heads=8, ffn_dim=256, dropout=.05, max_logit_correction=4.)
    if arch == "gt_ep":
        from models.graph.gt_ep_detector import GraphTransformerEPDetector
        return GraphTransformerEPDetector(num_layers=4, edge_dim=32, edge_mode="full",
                                          message_mode="cross_user", **common)
    if arch == "detr_ep":
        from models.baselines.detr_ep_detector import DETREPDetector
        return DETREPDetector(num_layers=3, **common)
    raise ValueError(f"Unknown architecture: {arch}")


@torch.no_grad()
def data_re_statistics(batch, ordinals, bits_per_symbol=4):
    """Select data/codeword ordinals, then reuse the classical practical frontend."""
    if batch["metadata"]["csi"] != "practical":
        raise ValueError("Neural training requires practical CSI.")
    b, t, f, m = batch["y"].shape
    k = batch["h_hat"].shape[-1]
    data_indices = batch["data_indices"]
    ordinals = ordinals.to(device=data_indices.device, dtype=torch.long)
    idx = data_indices[ordinals]
    y = batch["y"].reshape(b, t * f, m)[:, idx].unsqueeze(1)
    h = batch["h_hat"].reshape(b, t * f, m, k)[:, idx].unsqueeze(1)
    error = batch["err_var"].reshape(b, t * f, m, k)[:, idx].unsqueeze(1)
    y, h = whiten_diagonal_uncertainty(y, h, error, batch["n0"])
    identity = torch.eye(m, dtype=h.dtype, device=h.device).expand(b, -1, -1)
    stats = CovarianceAwareFrontEnd()(y, h, identity)
    # coded_bits includes TB scrambling/rate matching; payload bits are not labels.
    bits = batch["coded_bits"].reshape(b, k, len(data_indices), bits_per_symbol)
    bits = bits.permute(0, 2, 1, 3)[:, ordinals]
    return dict(z=stats["z"].reshape(-1, k), gram=stats["gram"].reshape(-1, k, k),
                bits=bits.reshape(-1, k, bits_per_symbol).float())


@torch.no_grad()
def sample_re(link, ebno_db, seed, count):
    if link.mode != "paper_reference" or link.csi != "practical":
        raise ValueError("Training requires Mode-B practical CSI.")
    sample = link.transmit(1, ebno_db, seed)
    batch = make_detector_batch(link, sample, "practical")
    # Separate CPU generator keeps selection independent of architecture/dropout.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ordinals = torch.randperm(len(batch["data_indices"]), generator=generator)[:count]
    return data_re_statistics(batch, ordinals, link.codec.qm)


def sampling_schedule(args, calibration_seeds):
    """Disjoint calibration/train/validation channels; paired across architectures."""
    rng = random.Random(args.seed)
    used = set(calibration_seeds)
    seeds = []
    while len(seeds) < args.val_channels + args.steps:
        seed = rng.randrange(2 ** 31)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    ebno_rng = random.Random(args.seed + 314159)
    ebnos = [ebno_rng.uniform(args.ebno_min_db, args.ebno_max_db) for _ in range(args.steps)]
    return seeds[args.val_channels:], seeds[:args.val_channels], ebnos


def bit_loss(llr, bits):
    """LLR = log(P1/P0); positive logits target bit 1, with no sign reversal."""
    if llr.shape != bits.shape:
        raise ValueError("LLR and coded-bit label shapes differ.")
    loss = F.binary_cross_entropy_with_logits(llr, bits)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite bit BCE.")
    return loss


def hard_errors(llr, bits):
    return int(((llr > 0) != (bits > .5)).sum().item())


@torch.no_grad()
def validate(model, ep, validation, device, chunk_size, check_anchor=False):
    model.eval()
    errors_ep = errors_neural = total_bits = total_re = 0
    diagnostics = {}
    anchor_max = 0.
    for start in range(0, len(validation["bits"]), chunk_size):
        chunk = {key: value[start:start + chunk_size].to(device) for key, value in validation.items()}
        z, gram, bits = chunk["z"], chunk["gram"], chunk["bits"]
        baseline = ep(z, gram, return_iterations=(5,))["llr"]
        output = model(z, gram, return_iterations=(5,))
        llr = output["llr"]
        if not torch.isfinite(llr).all() or not torch.isfinite(baseline).all():
            raise RuntimeError("Non-finite validation LLRs.")
        if check_anchor:
            anchor_max = max(anchor_max, float((llr - baseline).abs().max().item()))
            if anchor_max > 1e-5:
                raise RuntimeError(f"Fresh model violates EP5 anchor: max LLR difference={anchor_max:.6e}")
        errors_ep += hard_errors(baseline, bits)
        errors_neural += hard_errors(llr, bits)
        total_bits += bits.numel()
        total_re += len(bits)
        for key in ("correction_rms", "valid_update_fraction", "residual_rms"):
            if key in output:
                value = output[key].double().cpu()
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite {key}.")
                diagnostics[key] = diagnostics.get(key, 0.) + len(bits) * (value.square() if key.endswith("rms") else value)
    ep_ber, neural_ber = errors_ep / total_bits, errors_neural / total_bits
    metrics = dict(ber_ep5=ep_ber, ber_neural=neural_ber, bits=total_bits,
                   errors_ep5=errors_ep, errors_neural=errors_neural,
                   relative_ber_change=(neural_ber - ep_ber) / ep_ber if ep_ber else None)
    for key, value in diagnostics.items():
        mean = value / total_re
        metrics[key] = (mean.sqrt() if key.endswith("rms") else mean).tolist()
    if check_anchor:
        metrics["anchor_max_abs_llr_difference"] = anchor_max
    return metrics


def checkpoint_metadata(args, link, source):
    covariance = link.covariance_metadata
    if link.mode != "paper_reference" or link.csi != "practical" or not covariance:
        raise ValueError("Checkpoint requires Mode-B practical CSI and covariance provenance.")
    if covariance.get("distribution_id") != distribution_id(link.cfg) or not covariance.get("sha256"):
        raise ValueError("Legacy/mismatched covariance provenance cannot label a Mode-B checkpoint.")
    return dict(kind="paper_reference_detector_v1", arch=args.arch, bits_per_symbol=link.codec.qm,
                mcs_table=link.cfg["nr"]["mcs_table"], mcs_index=link.cfg["nr"]["mcs_index"],
                train_ebno_min_db=args.ebno_min_db, train_ebno_max_db=args.ebno_max_db,
                val_ebno_db=args.val_ebno_db, csi="practical", custom_ce_policy="sionna_diagonal",
                ce_covariance_path=covariance["path"], ce_covariance_sha256=covariance["sha256"],
                distribution_id=distribution_id(link.cfg), git_commit=source["git_commit"],
                seed=args.seed, provenance=source, system_config=link.cfg,
                grid=link.grid_metadata, codec=link.codec.metadata(),
                validation_metric="pre_decoder_coded_bit_ber", llr_convention="log(P1/P0)")


def save_checkpoint(path, model, optimizer, metadata, step, metrics):
    payload = dict(metadata, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                   step=step, validation_metric_value=metrics["ber_neural"], metrics=metrics)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Choose a new run directory.")
    cfg = training_config(args)
    link = SionnaReferenceLink(cfg)  # validates the existing covariance before any waveform generation
    if (link.codec.qm, link.codec.coded_bits, link.codec.info_bits) != (4, 9216, 3104):
        raise ValueError("Frozen T1/MCS10 Qm/G/TB contract changed.")
    metadata = checkpoint_metadata(args, link, provenance(args))
    train_seeds, val_seeds, ebnos = sampling_schedule(args, link.covariance_metadata["calibration_seeds"])
    metadata.update(train_seeds=train_seeds, validation_seeds=val_seeds, train_ebno_schedule=ebnos)
    model_cfg = copy.deepcopy(cfg)
    model_cfg["modulation"] = {"bits_per_symbol": link.codec.qm}
    torch.manual_seed(args.seed)
    model = make_model(model_cfg, args.arch).to(args.device)
    from detectors.classical.ep import ExpectationPropagationDetector
    ep = ExpectationPropagationDetector(model_cfg, num_iterations=5, damping=.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(metadata=metadata, status="running", history=[])
    save_json(output_dir / "history.json", report)
    try:
        storage = {key: [] for key in ("z", "gram", "bits")}
        for i, seed in enumerate(val_seeds):
            sample = sample_re(link, args.val_ebno_db, seed, args.val_re_per_channel)
            for key in storage:
                storage[key].append(sample[key].cpu())
            print(f"Fixed practical validation: {i + 1}/{len(val_seeds)} channels", flush=True)
        validation = {key: torch.cat(values) for key, values in storage.items()}
        del storage, sample
        initial = validate(model, ep, validation, args.device, args.val_chunk, check_anchor=True)
        print(f"Fresh EP5 anchor PASS: max |LLR difference|={initial['anchor_max_abs_llr_difference']:.6e}", flush=True)
        best_ber = float("inf")
        running_loss = 0.
        interval_steps = 0
        torch.manual_seed(args.seed)
        for step in range(args.steps + 1):
            if step:
                model.train()
                sample = sample_re(link, ebnos[step - 1], train_seeds[step - 1], args.re_per_step)
                optimizer.zero_grad(set_to_none=True)
                output = model(sample["z"], sample["gram"], return_iterations=(5,))
                loss = bit_loss(output["llr"], sample["bits"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                running_loss += loss.item()
                interval_steps += 1
                del sample, output, loss
                if step % args.val_every and step != args.steps:
                    continue
            metrics = initial if step == 0 else validate(model, ep, validation, args.device, args.val_chunk)
            metrics = dict(metrics, training_loss=running_loss / interval_steps if interval_steps else None)
            report["history"].append(dict(step=step, **metrics))
            change = metrics["relative_ber_change"]
            change_text = "n/a (EP5 BER=0)" if change is None else f"{100 * change:+.3f}%"
            print(f"VAL step={step} BCE={metrics['training_loss']} EP5={metrics['ber_ep5']:.6e} "
                  f"neural={metrics['ber_neural']:.6e} relative_BER_change={change_text} "
                  f"correction_rms={metrics.get('correction_rms')} "
                  f"valid_update_fraction={metrics.get('valid_update_fraction')}", flush=True)
            save_checkpoint(output_dir / "last.pth", model, optimizer, metadata, step, metrics)
            if metrics["ber_neural"] < best_ber:
                best_ber = metrics["ber_neural"]
                report.update(best_step=step, best_ber=best_ber)
                save_checkpoint(output_dir / "best.pth", model, optimizer, metadata, step, metrics)
            save_json(output_dir / "history.json", report)
            running_loss, interval_steps = 0., 0
        report["status"] = "complete"
        save_json(output_dir / "history.json", report)
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_json(output_dir / "history.json", report)
        raise
    print(f"Saved {output_dir}; best coded-bit BER={best_ber:.6e} at step={report['best_step']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
