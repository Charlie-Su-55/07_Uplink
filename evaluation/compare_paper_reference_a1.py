"""Paired Mode-B practical-CSI evaluation of frozen A1, EP5 and GT/DETR.

Default: complete coded slots and actual TB BLER. --re-per-channel selects a
BER-only diagnostic subset; it never decodes an incomplete codeword.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import time

import torch

from evaluation.evaluate_sionna_bler import (
    add_statistics, block_statistics, ebno_grid, positive_int, provenance, save_json,
)
from link_level.detector_adapter import (
    ClassicalDetectorAdapter, compare_lmmse_outputs, llrs_to_native_order, make_detector_batch,
)
from link_level.partner_a1_adapter import (
    DEFAULT_A1_ROOT, PartnerA1Adapter, load_neural_checkpoint, synchronize,
)
from link_level.sionna_ce import csi_diagnostics, distribution_id
from link_level.sionna_reference import SionnaReferenceLink
from training.train_paper_reference_detector import data_re_statistics, training_config


def selected_native_llrs(llr, ordinals):
    b, k, _, g = llr.shape
    values = llr.reshape(b, k, g // 4, 4).permute(0, 2, 1, 3)
    return values[:, ordinals.to(values.device)].cpu()


def selected_bits(batch, ordinals):
    b, k, g = batch["coded_bits"].shape
    values = batch["coded_bits"].reshape(b, k, g // 4, 4).permute(0, 2, 1, 3)
    return values[:, ordinals.to(values.device)].cpu()


@torch.inference_mode()
def evaluate_neural(models, batch, ordinals, chunk_size):
    parts = {name: [] for name in models}
    timing = dict(shared_frontend_seconds=0., forward_seconds={name: 0. for name in models},
                  scope="CE whitening + z/G once; separate model forward times; excludes TB decode and transfers")
    if not models:
        return {}, timing
    device = batch["y"].device
    for start in range(0, len(ordinals), chunk_size):
        synchronize(device)
        before = time.perf_counter()
        stats = data_re_statistics(batch, ordinals[start:start + chunk_size])
        synchronize(device)
        timing["shared_frontend_seconds"] += time.perf_counter() - before
        for name, model in models.items():
            before = time.perf_counter()
            output = model(stats["z"], stats["gram"], return_iterations=(5,))["llr"]
            synchronize(device)
            timing["forward_seconds"][name] += time.perf_counter() - before
            if output.shape != stats["bits"].shape or not torch.isfinite(output).all():
                raise ValueError(f"Invalid {name} LLRs.")
            parts[name].append(output.cpu())
    # This evaluator deliberately fixes one independent physical channel/batch.
    return {name: torch.cat(values).unsqueeze(0) for name, values in parts.items()}, timing


def add_soft_statistics(total, counts):
    for key in ("bits", "bit_errors", "bce_sum_nats"):
        total[key] = total.get(key, 0) + counts[key]
    total.update(ber=total["bit_errors"] / total["bits"],
                 bce_nats=total["bce_sum_nats"] / total["bits"])
    total["normalized_gmi_s1"] = 1 - total["bce_nats"] / math.log(2)


def summarize_point(point, partner_metrics, coded):
    rows = point["channels"]
    result = {}
    names = list(point["receivers"])
    for base in ("sionna_lmmse", "ep5", "gt_ep", "detr_ep"):
        if base not in names:
            continue
        for candidate in names:
            if candidate == base:
                continue
            key = base + "_minus_" + candidate
            result[key] = dict(coded_bit_ber=partner_metrics.cluster_comparison(
                [r["receivers"][base]["soft"]["ber"] for r in rows],
                [r["receivers"][candidate]["soft"]["ber"] for r in rows]))
            if coded:
                result[key]["tb_bler"] = partner_metrics.cluster_comparison(
                    [r["receivers"][base]["coded"]["bler"] for r in rows],
                    [r["receivers"][candidate]["coded"]["bler"] for r in rows])
    point["paired_comparisons"] = result
    point["comparison_sign"] = "positive base-minus-candidate means candidate has fewer errors"


def save_csv(path, report):
    path = Path(path)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        fields = ("ebno_db", "status", "detector", "channels", "coded_bit_ber", "bce_nats",
                  "normalized_gmi_s1", "tb_bler", "payload_ber", "blocks")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in report["points"]:
            for name, values in point["receivers"].items():
                soft, coded = values["soft"], values.get("coded", {})
                writer.writerow(dict(ebno_db=point["ebno_db"], status=point["status"], detector=name,
                                     channels=len(point["channels"]), coded_bit_ber=soft["ber"],
                                     bce_nats=soft["bce_nats"], normalized_gmi_s1=soft["normalized_gmi_s1"],
                                     tb_bler=coded.get("bler"), payload_ber=coded.get("ber"), blocks=coded.get("blocks")))
    temporary.replace(path)


@torch.inference_mode()
def evaluate_channel(link, classical, partner, models, sample, ordinals, policies, re_chunk, sampling_seed, coded):
    batch = make_detector_batch(link, sample, "practical")
    bits = selected_bits(batch, ordinals)
    synchronize(link.device)
    before = time.perf_counter()
    classical_outputs = classical.evaluate(sample, batch)
    synchronize(link.device)
    timing = dict(classical_group_seconds=time.perf_counter() - before,
                  classical_scope="native/custom LMMSE + EP5, full grid including TB decode",
                  note="execution diagnostics, not a matched standalone latency benchmark; first channel is cold")
    predictions = {name: selected_native_llrs(value["llr"], ordinals) for name, value in classical_outputs.items()}
    neural, timing["neural"] = evaluate_neural(models, batch, ordinals, re_chunk)
    predictions.update(neural)
    for policy in policies:
        outputs, timing[policy] = partner.evaluate(batch, ordinals, policy, sampling_seed)
        prefix = "a1_ce" if policy == "sionna_diagonal" else "a1_thermal"
        predictions.update({prefix + "_k64": outputs["raw_k64"], prefix + "_k256": outputs["raw_k256"],
                            prefix + "_native_lmmse": outputs["lmmse"]})
    receivers = {}
    for name, llr in predictions.items():
        receivers[name] = dict(soft=partner.metrics.soft_metrics(llr, bits))
        if coded:
            if name in classical_outputs:
                received = classical_outputs[name]
            else:
                decoded, crc = link.codec.decode(llrs_to_native_order(llr.to(link.device)))
                received = dict(decoded=decoded, crc_ok=crc)
            counts = block_statistics(sample, received)
            # Retain raw integer counts and channel-level denominators.
            counts.update(bler=counts["block_errors"] / counts["blocks"], ber=counts["bit_errors"] / counts["bits"])
            receivers[name]["coded"] = counts
    delta = None
    if "a1_ce_native_lmmse" in predictions:
        target, actual = predictions["sionna_lmmse"], predictions["a1_ce_native_lmmse"]
        delta = float(((actual - target).square().mean() / target.square().mean().clamp_min(1e-30)).sqrt())
    return dict(seed=sample["seed"], sampling_seed=sampling_seed, data_ordinals=ordinals.tolist(),
                receivers=receivers, timing=timing, csi=csi_diagnostics(batch),
                classical_lmmse_comparison=compare_lmmse_outputs(classical_outputs),
                a1_native_vs_grid_lmmse_llr_relative_rms=delta)


def run(args):
    path = Path(args.output)
    csv_path = path.with_suffix(".csv")
    if path.suffix.lower() != ".json":
        raise ValueError("--output must end in .json; the companion CSV uses the same stem.")
    if path.exists() or csv_path.exists():
        raise FileExistsError("Output JSON/CSV exists; use a new output name.")
    if args.seed < 0 or args.seed + args.channels >= 2 ** 31:
        raise ValueError("Evaluation seeds must lie within [0, 2**31).")
    if args.re_per_channel is not None and args.re_per_channel > 2304:
        raise ValueError("Mode B has only 2304 data REs.")
    # Follow the package's numeric contract; applied equally to every detector.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = training_config(args)
    partner = PartnerA1Adapter(args.a1_root, args.device)
    link = SionnaReferenceLink(cfg)
    if (link.codec.qm, link.codec.coded_bits, link.codec.info_bits) != (4, 9216, 3104):
        raise ValueError("Frozen T1/MCS10 Qm/G/TB contract changed.")
    partner.check_constellation(link)
    seeds = list(range(args.seed, args.seed + args.channels))
    if set(seeds).intersection(link.covariance_metadata["calibration_seeds"]):
        raise ValueError("Evaluation and CE calibration seeds overlap.")
    models, checkpoint_metadata = {}, {}
    for arch, checkpoint in (("gt_ep", args.gt_checkpoint), ("detr_ep", args.detr_checkpoint)):
        if checkpoint:
            models[arch], checkpoint_metadata[arch] = load_neural_checkpoint(checkpoint, arch, link, seeds)
    classical = ClassicalDetectorAdapter(link, custom_ce_policy="sionna_diagonal", re_chunk=args.re_chunk)
    coded = args.re_per_channel is None
    policies = ("sionna_diagonal", "thermal_only") if args.a1_ce_policy == "both" else (args.a1_ce_policy,)
    report = dict(status="running", experiment="Mode_B_practical_A1_transfer_comparison", schema_version=1,
                  provenance=provenance(args), distribution_id=distribution_id(cfg), reference=link.metadata(),
                  a1=partner.metadata, neural_checkpoints=checkpoint_metadata, policies=list(policies),
                  coded_bler_evaluated=coded, selected_data_re=args.re_per_channel or 2304,
                  gt_ep_comparison_included="gt_ep" in models, detr_ep_comparison_included="detr_ep" in models,
                  classical_and_gt_detr_ce_policy="sionna_diagonal",
                  sampling="one independent channel/slot per seed; paired across receivers and Eb/N0",
                  a1_sampling="original K256 with exact K64 RB prefix; seed=channel_seed+3000007",
                  ce_semantics="CE branch sends y/sqrt(N0+sum err_var), Hhat/sqrt(...), I to A1; its internal normalization is unchanged",
                  thermal_control="optional original y/Hhat/N0I; no explicit CE prior for A1 in this auxiliary receiver",
                  validation="fixed channel budget, no checkpoint selection or performance-based early stopping",
                  scientific_limitations=["Frozen A1 Mode-B training provenance is unverified; this is a transfer comparison",
                                          "A1 training seed overlap cannot be checked from the supplied artifact",
                                          "Timing groups have different scopes; do not infer standalone speedups",
                                          "Selected-RE mode reports no coded BLER"], points=[])
    save_json(path, report)
    print(f"Neural controls: {list(models) or 'none (classical vs A1 only)'}; frozen A1 transfer comparison", flush=True)
    try:
        for ebno in args.ebno_dbs:
            point = dict(ebno_db=ebno, status="running", receivers={}, channels=[])
            report["points"].append(point)
            for index, seed in enumerate(seeds):
                sample = link.transmit(1, ebno, seed)
                if "codec_identity" not in report:
                    report["codec_identity"] = link.check_codec_identity(sample)
                if coded:
                    ordinals = torch.arange(len(link.data_indices))  # complete codeword, native order
                else:
                    generator = torch.Generator().manual_seed(seed + 2000003)
                    ordinals = torch.randperm(len(link.data_indices), generator=generator)[:args.re_per_channel]
                row = evaluate_channel(link, classical, partner, models, sample, ordinals,
                                       policies, args.re_chunk, seed + 3000007, coded)
                del sample
                point["channels"].append(row)
                for name, counts in row["receivers"].items():
                    total = point["receivers"].setdefault(name, {"soft": {}})
                    add_soft_statistics(total["soft"], counts["soft"])
                    if coded:
                        add_statistics(total.setdefault("coded", {}), counts["coded"])
                save_json(path, report)
                metric = "BLER" if coded else "coded-bit BER (subset only)"
                rates = " ".join(f"{name}={values['coded']['bler'] if coded else values['soft']['ber']:.5g}"
                                 for name, values in point["receivers"].items())
                print(f"Eb/N0={ebno:g} channels={index + 1}/{args.channels} {metric}: {rates}", flush=True)
            summarize_point(point, partner.metrics, coded)
            point["status"] = "complete"
            save_json(path, report)
            save_csv(csv_path, report)
        report["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_json(path, report)
        save_csv(csv_path, report)
    print(f"Saved {path} and {csv_path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-root", default=DEFAULT_A1_ROOT)
    parser.add_argument("--ce-covariance", required=True)
    parser.add_argument("--gt-checkpoint")
    parser.add_argument("--detr-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channels", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--ebno-dbs", type=ebno_grid, default=ebno_grid("-12,-10.5,-9"))
    parser.add_argument("--re-per-channel", type=positive_int,
                        help="Optional BER-only subset. Omit for all 2304 REs and actual coded BLER.")
    parser.add_argument("--re-chunk", type=positive_int, default=128)
    parser.add_argument("--a1-ce-policy", choices=("sionna_diagonal", "thermal_only", "both"), default="sionna_diagonal")
    parser.add_argument("--output", default="results/paper_reference/a1_comparison.json")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
