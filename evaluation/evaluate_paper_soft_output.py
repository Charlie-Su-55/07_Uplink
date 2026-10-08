"""Fit fixed positive LLR scales on calibration channels and decode cached development TBs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import torch
import torch.nn.functional as functional

from evaluation.evaluate_sionna_bler import save_json
from link_level.detector_adapter import llrs_to_native_order


VARIANTS = ("raw", "global_scale", "per_bit_scale")
LLR_CONVENTION = "log P(bit=1)/P(bit=0); hard decision = LLR > 0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def coded_labels(channel):
    """The final axis is the unchanged mapper bit position, never the UE index."""
    coded = channel["coded_bits"]
    count = channel["data_indices"].numel()
    if coded.ndim != 3 or coded.shape[-1] % count:
        raise ValueError("Incomplete coded bit order or data RE sequence")
    bits_per_symbol = coded.shape[-1] // count
    if bits_per_symbol != 4:
        raise ValueError("This frozen T1 MCS10 experiment requires Qm=4")
    return coded.reshape(*coded.shape[:2], count, bits_per_symbol).permute(0, 2, 1, 3)


def scale_llr(llr, alpha):
    alpha = torch.as_tensor(alpha, dtype=llr.dtype, device=llr.device)
    if alpha.ndim > 1 or alpha.numel() not in (1, llr.shape[-1]):
        raise ValueError("Scale must be global or one value per mapper bit position")
    if not torch.isfinite(alpha).all() or (alpha <= 0).any() or not torch.isfinite(llr).all():
        raise ValueError("Finite LLRs and strictly positive finite scales are required")
    scaled = llr * alpha
    if not torch.isfinite(scaled).all() or not torch.equal(scaled > 0, llr > 0):
        raise ValueError("Scaling changed hard decisions or produced nonfinite LLRs")
    return scaled


def soft_statistics(llr, labels):
    if llr.shape != labels.shape or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("LLR/label shape or coded bit values are invalid")
    signed = llr.double() * (2 * labels.double() - 1)
    loss = functional.softplus(-signed)
    errors = (llr > 0) != labels.bool()
    return dict(coded_bits=labels.numel(), coded_bit_errors=int(errors.sum()),
                bce_sum_nats=float(loss.sum()), bce_nats=float(loss.mean()),
                coded_ber=float(errors.double().mean()),
                fixed_scale_gmi_proxy=1 - float(loss.mean()) / math.log(2),
                per_bit_bce_nats=loss.reshape(-1, loss.shape[-1]).mean(0).tolist())


def fit_positive_scale(signed_llr, settings, per_bit=False):
    """Convex BCE derivative bisection, pooled across every calibration Eb/N0."""
    if settings["method"] != "bounded_derivative_bisection" or settings["precision"] != "float64":
        raise ValueError("Only the predeclared float64 derivative bisection is supported")
    lower, upper = float(settings["alpha_min"]), float(settings["alpha_max"])
    alpha_tolerance = float(settings["alpha_tolerance"])
    gradient_tolerance = float(settings["gradient_tolerance"])
    max_iterations = int(settings["max_iterations"])
    if not (0 < lower < upper and alpha_tolerance > 0 and gradient_tolerance > 0 and max_iterations > 0):
        raise ValueError("Invalid fixed temperature bounds or stopping conditions")
    values = signed_llr.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not torch.isfinite(values).all():
        raise ValueError("Expected finite nonempty [coded symbols, Qm] signed LLRs")
    dimensions = values.shape[-1] if per_bit else 1
    left = torch.full((dimensions,), lower, dtype=torch.float64)
    right = torch.full((dimensions,), upper, dtype=torch.float64)

    def derivative(alpha):
        gradients = -values * torch.sigmoid(-values * alpha)
        return gradients.mean(0) if per_bit else gradients.mean().reshape(1)

    at_left, at_right = derivative(left), derivative(right)
    at_lower = at_left >= 0
    at_upper = (at_right <= 0) & ~at_lower
    alpha = (left + right) / 2
    alpha[at_lower], alpha[at_upper] = lower, upper
    active = ~(at_lower | at_upper)
    reasons = ["lower_boundary" if at_lower[index] else "upper_boundary" if at_upper[index]
               else "max_iterations" for index in range(dimensions)]
    history = []
    for iteration in range(max_iterations):
        gradient = derivative(alpha)
        history.append(dict(iteration=iteration, alpha=alpha.tolist(), derivative=gradient.tolist(),
                            bracket_width=(right - left).tolist()))
        stopped_gradient = active & (gradient.abs() <= gradient_tolerance)
        stopped_width = active & ((right - left) <= alpha_tolerance)
        for index in range(dimensions):
            if stopped_gradient[index]:
                reasons[index] = "gradient_tolerance"
            elif stopped_width[index]:
                reasons[index] = "alpha_tolerance"
        active &= ~(stopped_gradient | stopped_width)
        if not active.any():
            break
        left = torch.where(active & (gradient < 0), alpha, left)
        right = torch.where(active & (gradient >= 0), alpha, right)
        alpha = torch.where(active, (left + right) / 2, alpha)
    final_gradient = derivative(alpha)
    return dict(alpha=alpha.tolist(), per_bit=bool(per_bit), stop_reason=reasons,
                boundary_hit=(at_lower | at_upper).tolist(), iterations=len(history),
                derivative=final_gradient.tolist(), fit_history=history,
                bce_nats=float(functional.softplus(-values * alpha).mean()),
                raw_bce_nats=float(functional.softplus(-values).mean()),
                coded_bits=values.numel(), precision="float64")


def fit_cache_temperatures(manifest, loader, settings):
    """Only calibration records are loaded; development labels are inaccessible here."""
    records = [record for record in manifest["channels"] if record["split"] == "calibration"]
    if not records or any(record["status"] != "complete" for record in records):
        raise ValueError("Complete calibration channels are required")
    started = time.monotonic()
    signed = {}
    methods = None
    for index, record in enumerate(records):
        channel_started = time.monotonic()
        channel = loader(record)
        if channel["split"] != "calibration":
            raise ValueError("Development labels cannot enter temperature fitting")
        labels = coded_labels(channel)
        if methods is None:
            methods = tuple(channel["methods"])
            signed = {method: [] for method in methods}
        if set(methods) != set(channel["methods"]):
            raise ValueError("Receiver set changes inside calibration cache")
        for method in methods:
            llr = channel["methods"][method]["llr"]
            if llr.shape != labels.shape or llr.dtype != torch.float32:
                raise ValueError("Calibration requires full float32 mapper-order LLRs")
            signed[method].append((llr * (2 * labels - 1)).reshape(-1, labels.shape[-1]))
        print(f"fit read calibration {index + 1}/{len(records)} seed={record['seed']} Eb/N0={record['ebno_db']:g} "
              f"elapsed={time.monotonic() - channel_started:.1f}s total={time.monotonic() - started:.1f}s", flush=True)
    fitted = {}
    for method in methods:
        started = time.monotonic()
        print(f"fit {method} START float64 bounded derivative bisection", flush=True)
        values = torch.cat(signed.pop(method), dim=0)
        fitted[method] = dict(global_scale=fit_positive_scale(values, settings),
                              per_bit_scale=fit_positive_scale(values, settings, per_bit=True))
        print(f"fit {method} elapsed={time.monotonic() - started:.1f}s "
              f"alpha={fitted[method]['global_scale']['alpha']} "
              f"alpha_q={fitted[method]['per_bit_scale']['alpha']}", flush=True)
    return dict(methods=fitted, fit_split="calibration", pooled_ebno_dbs=sorted({r["ebno_db"] for r in records}),
                independent_channel_seeds=sorted({r["seed"] for r in records}),
                channel_ebno_observations=len(records), settings=settings,
                fit_sources=[dict(file=r["file"], sha256=r["sha256"]) for r in records],
                llr_convention=LLR_CONVENTION, bit_axis="original Qm mapper bit position; shared across UEs",
                boundary_policy="Report fixed-boundary optima; never expand bounds using development results")


def payload_statistics(info_bits, decoded, crc_ok):
    if decoded.shape != info_bits.shape or crc_ok.shape != info_bits.shape[:-1]:
        raise ValueError("Decoder returned incompatible complete-TB shapes")
    bad_bits = decoded != info_bits
    bit_errors = bad_bits.sum(-1)
    block_error = bit_errors > 0
    crc_ok = crc_ok.bool()
    per_ue = dict(decoded_payload_bits=decoded.detach().cpu().contiguous(),
                  crc_ok=crc_ok.detach().cpu().contiguous(),
                  payload_bit_errors=bit_errors.cpu(), block_error=block_error.cpu(),
                  total_payload_bits_per_ue=torch.full_like(bit_errors, info_bits.shape[-1]).cpu())
    totals = dict(blocks=block_error.numel(), block_errors=int(block_error.sum()),
                  payload_bits=info_bits.numel(), payload_bit_errors=int(bit_errors.sum()),
                  crc_failures=int((~crc_ok).sum()),
                  undetected_block_errors=int((block_error & crc_ok).sum()),
                  crc_failures_without_payload_error=int((~block_error & ~crc_ok).sum()))
    return per_ue, totals


@torch.inference_mode()
def evaluate_channel(channel, temperatures, codec, device="cpu"):
    labels = coded_labels(channel)
    outcomes, rows = {}, []
    if set(channel["methods"]) != set(temperatures["methods"]):
        raise ValueError("Cache receiver identities differ from fitted temperatures")
    for method, cached in channel["methods"].items():
        llr = cached["llr"]
        if not torch.equal(llrs_to_native_order(llr), cached["decoder_input_llr"]):
            raise ValueError(f"{method}: cached native/evaluator LLR order mismatch")
        scales = dict(raw=[1.0], **{variant: temperatures["methods"][method][variant]["alpha"]
                                  for variant in VARIANTS[1:]})
        outcomes[method] = {}
        raw_blocks = None
        for variant in VARIANTS:
            scaled = scale_llr(llr, scales[variant])
            native = llrs_to_native_order(scaled)
            decoded, crc_ok = codec.decode(native.to(device))
            decoded, crc_ok = decoded.cpu(), crc_ok.cpu()
            per_ue, totals = payload_statistics(channel["info_bits"], decoded, crc_ok)
            if variant == "raw":
                if not torch.equal(scaled, llr) or not torch.equal(native, cached["decoder_input_llr"]):
                    raise ValueError("alpha=1 changed original LLRs")
                for field in per_ue:
                    if not torch.equal(per_ue[field], cached[field]):
                        raise ValueError(f"{method}: cached raw re-decode differs in {field}")
                raw_blocks = per_ue["block_error"]
            rescue = raw_blocks & ~per_ue["block_error"]
            harm = ~raw_blocks & per_ue["block_error"]
            per_ue.update(rescue_vs_raw=rescue, harm_vs_raw=harm)
            outcomes[method][variant] = per_ue
            rows.append(dict(split=channel["split"], seed=channel["seed"], ebno_db=channel["ebno_db"],
                             method=method, variant=variant, alpha=scales[variant], **totals,
                             **soft_statistics(scaled, labels), rescue_vs_raw=int(rescue.sum()),
                             harm_vs_raw=int(harm.sum()), alpha_one_raw_identity_passed=True,
                             per_ue={field: value.tolist() for field, value in per_ue.items()
                                     if field != "decoded_payload_bits"}))
    return outcomes, rows


def aggregate(rows):
    fields = ("coded_bits", "coded_bit_errors", "bce_sum_nats", "blocks", "block_errors",
              "payload_bits", "payload_bit_errors", "crc_failures", "undetected_block_errors",
              "crc_failures_without_payload_error", "rescue_vs_raw", "harm_vs_raw")
    totals = {field: sum(row[field] for row in rows) for field in fields}
    totals.update(independent_channels=len({row["seed"] for row in rows}), channel_ebno_observations=len(rows),
                  coded_ber=totals["coded_bit_errors"] / totals["coded_bits"],
                  bce_nats=totals["bce_sum_nats"] / totals["coded_bits"],
                  bler=totals["block_errors"] / totals["blocks"],
                  payload_ber=totals["payload_bit_errors"] / totals["payload_bits"])
    totals["fixed_scale_gmi_proxy"] = 1 - totals["bce_nats"] / math.log(2)
    return totals


def paired_bootstrap(rows, baseline, settings):
    """Resample independent seed clusters; all Eb/N0 observations follow their seed."""
    lookup = {(row["seed"], row["ebno_db"]): row for row in baseline}
    if len(lookup) != len(baseline) or len(rows) != len(baseline):
        raise ValueError("Paired bootstrap needs one complete counterpart per channel/EbNo")
    clustered = {}
    seen = set()
    for row in rows:
        key = (row["seed"], row["ebno_db"])
        if key in seen or key not in lookup or row["blocks"] != lookup[key]["blocks"]:
            raise ValueError("Bootstrap realizations/denominators are not paired")
        seen.add(key)
        values = clustered.setdefault(row["seed"], [0, 0])
        values[0] += row["block_errors"] - lookup[key]["block_errors"]
        values[1] += row["blocks"]
    if not clustered:
        raise ValueError("Empty bootstrap sample")
    replicates = int(settings["replicates"])
    confidence = float(settings["confidence"])
    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid fixed bootstrap configuration")
    values = torch.tensor([clustered[seed] for seed in sorted(clustered)], dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(settings["seed"]))
    draws = torch.randint(len(values), (replicates, len(values)), generator=generator)
    resampled = values[draws].sum(1)
    delta = resampled[:, 0] / resampled[:, 1]
    tail = (1 - confidence) / 2
    interval = torch.quantile(delta, torch.tensor([tail, 1 - tail], dtype=torch.float64)).tolist()
    return dict(delta_bler=float(values[:, 0].sum() / values[:, 1].sum()), confidence=confidence,
                interval=interval, independent_channels=len(clustered), channel_ebno_observations=len(rows),
                replicates=replicates, bootstrap_seed=int(settings["seed"]),
                resampling_unit="channel seed; all UEs and all Eb/N0 points move together",
                limitation="Small channel samples give uncertain percentile intervals; development comparison, not final paper test",
                interval_informative=len(clustered) >= 2)


def summarize(rows, settings):
    groups, paired, checkpoint_pairs = [], [], []
    methods = sorted({row["method"] for row in rows})
    points = [None] + sorted({row["ebno_db"] for row in rows})
    for ebno in points:
        selected = [row for row in rows if ebno is None or row["ebno_db"] == ebno]
        for method in methods:
            by_variant = {variant: [row for row in selected if row["method"] == method and row["variant"] == variant]
                          for variant in VARIANTS}
            for variant in VARIANTS:
                groups.append(dict(method=method, variant=variant, ebno_db=ebno, **aggregate(by_variant[variant])))
            for variant in VARIANTS[1:]:
                paired.append(dict(method=method, variant=variant, baseline="raw", ebno_db=ebno,
                                   **paired_bootstrap(by_variant[variant], by_variant["raw"], settings)))
        for architecture in ("gt", "detr"):
            for variant in VARIANTS:
                best = [row for row in selected if row["method"] == architecture + "_best" and row["variant"] == variant]
                last = [row for row in selected if row["method"] == architecture + "_last" and row["variant"] == variant]
                if best and last:
                    checkpoint_pairs.append(dict(architecture=architecture, variant=variant, ebno_db=ebno,
                                                 comparison="last minus best", **paired_bootstrap(last, best, settings)))
    return dict(metrics=groups, calibration_vs_raw=paired, best_vs_last=checkpoint_pairs,
                interpretation="BCE improvement does not imply BLER improvement. Calibration is a baseline, not a claimed new paper contribution.",
                gmi_definition="1 - mean coded-bit BCE(nats)/ln(2), at the saved fixed scale; no development optimization or clipping")


def atomic_torch_save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite decoded channel: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def codec_from_cache(channel, manifest, device):
    from link_level.nr_codec import NRTransportBlockCodec

    reference = manifest["identity"]["reference"]
    metadata = reference["codec"]
    codec = NRTransportBlockCodec(num_data_symbols=channel["data_indices"].numel(),
                                 num_ues=channel["info_bits"].shape[1], table=metadata["mcs_table"],
                                 index=metadata["mcs_index"], num_bp_iter=metadata["num_bp_iter"],
                                 precision="single", device=device)
    if codec.metadata() != metadata:
        raise ValueError("Runtime TB codec metadata differs from the frozen cache")
    return codec


def run(args):
    from evaluation.paper_soft_common import (
        load_experiment_config, load_complete_cache, load_channel, run_paths, numeric_runtime, source_identity,
    )

    configuration = load_experiment_config(args.config, mode=args.mode)
    paths = run_paths(configuration, args.run_id)
    cache_dir, results_dir, calibration_dir = (paths[name] for name in ("cache", "results", "calibration"))
    manifest = load_complete_cache(cache_dir)
    if manifest["identity"]["experiment_config"] != configuration:
        raise ValueError("Cache experiment configuration differs from the requested fixed configuration")
    torch.set_num_threads(configuration["runtime"]["torch_threads"])
    identity = manifest["identity_id"]
    loader = lambda record: load_channel(cache_dir, record, identity)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    temperature_path = calibration_dir / "temperatures.json"
    fit_identity = digest(dict(cache_identity=identity, settings=configuration["calibration"],
                               files=[(r["file"], r["sha256"]) for r in manifest["channels"] if r["split"] == "calibration"]))
    if args.stage in ("fit", "all"):
        if temperature_path.exists():
            temperatures = json.loads(temperature_path.read_text(encoding="utf-8"))
            if not args.reuse_results or temperatures.get("status") != "complete" or temperatures.get("fit_identity") != fit_identity:
                raise FileExistsError("Temperature artifact exists; only explicit identical complete reuse is allowed")
            print(f"Reusing complete fitted temperatures {temperature_path}", flush=True)
        else:
            temperatures = dict(schema_version=1, status="running", identity_id=identity, fit_identity=fit_identity,
                                cache_identity=manifest["identity"])
            save_json(temperature_path, temperatures)
            try:
                temperatures.update(fit_cache_temperatures(manifest, loader, configuration["calibration"]))
                temperatures["status"] = "complete"
                temperatures["artifact_digest"] = digest(temperatures)
                save_json(temperature_path, temperatures)
            except (Exception, KeyboardInterrupt) as error:
                temperatures.update(status="failed", error=f"{type(error).__name__}: {error}")
                save_json(temperature_path, temperatures)
                raise
    else:
        temperatures = json.loads(temperature_path.read_text(encoding="utf-8"))
    if temperatures.get("status") != "complete" or temperatures.get("fit_identity") != fit_identity:
        raise ValueError("Fitted scales do not match this exact cache and calibration configuration")
    if temperatures.get("artifact_digest") != digest({key: value for key, value in temperatures.items() if key != "artifact_digest"}):
        raise ValueError("Fitted temperature artifact checksum mismatch")
    if args.stage == "fit":
        print(f"Saved calibration artifact {temperature_path}", flush=True)
        return 0
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"
    evaluation_environment = numeric_runtime(configuration)
    evaluation_source = source_identity()
    cache_manifest_hash = file_hash(cache_dir / "manifest.json")
    evaluation_identity = digest(dict(cache_identity=identity, cache_manifest_sha256=cache_manifest_hash,
                                      temperature_sha256=file_hash(temperature_path),
                                      bootstrap=configuration["bootstrap"], environment=evaluation_environment,
                                      source=evaluation_source))
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if not args.reuse_results or previous.get("status") != "complete" or previous.get("evaluation_identity") != evaluation_identity:
            raise FileExistsError("Results exist; use a new run-id or explicitly reuse identical complete results")
        for artifact in previous["artifacts"]:
            artifact_path = (results_dir / artifact["file"]).resolve()
            if not artifact_path.is_relative_to(results_dir.resolve()):
                raise ValueError("Result artifact path escapes the results directory")
            if file_hash(artifact_path) != artifact["sha256"]:
                raise ValueError("Result artifact hash mismatch")
        print(f"Reusing complete results {summary_path}", flush=True)
        return 0
    summary = dict(schema_version=1, status="running", identity_id=identity, evaluation_identity=evaluation_identity,
                   cache_manifest=str(cache_dir / "manifest.json"), cache_manifest_sha256=cache_manifest_hash,
                   temperatures=str(temperature_path), temperatures_sha256=file_hash(temperature_path),
                   alpha=temperatures["methods"], llr_convention=LLR_CONVENTION,
                   llr_processing="Cast alpha to cached float32 LLR dtype; multiply in mapper q order; native permute(0,2,1,3).reshape(B,UE,1,G); no bias or clipping",
                   rescue_harm_unit="Per UE transport block, based on decoded payload errors",
                   split="development", mode=args.mode, artifacts=[], completed_channels=0,
                   environment=evaluation_environment, source=evaluation_source,
                   cache_runtime_matches=manifest["identity"]["environment"] == evaluation_environment)
    save_json(summary_path, summary)
    started = time.monotonic()
    try:
        records = [record for record in manifest["channels"] if record["split"] == "development"]
        if not records:
            raise ValueError("No development channels")
        decoded_dir = results_dir / "decoded_channels"
        decoded_dir.mkdir(exist_ok=False)
        rows = []
        codec = None
        device = configuration["runtime"]["device"]
        for index, record in enumerate(records):
            channel_started = time.monotonic()
            channel = loader(record)
            if channel["split"] != "development":
                raise ValueError("Unexpected development record split")
            if codec is None:
                codec = codec_from_cache(channel, manifest, device)
            outcomes, channel_rows = evaluate_channel(channel, temperatures, codec, device)
            output_path = decoded_dir / Path(record["file"]).name
            atomic_torch_save(output_path, dict(schema_version=1, status="complete", identity_id=identity,
                                              split="development", seed=record["seed"], ebno_db=record["ebno_db"],
                                              methods=outcomes, axes=dict(decoded_payload_bits=["batch", "UE", "payload_bit"],
                                                                       all_other_fields=["batch", "UE"])))
            artifact = dict(file=str(output_path.relative_to(results_dir)).replace("\\", "/"),
                            sha256=file_hash(output_path), bytes=output_path.stat().st_size)
            summary["artifacts"].append(artifact)
            for row in channel_rows:
                row["decoded_payload_artifact"] = artifact["file"]
            rows.extend(channel_rows)
            summary["completed_channels"] = index + 1
            save_json(summary_path, summary)
            print(f"decode development {index + 1}/{len(records)} seed={record['seed']} Eb/N0={record['ebno_db']:g} "
                  f"elapsed={time.monotonic() - channel_started:.1f}s total={time.monotonic() - started:.1f}s raw identity PASS", flush=True)
        summary.update(summarize(rows, configuration["bootstrap"]))
        for name in ("channels.jsonl", "summary.csv"):
            target = results_dir / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                if name.endswith("jsonl"):
                    for row in rows:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                else:
                    writer = csv.DictWriter(handle, fieldnames=list(summary["metrics"][0]))
                    writer.writeheader()
                    writer.writerows(summary["metrics"])
            temporary.replace(target)
            summary["artifacts"].append(dict(file=name, sha256=file_hash(target), bytes=target.stat().st_size))
        summary.update(status="complete", elapsed_seconds=time.monotonic() - started,
                       all_raw_redecode_checks_passed=True, all_positive_scale_hard_decisions_unchanged=True)
        save_json(summary_path, summary)
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(summary_path, summary)
        raise
    print(f"Complete development evaluation: {summary_path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/paper_soft_output_dev.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), default="development")
    parser.add_argument("--stage", choices=("fit", "evaluate", "all"), default="all")
    parser.add_argument("--reuse-results", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
