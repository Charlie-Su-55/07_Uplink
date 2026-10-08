"""Compare fixed cached-EP LLR calibrators on complete, paired NR codewords."""

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from evaluation.evaluate_paper_soft_output import (
    aggregate, atomic_torch_save, coded_labels, codec_from_cache, digest, file_hash,
    llrs_to_native_order, paired_bootstrap, payload_statistics, scale_llr,
    soft_statistics,
)
from evaluation.evaluate_sionna_bler import save_json


BASELINES = ("ep5_raw", "global_train24", "per_bit_train24")
DEFAULT_CONFIG = "configs/evaluation/cached_llr_scaling.yaml"


def contained_file(root, name, expected_hash):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected_hash:
        raise ValueError(f"Missing, escaped, or changed artifact: {name}")
    return path


def source_identity(context):
    return context["cache_manifest"]["identity_id"]


def training_artifacts(training_dir, context, config, load_models=True):
    from training.train_cached_llr_scaling import reuse_manifest

    manifest_path = Path(training_dir) / "manifest.json"
    manifest = reuse_manifest(training_dir, context, config)
    if manifest.get("status") != "complete" or manifest.get("run_identity_id") != context["run_identity_id"]:
        raise ValueError("All candidate training must complete under the exact current run identity before evaluation")
    expected = {(candidate, seed) for candidate in config["candidates"] for seed in config["training"]["seeds"]}
    actual = [(record["candidate"], record["training_seed"]) for record in manifest["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Training manifest has missing, duplicate, or unexpected candidates/seeds")
    receivers = [dict(method="ep5_raw", candidate="ep5", training_seed=None, checkpoint_type="raw", step=None,
                      parameter_count=0, alpha=[1.])]
    for method, variant in zip(BASELINES[1:], ("global_scale", "per_bit_scale")):
        receivers.append(dict(method=method, candidate=method, training_seed=None, checkpoint_type="fixed", step=None,
                              parameter_count=1 if variant == "global_scale" else 4,
                              alpha=manifest["constants"][variant]["alpha"]))
    artifacts = [dict(file="manifest.json", sha256=file_hash(manifest_path))]
    for record in manifest["records"]:
        for kind in ("best", "last"):
            artifact = record[kind]
            path = contained_file(training_dir, artifact["file"], artifact["sha256"])
            artifacts.append(dict(file=artifact["file"], sha256=artifact["sha256"]))
            receiver = dict(method=f"{record['candidate']}_seed{record['training_seed']}_{kind}",
                            candidate=record["candidate"], training_seed=record["training_seed"],
                            checkpoint_type=kind, step=artifact["step"], parameter_count=record["parameter_count"],
                            checkpoint_file=artifact["file"], checkpoint_sha256=artifact["sha256"])
            if load_models:
                from models.conditional_llr_scaling import make_model

                checkpoint = torch.load(path, weights_only=True, map_location="cpu")
                expected_metadata = dict(candidate=record["candidate"], training_seed=record["training_seed"],
                                         step=artifact["step"], run_identity_id=context["run_identity_id"],
                                         source_cache_identity_id=source_identity(context))
                if any(checkpoint.get(key) != value for key, value in expected_metadata.items()):
                    raise ValueError(f"Checkpoint provenance differs from completed training: {path}")
                if checkpoint.get("normalization") != manifest["normalization"]:
                    raise ValueError("Checkpoint normalization differs from the train-only frozen normalization")
                model = make_model(record["candidate"], config).to(config["runtime"]["device"])
                model.load_state_dict(checkpoint["model_state"], strict=True)
                if sum(parameter.numel() for parameter in model.parameters()) != record["parameter_count"]:
                    raise ValueError("Checkpoint parameter count does not match its declared architecture")
                receiver["model"] = model.eval().requires_grad_(False)
            receivers.append(receiver)
    return manifest, receivers, artifacts


def alpha_statistics(alpha, config):
    values = alpha.detach().cpu().float().reshape(-1, 4)
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Conditional scaling must remain finite and strictly positive")
    bounds = config["model"]
    tolerance = bounds["saturation_tolerance"]
    if (values < bounds["alpha_min"]).any() or (values > bounds["alpha_max"]).any():
        raise ValueError("Conditional alpha exceeded the predeclared fixed bounds")
    return dict(count=values.numel(), sum=float(values.double().sum()), min=float(values.min()), max=float(values.max()),
                mean=float(values.double().mean()), per_bit_mean=values.double().mean(0).tolist(),
                lower_count=int((values <= bounds["alpha_min"] + tolerance).sum()),
                upper_count=int((values >= bounds["alpha_max"] - tolerance).sum()))


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_channel(channel, receivers, codec, config, normalization, feature_function, normalize_function):
    from models.conditional_llr_scaling import scale_llr as conditional_scale

    device = config["runtime"]["device"]
    cached = channel["methods"]["ep5"]
    raw = cached["llr"]
    if raw.dtype != torch.float32 or raw.shape[0] != 1:
        raise ValueError("Require one complete float32 codeword batch")
    if not torch.equal(llrs_to_native_order(raw), cached["decoder_input_llr"]):
        raise ValueError("Cached EP5 LLR/codeword ordering mismatch")
    labels = coded_labels(channel)
    synchronize(device)
    feature_started = time.perf_counter()
    nodes, edges = feature_function(channel, config)
    nodes = normalize_function(nodes, normalization, config).to(device)
    edges = edges.to(device)
    synchronize(device)
    feature_seconds = time.perf_counter() - feature_started
    results, rows = {}, []
    raw_blocks = None
    if not receivers or receivers[0]["method"] != "ep5_raw":
        raise ValueError("Raw EP5 must establish the paired reference first")
    for receiver in receivers:
        synchronize(device)
        infer_started = time.perf_counter()
        if "model" in receiver:
            alphas = []
            for start in range(0, len(nodes), config["evaluation_chunk"]):
                stop = start + config["evaluation_chunk"]
                alphas.append(receiver["model"](nodes[start:stop], edges[start:stop]).cpu())
            alpha = torch.cat(alphas, 0)
            scaled = conditional_scale(raw[0], alpha).unsqueeze(0)
        else:
            scaled = scale_llr(raw, receiver["alpha"])
            alpha = torch.as_tensor(receiver["alpha"], dtype=torch.float32).expand_as(raw[0])
        synchronize(device)
        inference_seconds = time.perf_counter() - infer_started
        if scaled.dtype != raw.dtype or scaled.shape != raw.shape or not torch.equal(scaled > 0, raw > 0):
            raise ValueError("Positive calibration changed raw EP5 hard decisions, dtype, or codeword shape")
        if not torch.isfinite(scaled).all():
            raise ValueError("Nonfinite conditional LLRs")
        boundary = alpha_statistics(alpha, config)
        native = llrs_to_native_order(scaled)
        synchronize(device)
        decode_started = time.perf_counter()
        decoded, crc = codec.decode(native.to(device))
        synchronize(device)
        decode_seconds = time.perf_counter() - decode_started
        outcomes, totals = payload_statistics(channel["info_bits"], decoded.cpu(), crc.cpu())
        if receiver["method"] == "ep5_raw":
            if not torch.equal(scaled, raw) or not torch.equal(native, cached["decoder_input_llr"]):
                raise ValueError("alpha=1 did not preserve cached LLRs exactly")
            for field, value in outcomes.items():
                if not torch.equal(value, cached[field]):
                    raise ValueError(f"Raw cached EP5 re-decode differs in {field}")
            raw_blocks = outcomes["block_error"]
        rescue = raw_blocks & ~outcomes["block_error"]
        harm = ~raw_blocks & outcomes["block_error"]
        outcomes.update(rescue_vs_raw=rescue, harm_vs_raw=harm)
        results[receiver["method"]] = outcomes
        descriptor = {key: value for key, value in receiver.items() if key not in ("model", "alpha")}
        rows.append(dict(**descriptor, seed=channel["seed"], ebno_db=channel["ebno_db"], **totals,
                         **soft_statistics(scaled, labels), rescue_vs_raw=int(rescue.sum()), harm_vs_raw=int(harm.sum()),
                         alpha=boundary, inference_seconds=inference_seconds, decode_seconds=decode_seconds,
                         raw_alpha_one_identity_passed=True, positive_scale_hard_decisions_unchanged=True,
                         per_ue={key: value.tolist() for key, value in outcomes.items() if key != "decoded_payload_bits"}))
    return results, rows, feature_seconds


def verify_initial_identity(channel, codec, config, normalization, feature_function, normalize_function):
    from models.conditional_llr_scaling import make_model

    initial = [dict(method="ep5_raw", candidate="ep5", training_seed=None, checkpoint_type="raw", step=None,
                    parameter_count=0, alpha=[1.])]
    for candidate in config["candidates"]:
        model = make_model(candidate, config).to(config["runtime"]["device"]).eval().requires_grad_(False)
        initial.append(dict(method=candidate, candidate=candidate, training_seed=None, checkpoint_type="initial", step=0,
                            parameter_count=sum(parameter.numel() for parameter in model.parameters()), model=model))
    outcomes, rows, _ = evaluate_channel(channel, initial, codec, config, normalization, feature_function, normalize_function)
    checked = {}
    for row in rows[1:]:
        method = row["method"]
        if row["alpha"]["min"] != 1. or row["alpha"]["max"] != 1.:
            raise ValueError(f"Fresh {method} did not initialize to exact alpha=1")
        for field in ("decoded_payload_bits", "crc_ok", "payload_bit_errors", "block_error"):
            if not torch.equal(outcomes[method][field], outcomes["ep5_raw"][field]):
                raise ValueError(f"Fresh alpha-one {method} changed the full-codeword decoder result")
        checked[method] = True
    return dict(candidates=checked, seed=channel["seed"], ebno_db=channel["ebno_db"],
                full_codeword=True, exact_alpha_one=True, decoded_payload_and_crc_identical=True)


def paired_comparison(rows, baseline, config):
    bce_rows = [dict(row, blocks=row["coded_bits"], block_errors=row["bce_sum_nats"]) for row in rows]
    bce_baseline = [dict(row, blocks=row["coded_bits"], block_errors=row["bce_sum_nats"]) for row in baseline]
    bce = paired_bootstrap(bce_rows, bce_baseline, config["bootstrap"])
    bce["delta_bce_nats"] = bce.pop("delta_bler")
    return dict(bler=paired_bootstrap(rows, baseline, config["bootstrap"]), bce=bce)


def summarize(rows, config):
    metrics, comparisons = [], []
    descriptors = {row["method"]: {key: row[key] for key in ("method", "candidate", "training_seed", "checkpoint_type", "step", "parameter_count")}
                   for row in rows}
    for ebno in [None] + sorted({row["ebno_db"] for row in rows}):
        grouped = {method: [row for row in rows if row["method"] == method and (ebno is None or row["ebno_db"] == ebno)]
                   for method in descriptors}
        for method, selected in grouped.items():
            alpha_count = sum(row["alpha"]["count"] for row in selected)
            metrics.append(dict(**descriptors[method], ebno_db=ebno, **aggregate(selected),
                                alpha_mean=sum(row["alpha"]["sum"] for row in selected) / alpha_count,
                                alpha_min=min(row["alpha"]["min"] for row in selected),
                                alpha_max=max(row["alpha"]["max"] for row in selected),
                                alpha_lower_fraction=sum(row["alpha"]["lower_count"] for row in selected) / alpha_count,
                                alpha_upper_fraction=sum(row["alpha"]["upper_count"] for row in selected) / alpha_count,
                                inference_seconds=sum(row["inference_seconds"] for row in selected),
                                decode_seconds=sum(row["decode_seconds"] for row in selected)))
            baselines = BASELINES if method not in BASELINES else BASELINES[:1]
            for baseline in baselines:
                if method != baseline:
                    comparisons.append(dict(method=method, baseline=baseline, category="ep5_baseline", ebno_db=ebno,
                                            **paired_comparison(selected, grouped[baseline], config)))
            candidate = descriptors[method]["candidate"]
            other = {"graph": "mlp", "mlp": "affine"}.get(candidate)
            if other:
                baseline = f"{other}_seed{descriptors[method]['training_seed']}_{descriptors[method]['checkpoint_type']}"
                comparisons.append(dict(method=method, baseline=baseline, category=f"{candidate}_vs_{other}", ebno_db=ebno,
                                        **paired_comparison(selected, grouped[baseline], config)))
            if candidate == "mlp":
                baseline = f"mlp_no_residual_eta_seed{descriptors[method]['training_seed']}_{descriptors[method]['checkpoint_type']}"
                comparisons.append(dict(method=method, baseline=baseline, category="mlp_vs_no_residual_eta", ebno_db=ebno,
                                        **paired_comparison(selected, grouped[baseline], config)))
            if descriptors[method]["checkpoint_type"] == "last":
                baseline = method.removesuffix("last") + "best"
                comparisons.append(dict(method=method, baseline=baseline, category="last_vs_best", ebno_db=ebno,
                                        **paired_comparison(selected, grouped[baseline], config)))
    seed_summaries = []
    for candidate in config["candidates"]:
        for kind in ("best", "last"):
            for ebno in [None] + sorted({row["ebno_db"] for row in rows}):
                selected = [item for item in metrics if item["candidate"] == candidate and item["checkpoint_type"] == kind and item["ebno_db"] == ebno]
                summary = dict(candidate=candidate, checkpoint_type=kind, ebno_db=ebno,
                               training_seeds=[item["training_seed"] for item in selected],
                               independent_channels=selected[0]["independent_channels"],
                               channel_ebno_observations_per_training_seed=selected[0]["channel_ebno_observations"],
                               note="Training seeds are repeated models on the same channel set, not extra independent channels")
                for field in ("bler", "bce_nats", "coded_ber", "payload_ber", "fixed_scale_gmi_proxy"):
                    values = [item[field] for item in selected]
                    summary[field] = dict(mean=sum(values) / len(values), min=min(values), max=max(values))
                seed_summaries.append(summary)
    decisions = []
    decision_groups = [(category, category, None, None) for category in
                       ("graph_vs_mlp", "mlp_vs_affine", "mlp_vs_no_residual_eta")]
    decision_groups += [(f"{candidate}_vs_{baseline}", "ep5_baseline", candidate, baseline)
                        for candidate in config["candidates"] for baseline in BASELINES[1:]]
    for label, category, candidate, baseline in decision_groups:
        for kind in ("best", "last"):
            selected = [item for item in comparisons if item["category"] == category and item["ebno_db"] is None
                        and descriptors[item["method"]]["checkpoint_type"] == kind
                        and (candidate is None or descriptors[item["method"]]["candidate"] == candidate)
                        and (baseline is None or item["baseline"] == baseline)]
            deltas = [dict(training_seed=descriptors[item["method"]]["training_seed"],
                           delta_bler=item["bler"]["delta_bler"], delta_bce_nats=item["bce"]["delta_bce_nats"],
                           bler_interval=item["bler"]["interval"], bce_interval=item["bce"]["interval"],
                           confidence=item["bler"]["confidence"],
                           bce_reduced_without_bler_reduction=item["bce"]["delta_bce_nats"] < 0 and item["bler"]["delta_bler"] >= 0)
                      for item in selected]
            decisions.append(dict(comparison=label, checkpoint_type=kind, per_training_seed=deltas,
                                  training_seed_count=len(deltas),
                                  bler_improved_seeds=sum(item["delta_bler"] < 0 for item in deltas),
                                  bce_improved_seeds=sum(item["delta_bce_nats"] < 0 for item in deltas),
                                  bler_worsened_seeds=sum(item["delta_bler"] > 0 for item in deltas),
                                  bce_worsened_seeds=sum(item["delta_bce_nats"] > 0 for item in deltas),
                                  bler_tied_seeds=sum(item["delta_bler"] == 0 for item in deltas),
                                  bce_tied_seeds=sum(item["delta_bce_nats"] == 0 for item in deltas),
                                  all_three_seeds_bler_improved=len(deltas) == 3 and all(item["delta_bler"] < 0 for item in deltas),
                                  all_three_seeds_bce_improved=len(deltas) == 3 and all(item["delta_bce_nats"] < 0 for item in deltas)))
    return dict(metrics=metrics, paired_comparisons=comparisons, training_seed_summary=seed_summaries,
                decision_summary=dict(comparisons=decisions,
                    primary_checkpoint="best selected on the fixed validation split; last reported separately",
                    interpretation="BCE and BLER directions are separate empirical findings. Bootstrap intervals are descriptive; no significance or final independent paper-test claim.",
                    feasibility_split=dict(train=24, validation=8, development=64),
                    limitations=["Only 64 development channel seeds, shared across three Eb/N0 points and all trained models.",
                                 "Three training seeds do not triple the number of independent development channels.",
                                 "Multiple predeclared candidates/comparisons have no multiple-comparison correction or significance claim.",
                                 "Selecting a future method after inspecting this development report creates selection bias; a new independent paper test is required.",
                                 "Smoke uses one validation channel only and cannot establish cross-seed performance."],
                    protocol="Fixed candidate set, bounds and training seeds. No development refitting, checkpoint selection, or threefold inflation of channel count."))


def historical_comparisons(context, config, rows):
    if config["mode"] == "smoke":
        return dict(status="not_evaluated", reason="Smoke uses validation only; historical comparison is development-only")
    root = Path(config["source_archive"])
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    temperatures = json.loads((root / "temperatures.json").read_text(encoding="utf-8"))
    if summary["identity_id"] != source_identity(context) or temperatures["identity_id"] != source_identity(context):
        raise ValueError("Archived comparison does not describe the exact current historical cache identity")
    archived = []
    with (root / "channels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["method"] in ("gt_best", "gt_last", "detr_best", "detr_last", "ep5"):
                archived.append(row)
    expected = {(row["seed"], row["ebno_db"]) for row in rows}
    comparisons = []
    for method in ("gt_best", "gt_last", "detr_best", "detr_last", "ep5"):
        for variant in ("raw", "global_scale", "per_bit_scale"):
            selected = [row for row in archived if row["method"] == method and row["variant"] == variant]
            coordinates = [(row["seed"], row["ebno_db"]) for row in selected]
            if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
                raise ValueError("Archived receiver rows are not paired to the exact current development channels")
            for ebno in [None] + sorted({row["ebno_db"] for row in rows}):
                baseline = [row for row in selected if ebno is None or row["ebno_db"] == ebno]
                new_methods = list(BASELINES)
                if method in ("gt_best", "detr_best") and variant == "per_bit_scale":
                    new_methods += sorted({row["method"] for row in rows if row["checkpoint_type"] == "best"})
                for new_method in new_methods:
                    current = [row for row in rows if row["method"] == new_method and (ebno is None or row["ebno_db"] == ebno)]
                    comparisons.append(dict(method=new_method, baseline=f"archived_32cal/{method}/{variant}", ebno_db=ebno,
                                            **paired_comparison(current, baseline, config)))
    return dict(status="complete", label="archived_32cal: historical scales fitted on all original 32 calibration channels",
                usage="Context only. Historical scales never initialize, fit, or select new train24 models/constants.",
                source_cache_identity_id=source_identity(context),
                sources={name: dict(path=str(root / name), sha256=file_hash(root / name))
                         for name in ("summary.json", "temperatures.json", "channels.jsonl")},
                metrics=[item for item in summary["metrics"] if item["method"] in ("gt_best", "gt_last", "detr_best", "detr_last", "ep5")],
                paired_comparisons=comparisons)


def validate_complete(results_dir, expected_identity, training_dir, expected_cases, expected_methods, candidates):
    path = Path(results_dir) / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("evaluation_identity") != expected_identity:
        raise ValueError("Result is incomplete or belongs to another run/training identity")
    if summary.get("artifact_digest") != digest({key: value for key, value in summary.items() if key != "artifact_digest"}):
        raise ValueError("Summary content checksum mismatch")
    if (summary.get("completed_channels") != len(expected_cases)
            or summary.get("channel_ebno_observations") != len(expected_cases)
            or summary.get("independent_channels") != len({seed for seed, _ in expected_cases})
            or summary.get("all_raw_redecode_checks_passed") is not True
            or summary.get("all_positive_scale_hard_decisions_unchanged") is not True):
        raise ValueError("Complete result counts or mandatory raw/positive-scale checks are missing")
    initial = summary.get("initial_alpha_one_identity", {})
    if (initial.get("candidates") != {candidate: True for candidate in candidates}
            or initial.get("full_codeword") is not True or initial.get("exact_alpha_one") is not True
            or initial.get("decoded_payload_and_crc_identical") is not True):
        raise ValueError("Fresh alpha-one full-codeword qualification is missing")
    artifact_names = [artifact["file"] for artifact in summary["artifacts"]]
    table_names = {"channels.jsonl", "summary.csv", "decision_summary.json"}
    if len(artifact_names) != len(set(artifact_names)) or not table_names.issubset(artifact_names):
        raise ValueError("Complete result artifact coverage is missing tables or has duplicate filenames")
    for artifact in summary["artifacts"]:
        contained_file(results_dir, artifact["file"], artifact["sha256"])
    with (Path(results_dir) / "channels.jsonl").open(encoding="utf-8") as handle:
        channel_rows = [json.loads(line) for line in handle]
    expected_rows = {(seed, ebno, method) for seed, ebno in expected_cases for method in expected_methods}
    actual_rows = [(row["seed"], row["ebno_db"], row["method"]) for row in channel_rows]
    if len(actual_rows) != len(set(actual_rows)) or set(actual_rows) != expected_rows:
        raise ValueError("Complete result has missing, duplicated, or unexpected per-channel receiver rows")
    decoded_by_case = {}
    for row in channel_rows:
        decoded_path = row.get("decoded_payload_artifact", "")
        if decoded_path not in artifact_names or not decoded_path.startswith("decoded_channels/"):
            raise ValueError("Complete result artifact coverage omits a per-channel decoded payload")
        decoded_by_case.setdefault((row["seed"], row["ebno_db"]), set()).add(decoded_path)
    decoded_names = set().union(*decoded_by_case.values())
    if (any(len(names) != 1 for names in decoded_by_case.values()) or len(decoded_names) != len(expected_cases)
            or set(artifact_names) != table_names | decoded_names):
        raise ValueError("Complete result artifact coverage requires one unique decoded payload file per channel")
    for artifact in summary["training_artifacts"]:
        contained_file(training_dir, artifact["file"], artifact["sha256"])
    return summary


def run(args):
    from evaluation.cached_llr_common import (
        feature_tensors, load_config, normalize_nodes, prepare, records_for, run_paths,
    )
    from evaluation.paper_soft_common import load_channel

    config = load_config(args.config, mode=args.mode)
    context = prepare(config, require_gpu=not args.check_complete)
    paths = run_paths(config, args.run_id)
    training_dir, results_dir = paths["training"], paths["results"]
    trained, receivers, training_files = training_artifacts(training_dir, context, config, load_models=False)
    identity = digest(dict(run_identity_id=context["run_identity_id"], training_artifacts=training_files,
                           bootstrap=config["bootstrap"], evaluation_chunk=config["evaluation_chunk"]))
    evaluation_split = "validation" if args.mode == "smoke" else "development"
    records = records_for(context, evaluation_split)
    expected_cases = {(record["seed"], record["ebno_db"]) for record in records}
    summary_path = results_dir / "summary.json"
    if args.check_complete or (args.reuse_complete and summary_path.exists()):
        validate_complete(results_dir, identity, training_dir, expected_cases,
                          {receiver["method"] for receiver in receivers}, config["candidates"])
        print(f"Complete cached-scaling artifacts verified: {summary_path}", flush=True)
        return 0
    if results_dir.exists():
        raise FileExistsError("Result directory already exists; no incomplete or unrelated outputs are overwritten")
    results_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=1, status="running", evaluation_identity=identity, run_identity_id=context["run_identity_id"],
                   source_cache_identity_id=source_identity(context), source_cache=context["source_cache"],
                   new_code_identity=context["code_identity"], environment=context["environment"],
                   training_artifacts=training_files, mode=args.mode, split=evaluation_split, experiment_config=config,
                   constants_train24=trained["constants"], normalization=trained["normalization"],
                   training_runs=[dict(candidate=record["candidate"], training_seed=record["training_seed"],
                                       parameter_count=record["parameter_count"], elapsed_seconds=record.get("elapsed_seconds"),
                                       best_step=record["best"]["step"], last_step=record["last"]["step"])
                                  for record in trained["records"]],
                   artifacts=[], completed_channels=0, llr_sign="log P(bit=1)/P(bit=0); hard decision >0",
                   decoder_mapping="float32 positive alpha times cached EP5 LLR[B,N,UE,Qm]; permute(0,2,1,3).reshape(B,UE,1,G); no LLR bias/clipping",
                   timing_definition="Wall-clock synchronized CUDA inference and TB decode separately; shared feature construction measured per channel. Cached scale work only; excludes original PHY/CE/EP/detector inference and is not full receiver speedup.")
    save_json(summary_path, summary)
    started = time.perf_counter()
    try:
        trained, receivers, _ = training_artifacts(training_dir, context, config, load_models=True)
        if not records:
            raise ValueError("No complete codewords in the requested evaluation split")
        decoded_dir = results_dir / "decoded_channels"
        decoded_dir.mkdir()
        rows, codec = [], None
        for index, record in enumerate(records):
            channel_started = time.perf_counter()
            print(f"evaluate {summary['split']} {index + 1}/{len(records)} seed={record['seed']} Eb/N0={record['ebno_db']:g} START", flush=True)
            channel = load_channel(context["cache_dir"], record, source_identity(context))
            if codec is None:
                codec = codec_from_cache(channel, context["cache_manifest"], config["runtime"]["device"])
                summary["initial_alpha_one_identity"] = verify_initial_identity(
                    channel, codec, config, trained["normalization"], feature_tensors, normalize_nodes)
            outcomes, channel_rows, feature_seconds = evaluate_channel(channel, receivers, codec, config,
                trained["normalization"], feature_tensors, normalize_nodes)
            path = decoded_dir / Path(record["file"]).name
            atomic_torch_save(path, dict(schema_version=1, status="complete", evaluation_identity=identity,
                                        seed=record["seed"], ebno_db=record["ebno_db"], split=summary["split"],
                                        methods=outcomes, axes=dict(decoded_payload_bits=["batch", "UE", "payload_bit"],
                                                                  other_outcomes=["batch", "UE"])))
            artifact = dict(file=path.relative_to(results_dir).as_posix(), sha256=file_hash(path), bytes=path.stat().st_size)
            summary["artifacts"].append(artifact)
            for row in channel_rows:
                row.update(decoded_payload_artifact=artifact["file"], split=summary["split"], shared_feature_seconds=feature_seconds)
            rows.extend(channel_rows)
            summary["completed_channels"] = index + 1
            save_json(summary_path, summary)
            print(f"evaluate {index + 1}/{len(records)} COMPLETE elapsed={time.perf_counter() - channel_started:.2f}s "
                  f"total={time.perf_counter() - started:.2f}s raw identity PASS", flush=True)
        summary.update(summarize(rows, config))
        summary["archived_32cal"] = historical_comparisons(context, config, rows)
        for name in ("channels.jsonl", "summary.csv", "decision_summary.json"):
            path = results_dir / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                if name == "channels.jsonl":
                    for row in rows:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                elif name == "summary.csv":
                    writer = csv.DictWriter(handle, fieldnames=list(summary["metrics"][0]))
                    writer.writeheader()
                    writer.writerows(summary["metrics"])
                else:
                    json.dump(summary["decision_summary"], handle, indent=2, allow_nan=False)
            temporary.replace(path)
            summary["artifacts"].append(dict(file=name, sha256=file_hash(path), bytes=path.stat().st_size))
        summary.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                       all_raw_redecode_checks_passed=True, all_positive_scale_hard_decisions_unchanged=True,
                       independent_channels=len({row["seed"] for row in rows}),
                       channel_ebno_observations=len({(row["seed"], row["ebno_db"]) for row in rows}))
        summary["artifact_digest"] = digest(summary)
        save_json(summary_path, summary)
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(summary_path, summary)
        raise
    print(f"Complete cached scaling evaluation: {summary_path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), default="development")
    parser.add_argument("--reuse-complete", action="store_true")
    parser.add_argument("--check-complete", action="store_true", help="Verify identities and artifact hashes without GPU inference or decoding")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
