"""Stream the fixed independent Mode-B confirmation budget after all controls are frozen."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from evaluation.build_paper_soft_cache import posterior_residual, shared_statistics, tb_record
from evaluation.cached_llr_common import feature_tensors, normalize_nodes
from evaluation.evaluate_cached_llr_scaling import alpha_statistics, contained_file, synchronize
from evaluation.evaluate_paper_soft_output import (
    coded_labels, codec_from_cache, digest, file_hash, llrs_to_native_order,
    payload_statistics, scale_llr, soft_statistics,
)
from evaluation.evaluate_sionna_bler import save_json
from link_level.detector_adapter import make_detector_batch
from models.conditional_llr_scaling import scale_llr as conditional_scale
from models.llr_mechanism_controls import interpolate_noise_alpha, make_model


DEFAULT_CONFIG = "configs/evaluation/llr_mechanism_controls.yaml"
BASELINES = ("native_lmmse", "ep5_raw", "global_train24", "per_bit_train24", "noise_per_bit_train24")


def tensor_identity(value):
    tensor = value.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())


def scale_distribution(alpha, config):
    statistics = alpha_statistics(alpha, config)
    settings = config["scale_distribution"]
    if settings["convention"] != "left_closed_right_open_last_closed":
        raise ValueError("Unexpected prespecified scale histogram convention")
    edges = torch.tensor(settings["bin_edges"], dtype=torch.float64)
    counts, _ = torch.histogram(alpha.detach().cpu().double().reshape(-1), bins=edges)
    if int(counts.sum()) != statistics["count"]:
        raise ValueError("Scale histogram failed to account for every positive alpha")
    statistics.update(histogram_edges=edges.tolist(), histogram_counts=counts.to(torch.int64).tolist())
    return statistics


def case_name(seed, ebno):
    token = f"{ebno:g}".replace("-", "m").replace(".", "p")
    return f"cases/ebno_{token}_seed_{seed}.json"


@torch.inference_mode()
def stream_observation(link, ep, seed, ebno, config, check_codec=False):
    """One waveform and actual CE estimate; frozen native objects and EP core do the math."""
    device = link.device
    settings = config["receiver"]
    timings = {}
    synchronize(device)
    started = time.perf_counter()
    sample = link.transmit(1, ebno, seed)
    synchronize(device)
    timings["transmit_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    batch = make_detector_batch(link, sample, "practical")
    synchronize(device)
    timings["ce_seconds"] = time.perf_counter() - started
    if check_codec:
        link.check_codec_identity(sample)
        synchronize(device)
    h_hat, err_var = sample["csi_estimates"]["practical"]
    started = time.perf_counter()
    symbols, effective_noise = link.equalizer(sample["y"], h_hat, err_var, sample["no"])
    if not torch.isfinite(symbols).all() or not torch.isfinite(effective_noise).all() or (effective_noise <= 0).any():
        raise ValueError("Invalid frozen native equalizer output")
    native_llr = link.demapper(symbols, effective_noise)
    if not torch.isfinite(native_llr).all():
        raise ValueError("Nonfinite frozen native APP LLRs")
    synchronize(device)
    timings["native_equalizer_app_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    decoded, crc = link.codec.decode(native_llr)
    synchronize(device)
    timings["native_decode_seconds"] = time.perf_counter() - started
    batch_size, users, streams, coded_length = native_llr.shape
    data_count = len(batch["data_indices"])
    if batch_size != 1 or streams != 1 or coded_length != 4 * data_count:
        raise ValueError("Expected complete single-stream Mode-B codewords")
    mapper_llr = native_llr.reshape(1, users, data_count, 4).permute(0, 2, 1, 3)
    native_record = tb_record(mapper_llr, native_llr, decoded, crc, sample["info"])
    llr_chunks = []
    feature_chunks = {name: [] for name in ("ep5_mean", "ep5_variance", "gram", "residual", "eta")}
    timings.update(frontend_seconds=0., ep_seconds=0., feature_seconds=0.)
    for start in range(0, data_count, settings["re_chunk"]):
        ordinals = torch.arange(start, min(start + settings["re_chunk"], data_count))
        started = time.perf_counter()
        statistics = shared_statistics(batch, ordinals, settings["ce_epsilon"])
        synchronize(device)
        timings["frontend_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        posterior = ep(statistics["z"].reshape(-1, users), statistics["gram"].reshape(-1, users, users), return_iterations=(5,))
        mean = posterior["x_hat"].reshape(1, -1, users)
        variance = posterior["posterior_variance"].reshape(1, -1, users)
        synchronize(device)
        timings["ep_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        statistics["residual"] = posterior_residual(statistics, mean, variance, batch["y"].shape[-1],
                                                     settings["residual_atol"], settings["residual_rtol"])
        statistics.update(ep5_mean=mean, ep5_variance=variance)
        for name in feature_chunks:
            feature_chunks[name].append(statistics[name].cpu())
        llr_chunks.append(posterior["llr"].reshape(1, -1, users, 4).cpu())
        synchronize(device)
        timings["feature_seconds"] += time.perf_counter() - started
    ep_llr = torch.cat(llr_chunks, 1)
    if ep_llr.dtype != torch.float32 or not torch.isfinite(ep_llr).all():
        raise ValueError("Invalid frozen EP5 LLR output")
    decoder_input = llrs_to_native_order(ep_llr.to(device))
    started = time.perf_counter()
    decoded, crc = link.codec.decode(decoder_input)
    synchronize(device)
    timings["ep_decode_seconds"] = time.perf_counter() - started
    channel = dict(split="confirmation", seed=int(seed), ebno_db=float(ebno), n0=batch["n0"].cpu().clone(),
                   data_indices=batch["data_indices"].cpu().clone(), native_data_ordinals=torch.arange(data_count),
                   coded_bits=batch["coded_bits"].cpu().clone(), info_bits=batch["bits"].cpu().clone(),
                   methods=dict(sionna_lmmse=native_record,
                                ep5=tb_record(ep_llr, decoder_input, decoded, crc, sample["info"])),
                   features={name: torch.cat(parts, 1) for name, parts in feature_chunks.items()})
    del sample, batch, h_hat, err_var, symbols, native_llr, posterior, statistics
    return channel, timings


def load_receivers(training_dir, context, config, load_models=False):
    from evaluation.llr_mechanism_common import load_frozen_models
    from training.train_llr_mechanism_controls import validate_training

    trained = validate_training(training_dir, context, config)
    receivers = [dict(method=method, candidate=method, training_seed=None, checkpoint_type="fixed", step=None,
                      parameter_count=count) for method, count in zip(BASELINES, (0, 0, 1, 4, 12))]
    receivers[1]["alpha"] = [1.]
    receivers[2]["alpha"] = trained["constants"]["global_scale"]["alpha"]
    receivers[3]["alpha"] = trained["constants"]["per_bit_scale"]["alpha"]
    receivers[4]["noise_fit"] = trained["constants"]["noise_per_bit"]
    artifacts = [dict(file="manifest.json", sha256=file_hash(Path(training_dir) / "manifest.json"))]
    for record in trained["records"]:
        for kind in ("best", "last"):
            artifact = record[kind]
            contained_file(training_dir, artifact["file"], artifact["sha256"])
            artifacts.append(dict(file=artifact["file"], sha256=artifact["sha256"]))
        selected = record["best"]
        receiver = dict(method=f"{record['candidate']}_seed{record['training_seed']}_best", candidate=record["candidate"],
                        training_seed=record["training_seed"], checkpoint_type="best", step=selected["step"],
                        parameter_count=record["parameter_count"], normalization=trained["normalization"],
                        checkpoint_sha256=selected["sha256"], checkpoint_file=selected["file"])
        if load_models:
            checkpoint = torch.load(Path(training_dir) / selected["file"], map_location="cpu", weights_only=True)
            model = make_model(record["candidate"], config).to(config["runtime"]["device"])
            model.load_state_dict(checkpoint["model_state"], strict=True)
            receiver["model"] = model.eval().requires_grad_(False)
        receivers.append(receiver)
    references = load_frozen_models(context, config["runtime"]["device"]) if load_models else context["frozen_references"]
    for reference in references:
        descriptor = dict(reference)
        if descriptor.get("original_config", config)["features"] != config["features"]:
            raise ValueError("Frozen reference feature preprocessing differs from the shared raw feature contract")
        descriptor.setdefault("method", f"frozen_reference_seed{reference['training_seed']}")
        descriptor.setdefault("candidate", "frozen_reference")
        descriptor.setdefault("checkpoint_type", "frozen_best")
        descriptor.setdefault("parameter_count", 1412)
        if load_models and "normalization" not in descriptor:
            raise ValueError("Frozen reference needs its original normalization")
        receivers.append(descriptor)
    if len({receiver["method"] for receiver in receivers}) != len(receivers):
        raise ValueError("Duplicate evaluation receiver identity")
    return trained, receivers, artifacts


@torch.inference_mode()
def evaluate_observation(channel, receivers, codec, config, default_normalization, timings, *, redecode_baselines=False):
    device = config["runtime"]["device"]
    raw = channel["methods"]["ep5"]["llr"]
    labels = coded_labels(channel)
    if raw.dtype != torch.float32 or raw.shape != labels.shape or raw.shape[0] != 1:
        raise ValueError("Expected complete float32 mapper-order EP5 codewords")
    if not torch.equal(llrs_to_native_order(raw), channel["methods"]["ep5"]["decoder_input_llr"]):
        raise ValueError("EP5 native and mapper LLR orders differ")
    synchronize(device)
    started = time.perf_counter()
    nodes, edges = feature_tensors(channel, config)
    edges = edges.to(device)
    normalized = {}
    for receiver in receivers:
        if "model" in receiver:
            normalizer = receiver.get("normalization", default_normalization)
            key = digest(normalizer)
            if key not in normalized:
                normalized[key] = normalize_nodes(nodes, normalizer, config).to(device)
    synchronize(device)
    timings["feature_seconds"] = timings.get("feature_seconds", 0.) + time.perf_counter() - started
    channel["inference_feature_identity"] = tensor_identity(nodes)
    raw_blocks = channel["methods"]["ep5"]["block_error"]
    rows = []
    for receiver in receivers:
        method = receiver["method"]
        noise_position = None
        synchronize(device)
        started = time.perf_counter()
        if method in ("native_lmmse", "ep5_raw"):
            cached = channel["methods"]["sionna_lmmse" if method == "native_lmmse" else "ep5"]
            llr = cached["llr"]
            if not torch.equal(llrs_to_native_order(llr), cached["decoder_input_llr"]):
                raise ValueError("Baseline full-codeword ordering differs")
            alpha = None if method == "native_lmmse" else torch.ones_like(raw[0])
            inference_seconds = 0.
        else:
            if "model" in receiver:
                features = normalized[digest(receiver.get("normalization", default_normalization))]
                parts = [receiver["model"](features[start:start + config["evaluation_chunk"]],
                                             edges[start:start + config["evaluation_chunk"]]).cpu()
                         for start in range(0, len(nodes), config["evaluation_chunk"])]
                alpha = torch.cat(parts, 0)
                llr = conditional_scale(raw[0], alpha).unsqueeze(0)
            else:
                if "noise_fit" in receiver:
                    values = interpolate_noise_alpha(channel["n0"], receiver["noise_fit"])
                    knots = [knot["n0"] for knot in receiver["noise_fit"]["knots"]]
                    current_n0 = float(channel["n0"])
                    position = "below_training_range" if current_n0 < knots[0] else "above_training_range" if current_n0 > knots[-1] else "training_knot" if current_n0 in knots else "interior"
                    noise_position = dict(position=position, actual_n0=current_n0, training_n0_range=[knots[0], knots[-1]], outside_range="endpoint_hold")
                else:
                    values = receiver["alpha"]
                llr = scale_llr(raw, values)
                alpha = torch.as_tensor(values, dtype=torch.float32).expand_as(raw[0])
            synchronize(device)
            inference_seconds = time.perf_counter() - started
        if not torch.isfinite(llr).all() or llr.dtype != torch.float32 or llr.shape != raw.shape:
            raise ValueError("Invalid complete float32 LLRs")
        if method != "native_lmmse" and not torch.equal(llr > 0, raw > 0):
            raise ValueError("Positive scaling changed raw EP5 hard decisions")
        if method == "ep5_raw" and not torch.equal(scale_llr(raw, [1.]), llr):
            raise ValueError("alpha=1 changed raw EP5 LLRs")
        if method in ("native_lmmse", "ep5_raw") and not redecode_baselines:
            decoded, crc = cached["decoded_payload_bits"], cached["crc_ok"]
            decode_seconds = timings["native_decode_seconds" if method == "native_lmmse" else "ep_decode_seconds"]
        else:
            synchronize(device)
            started = time.perf_counter()
            decoded, crc = codec.decode(llrs_to_native_order(llr).to(device))
            decoded, crc = decoded.cpu(), crc.cpu()
            synchronize(device)
            decode_seconds = time.perf_counter() - started
        outcomes, totals = payload_statistics(channel["info_bits"], decoded, crc)
        if method in ("native_lmmse", "ep5_raw"):
            for name, value in outcomes.items():
                if not torch.equal(value, cached[name]):
                    raise ValueError(f"{method}: complete-codeword baseline mismatch in {name}")
        rescue = raw_blocks & ~outcomes["block_error"]
        harm = ~raw_blocks & outcomes["block_error"]
        outcomes.update(rescue_vs_raw=rescue, harm_vs_raw=harm)
        descriptor = {key: receiver[key] for key in ("method", "candidate", "training_seed", "checkpoint_type", "step", "parameter_count")}
        rows.append(dict(**descriptor, seed=channel["seed"], ebno_db=channel["ebno_db"], n0=float(channel["n0"]),
                         **totals, **soft_statistics(llr, labels), rescue_vs_raw=int(rescue.sum()), harm_vs_raw=int(harm.sum()),
                         alpha=None if alpha is None else scale_distribution(alpha, config), noise_conditioning=noise_position,
                         scale_forward_seconds=inference_seconds, decode_seconds=decode_seconds,
                         positive_scale_hard_decisions_unchanged=None if method == "native_lmmse" else True,
                         decoded_payload_identity=tensor_identity(outcomes["decoded_payload_bits"]),
                         per_ue={key: value.tolist() for key, value in outcomes.items() if key != "decoded_payload_bits"}))
    return rows


def qualify_smoke(channel, codec, config, normalization):
    initial = [dict(method="ep5_raw", candidate="ep5_raw", training_seed=None, checkpoint_type="raw", step=0,
                    parameter_count=0, alpha=[1.])]
    for candidate in config["candidates"]:
        model = make_model(candidate, config).to(config["runtime"]["device"]).eval().requires_grad_(False)
        initial.append(dict(method=candidate, candidate=candidate, training_seed=None, checkpoint_type="initial", step=0,
                            normalization=normalization, parameter_count=sum(parameter.numel() for parameter in model.parameters()), model=model))
    rows = evaluate_observation(channel, initial, codec, config, normalization, {}, redecode_baselines=True)
    anchor = rows[0]
    for row in rows[1:]:
        if (row["alpha"]["min"] != 1. or row["alpha"]["max"] != 1.
                or row["decoded_payload_identity"] != anchor["decoded_payload_identity"] or row["per_ue"] != anchor["per_ue"]):
            raise ValueError("Fresh mechanism control did not preserve complete-codeword alpha=1 identity")
    return dict(seed=channel["seed"], ebno_db=channel["ebno_db"], source_split="old_validation",
                full_codeword=True, candidates={candidate: True for candidate in config["candidates"]},
                decoded_payload_crc_identity=True)


def link_and_ep(context, config):
    from detectors.classical.ep import ExpectationPropagationDetector
    from link_level.sionna_reference import SionnaReferenceLink

    reference = context["cache_manifest"]["identity"]["reference"]
    link_config = copy.deepcopy(reference["config"])
    link_config["general"]["device"] = config["runtime"]["device"]
    link_config["channel_estimation"]["mode"] = "practical"
    link = SionnaReferenceLink(link_config)
    if link.grid_metadata != reference["grid"] or link.codec.metadata() != reference["codec"]:
        raise ValueError("Confirmation grid/codec differs from the frozen source distribution")
    if link.covariance_metadata["sha256"] != reference["covariance"]["sha256"]:
        raise ValueError("Confirmation practical CE covariance identity differs")
    ep_config = dict(link.cfg, modulation=dict(bits_per_symbol=4))
    return link, ExpectationPropagationDetector(ep_config, num_iterations=5, damping=.5)


def validate_complete(directory, identity, expected, methods, training_dir):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "complete" or summary.get("evaluation_identity") != identity
            or summary.get("planned_case_count") != len(expected) or summary.get("channel_ebno_observations") != len(expected)
            or summary.get("independent_channels") != len({seed for seed, _ in expected})
            or summary.get("all_baseline_checks_passed") is not True
            or summary.get("all_positive_scale_hard_decisions_unchanged") is not True):
        raise ValueError("Evaluation is incomplete or identity differs")
    if summary.get("content_sha256") != digest({key: value for key, value in summary.items() if key != "content_sha256"}):
        raise ValueError("Summary content checksum mismatch")
    manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    actual = [(item["seed"], item["ebno_db"]) for item in manifest["cases"]]
    if (manifest.get("status") != "complete" or manifest.get("evaluation_identity") != identity
            or len(actual) != len(set(actual)) or set(actual) != expected):
        raise ValueError("Incomplete, duplicated, or unexpected committed confirmation cases")
    filenames = [artifact["file"] for artifact in summary["artifacts"]]
    required = {"evaluation_manifest.json", "channels.jsonl", "summary.csv", "decision_summary.json", "targets.json", "parameters_added_time.csv"}
    required.update(f"plots/{name}.{extension}" for name in ("bler_curves", "feature_ablation", "parameters_added_time") for extension in ("pdf", "svg"))
    if len(filenames) != len(set(filenames)) or not required.issubset(filenames):
        raise ValueError("Required unique report/table/vector artifacts are missing")
    for artifact in summary["artifacts"]:
        contained_file(directory, artifact["file"], artifact["sha256"])
    for artifact in summary["training_artifacts"]:
        contained_file(training_dir, artifact["file"], artifact["sha256"])
    case_files, row_hashes = set(), {}
    for artifact in manifest["cases"]:
        path = contained_file(directory, artifact["file"], artifact["sha256"])
        case_files.add(artifact["file"])
        case = json.loads(path.read_text(encoding="utf-8"))
        if (case.get("status") != "complete" or case.get("evaluation_identity") != identity
                or case["seed"] != artifact["seed"] or case["ebno_db"] != artifact["ebno_db"]
                or {row["method"] for row in case["rows"]} != methods or len(case["rows"]) != len(methods)):
            raise ValueError("Committed case does not contain every fixed receiver exactly once")
        for row in case["rows"]:
            if (row["seed"], row["ebno_db"]) != (case["seed"], case["ebno_db"]) or row["case_file"] != artifact["file"]:
                raise ValueError("Case row is attached to another observation")
            row_hashes[(row["seed"], row["ebno_db"], row["method"])] = digest(row)
    if set(filenames) != case_files | required or len(case_files) != len(expected):
        raise ValueError("Summary omits committed per-case artifacts")
    with (directory / "channels.jsonl").open(encoding="utf-8") as handle:
        coordinates = []
        for row in map(json.loads, handle):
            coordinate = (row["seed"], row["ebno_db"], row["method"])
            if row_hashes.get(coordinate) != digest(row):
                raise ValueError("Flattened report differs from committed case outcomes")
            coordinates.append(coordinate)
    expected_rows = {(seed, ebno, method) for seed, ebno in expected for method in methods}
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected_rows:
        raise ValueError("Flattened report omits or duplicates fixed-budget method observations")
    return summary


def run(args):
    from evaluation.llr_mechanism_common import load_config, prepare, records_for, run_paths
    from evaluation.report_llr_mechanism_controls import summarize, write_outputs
    from evaluation.paper_soft_common import load_channel

    config = load_config(args.config, mode=args.mode)
    context = prepare(config, require_gpu=not args.check_complete)
    paths = run_paths(config, args.run_id)
    training_dir, results_dir = paths["training"], paths["results"]
    trained, receivers, training_artifacts = load_receivers(training_dir, context, config)
    selected_split = "validation" if args.mode == "smoke" else "confirmation"
    records = records_for(context, selected_split)
    expected = {(record["seed"], record["ebno_db"]) for record in records}
    if not records or len(expected) != len(records):
        raise ValueError("Expected a nonempty, duplicate-free fixed evaluation plan")
    receiver_descriptors = [{key: receiver[key] for key in ("method", "candidate", "training_seed", "checkpoint_type", "step", "parameter_count")}
                            for receiver in receivers]
    identity = digest(dict(run_identity_id=context["run_identity_id"], training_artifacts=training_artifacts,
                           selected_checkpoint="best", receivers=receiver_descriptors, planned_cases=records))
    if args.check_complete or (args.reuse_complete and (results_dir / "summary.json").exists()):
        validate_complete(results_dir, identity, expected, {receiver["method"] for receiver in receivers}, training_dir)
        print(f"Complete fixed-budget evaluation verified: {results_dir / 'summary.json'}", flush=True)
        return 0
    if results_dir.exists():
        raise FileExistsError("Result directory exists; select a new run-id or reuse an entirely complete identical evaluation")
    results_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = results_dir / "evaluation_manifest.json"
    manifest = dict(schema_version=1, status="running", run_id=args.run_id, training_root=str(training_dir),
                    constants=trained.get("constants"), normalization=trained["normalization"],
                    evaluation_identity=identity, run_identity_id=context["run_identity_id"],
                    run_identity=context["run_identity"], code_identity=context["code_identity"], environment=context["environment"],
                    mode=args.mode, split=selected_split, fixed_budget=True, planned_cases=records,
                    unique_channel_seeds=sorted({record["seed"] for record in records}),
                    receivers=receiver_descriptors, training_artifacts=training_artifacts,
                    frozen_references=[{key: value for key, value in reference.items() if key != "model"}
                                       for reference in context["frozen_references"]],
                    reference=context["cache_manifest"]["identity"]["reference"], cases=[],
                    payload_retention="Per-UE errors/CRC and decoded-payload hash; no waveform/channel/LLR arrays retained",
                    stopping_rule="Execute every predeclared channel/EbNo pair; failures stop execution, never outcome-based early stopping")
    save_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        trained, receivers, _ = load_receivers(training_dir, context, config, load_models=True)
        if args.mode == "smoke":
            link, ep = None, None
        else:
            link, ep = link_and_ep(context, config)
        shared_totals = {}
        for number, record in enumerate(records, 1):
            case_started = time.perf_counter()
            print(f"{selected_split} {number}/{len(records)} seed={record['seed']} Eb/N0={record['ebno_db']:g} START", flush=True)
            if args.mode == "smoke":
                channel = load_channel(context["cache_dir"], record, context["cache_manifest"]["identity_id"])
                timings = dict(transmit_seconds=0., ce_seconds=0., native_equalizer_app_seconds=0., frontend_seconds=0.,
                               ep_seconds=0., native_decode_seconds=0., ep_decode_seconds=0.)
                codec = codec_from_cache(channel, context["cache_manifest"], config["runtime"]["device"])
                manifest["initial_alpha_one_identity"] = qualify_smoke(channel, codec, config, trained["normalization"])
            else:
                channel, timings = stream_observation(link, ep, record["seed"], record["ebno_db"], config, check_codec=(number == 1))
                codec = link.codec
            rows = evaluate_observation(channel, receivers, codec, config, trained["normalization"], timings,
                                        redecode_baselines=args.mode == "smoke")
            relative = case_name(record["seed"], record["ebno_db"])
            path = results_dir / relative
            if path.exists():
                raise FileExistsError("Refusing to overwrite a committed evaluation case")
            for row in rows:
                row.update(split=selected_split, case_file=relative)
            case = dict(schema_version=1, status="complete", evaluation_identity=identity, split=selected_split,
                        seed=record["seed"], ebno_db=record["ebno_db"], n0=float(channel["n0"]),
                        input_identity={name: tensor_identity(channel[name]) for name in
                                        ("coded_bits", "info_bits", "data_indices", "native_data_ordinals")},
                        ep5_llr_identity=tensor_identity(channel["methods"]["ep5"]["llr"]),
                        inference_feature_identity=channel["inference_feature_identity"],
                        input_order="coded[b,UE,n*4+q] corresponds to mapper LLR[b,n,UE,q]; original ResourceGrid data indices",
                        reference_identity=digest(manifest["reference"]), timings=timings, rows=rows,
                        elapsed_seconds=time.perf_counter() - case_started)
            save_json(path, case)
            artifact = dict(file=relative, sha256=file_hash(path), bytes=path.stat().st_size,
                            seed=record["seed"], ebno_db=record["ebno_db"], status="complete")
            manifest["cases"].append(artifact)
            manifest["elapsed_seconds"] = time.perf_counter() - started
            for name, value in timings.items():
                shared_totals[name] = shared_totals.get(name, 0.) + value
            save_json(manifest_path, manifest)
            mean_seconds = manifest["elapsed_seconds"] / number
            print(f"{selected_split} {number}/{len(records)} COMPLETE elapsed={case['elapsed_seconds']:.2f}s "
                  f"total={manifest['elapsed_seconds']:.2f}s observed_mean={mean_seconds:.2f}s "
                  f"estimated_remaining={mean_seconds * (len(records) - number):.2f}s", flush=True)
            del channel, rows, case
        manifest["status"] = "complete"
        save_json(manifest_path, manifest)
        rows = []
        channels_path = results_dir / "channels.jsonl"
        temporary = channels_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for artifact in manifest["cases"]:
                case = json.loads((results_dir / artifact["file"]).read_text(encoding="utf-8"))
                for row in case["rows"]:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
        temporary.replace(channels_path)
        summary = dict(schema_version=1, status="running", evaluation_identity=identity, run_identity_id=context["run_identity_id"],
                       mode=args.mode, split=selected_split, reference=manifest["reference"], planned_case_count=len(records),
                       independent_channels=len(manifest["unique_channel_seeds"]), channel_ebno_observations=len(records),
                       receivers=receiver_descriptors, training_artifacts=training_artifacts,
                       shared_pipeline_timing=dict(case_count=len(records), totals_seconds=shared_totals),
                       timing_definition="Synchronized wall times separate waveform, CE, frozen native equalizer/APP, shared frontend, EP, feature preparation, scale forward and decode. Cached smoke timings exclude original receiver generation; added scale cost is not full receiver speedup.",
                       initial_alpha_one_identity=manifest.get("initial_alpha_one_identity"),
                       all_baseline_checks_passed=True, all_positive_scale_hard_decisions_unchanged=True)
        summary.update(summarize(rows, config))
        artifacts = [dict(file=artifact["file"], sha256=artifact["sha256"], bytes=artifact["bytes"]) for artifact in manifest["cases"]]
        artifacts += [dict(file=path.name, sha256=file_hash(path), bytes=path.stat().st_size)
                      for path in (manifest_path, channels_path)]
        artifacts += write_outputs(results_dir, summary, rows, config)
        summary.update(status="complete", artifacts=artifacts, elapsed_seconds=time.perf_counter() - started)
        summary["content_sha256"] = digest(summary)
        save_json(results_dir / "summary.json", summary)
        validate_complete(results_dir, identity, expected, {receiver["method"] for receiver in receivers}, training_dir)
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
        save_json(manifest_path, manifest)
        if (results_dir / "summary.json").exists():
            failed = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
            failed.update(status="failed", error=manifest["error"])
            failed.pop("content_sha256", None)
            save_json(results_dir / "summary.json", failed)
        raise
    print(f"Complete {selected_split} report: {results_dir / 'summary.json'}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "confirmation"), default="confirmation")
    parser.add_argument("--reuse-complete", action="store_true")
    parser.add_argument("--check-complete", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
