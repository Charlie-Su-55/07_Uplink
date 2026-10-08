"""Read-only historical cache contracts and the fixed 24/8/64 feasibility split."""

import copy
import json
import math
from pathlib import Path
import re
import time

import torch

from evaluation.evaluate_paper_soft_output import coded_labels
from evaluation.paper_soft_common import (
    REPOSITORY, canonical_file_hash, digest, file_sha256, load_channel,
    load_complete_cache, numeric_runtime, source_identity,
)


DEFAULT_CONFIG = "configs/evaluation/cached_llr_scaling.yaml"
FEATURE_NAMES = ["mean_real", "mean_imag", "log_variance", "log_gram_diagonal", "log_residual", "eta"]
CANDIDATES = ["affine", "mlp", "graph", "mlp_no_residual_eta"]
PROTECTED = ["evaluation/build_paper_soft_cache.py", "evaluation/evaluate_paper_soft_output.py",
             "evaluation/paper_soft_common.py", "link_level/nr_codec.py", "link_level/detector_adapter.py",
             "link_level/sionna_reference.py", "link_level/sionna_ce.py", "link_level/sionna_ebno.py",
             "detectors/classical/ep.py", "training/train_paper_reference_detector.py"]


def load_config(path=DEFAULT_CONFIG, mode="development"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or mode not in ("smoke", "development"):
        raise ValueError("Unknown conditional-scaling schema or mode.")
    if config["protocol"] != dict(train_channels=24, validation_channels=8, development_channels=64,
                                   ebno_dbs=[-12, -10.5, -9]):
        raise ValueError("This feasibility protocol fixes the seed-grouped 24/8/64 split and three Eb/N0 points.")
    if config["candidates"] != CANDIDATES or config["features"]["names"] != FEATURE_NAMES:
        raise ValueError("The prespecified candidates and inference feature allowlist must match.")
    model, train = config["model"], config["training"]
    if model["alpha_min"] != .05 or model["alpha_max"] != 8 or model["layers"] != 2 or model["hidden"] != 32:
        raise ValueError("Use the fixed common scale range and two-layer, hidden-32 candidates.")
    fit = config["constant_fit"]
    if fit["alpha_min"] != model["alpha_min"] or fit["alpha_max"] != model["alpha_max"]:
        raise ValueError("Constant and conditional candidates must have the same scale bounds.")
    if fit["method"] != "bounded_derivative_bisection" or fit["precision"] != "float64":
        raise ValueError("Use the declared constant-scale optimizer.")
    if train["seeds"] != [42, 43, 44] or train["optimizer"] != "AdamW":
        raise ValueError("All candidates require the same three training seeds and AdamW optimizer.")
    for section, fields in ((train, ("steps", "re_per_step", "validate_every", "validation_chunk")),
                            (config["bootstrap"], ("replicates",)),
                            (fit, ("max_iterations",))):
        if any(not isinstance(section[key], int) or isinstance(section[key], bool) or section[key] < 1 for key in fields):
            raise ValueError("Iteration, chunk and sample counts must be positive integers.")
    positive = [train[key] for key in ("learning_rate", "grad_clip")]
    positive += [config["features"][key] for key in ("log_epsilon", "normalization_epsilon", "edge_epsilon")]
    positive += [fit[key] for key in ("alpha_tolerance", "gradient_tolerance")]
    positive += [model["saturation_tolerance"]]
    if any(not math.isfinite(value) or value <= 0 for value in positive) or not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
        raise ValueError("Invalid numerical/training tolerances.")
    for name in ("log_clip", "raw_clip", "standardized_clip", "edge_clip"):
        bounds = config["features"][name]
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not bounds[0] < bounds[1]:
            raise ValueError("Invalid predeclared feature clipping bounds.")
    if (not 0 < config["bootstrap"]["confidence"] < 1 or config["runtime"]["precision"] != "single"
            or config["runtime"]["torch_threads"] < 1 or config["evaluation_chunk"] < 1):
        raise ValueError("Invalid precision/runtime/bootstrap configuration.")
    smoke = config["smoke"]
    if (not isinstance(smoke["training_steps"], int) or isinstance(smoke["training_steps"], bool)
            or smoke["training_steps"] < 1 or smoke["training_seed"] != 42
            or isinstance(smoke["training_seed"], bool) or smoke["ebno_db"] != -10.5):
        raise ValueError("Smoke uses positive optimizer steps, training seed 42, and -10.5 dB.")
    config["mode"] = mode
    if mode == "smoke":
        train["steps"] = config["smoke"]["training_steps"]
        train["validate_every"] = train["steps"]
        train["seeds"] = [config["smoke"]["training_seed"]]
    return config


def run_paths(config, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or run_id in (".", ".."):
        raise ValueError("run-id must be a simple name without path separators.")
    paths = {name: Path(config["outputs"][name + "_root"]) / run_id for name in ("training", "results", "logs")}
    protected = [Path(config[key]).resolve() for key in ("source_cache", "source_archive")]
    resolved = [path.resolve() for path in paths.values()]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for index, a in enumerate(resolved) for b in resolved[index + 1:]):
        raise ValueError("Training, results and logs must have separate output directories.")
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a in resolved for b in protected):
        raise ValueError("New output directories must not overlap the source cache or archive.")
    return paths


def read_archive(config):
    root = Path(config["source_archive"])
    index = json.loads((root / "snapshot_index.json").read_text(encoding="utf-8"))
    files = {}
    for record in index["files"]:
        path = (root / record["archived"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Missing/invalid indexed softdev archive: {record['archived']}")
        if file_sha256(path) != record["sha256"] and canonical_file_hash(path) != record["sha256"]:
            raise ValueError(f"Historical archive hash mismatch: {record['archived']}")
        files[record["archived"]] = dict(record)
    required = {"cache_manifest.json", "summary.json", "summary.csv", "temperatures.json", "channels.jsonl"}
    if not required.issubset(files):
        raise ValueError("Incomplete archived softdev evidence.")
    manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or digest(manifest["identity"]) != manifest["identity_id"]:
        raise ValueError("Invalid original archived cache identity.")
    return dict(manifest=manifest, files=files, index_sha256=canonical_file_hash(root / "snapshot_index.json"))


def split_plan(manifest, config):
    """Every seed and all its Eb/N0 cases stay together; no RE-based splitting."""
    protocol, original = config["protocol"], manifest["seed_plan"]
    calibration, development = sorted(original["calibration"]), sorted(original["development"])
    if (len(calibration) != 32 or len(development) != 64 or len(set(calibration + development)) != 96
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in calibration + development)):
        raise ValueError("Require distinct original 32 calibration and 64 development channel identities.")
    excluded = set(original["excluded_seeds"])
    if excluded.intersection(calibration + development):
        raise ValueError("Historical channel exclusions overlap the source cache.")
    expected = {(group, seed, float(ebno)) for group, seeds in (("calibration", calibration), ("development", development))
                for seed in seeds for ebno in protocol["ebno_dbs"]}
    actual = [(record["split"], record["seed"], float(record["ebno_db"])) for record in manifest["channels"]]
    if len(actual) != len(set(actual)) or set(actual) != expected or any(record["status"] != "complete" for record in manifest["channels"]):
        raise ValueError("Incomplete/duplicate source-cache seed/EbNo groups.")
    train, validation = calibration[:24], calibration[24:]
    membership = {seed: group for group, seeds in (("train", train), ("validation", validation), ("development", development)) for seed in seeds}
    return dict(train=train, validation=validation, development=development,
                rule="Sorted source calibration seeds: first 24 train, last 8 validation; all 64 source development seeds held out until training finishes; every Eb/N0 follows its seed.",
                protocol="Feasibility development only; not a final paper training/test protocol.",
                files=[dict(source_split=r["split"], split=membership[r["seed"]], seed=r["seed"], ebno_db=r["ebno_db"],
                            file=r["file"], sha256=r["sha256"], bytes=r["bytes"]) for r in manifest["channels"]])


def training_code_identity():
    identity = source_identity()
    new_paths = ["models/conditional_llr_scaling.py", "configs/evaluation/cached_llr_scaling.yaml",
                 "evaluation/run_cached_llr_scaling_batch.sh"]
    for directory in ("evaluation", "training", "tests"):
        new_paths.extend(path.relative_to(REPOSITORY).as_posix() for path in (REPOSITORY / directory).glob("*cached_llr*.py"))
    identity["conditional_sources_lf_sha256"] = {name: canonical_file_hash(REPOSITORY / name)
                                                  for name in sorted(set(new_paths)) if (REPOSITORY / name).is_file()}
    return identity


def prepare(config, *, require_gpu=True):
    started = time.perf_counter()
    root = Path(config["source_cache"])
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"STOP: complete source cache is missing: {root}. No channel generation or cache rebuilding is allowed.")
    archive = read_archive(config)
    expected_hash = archive["files"]["cache_manifest.json"]["sha256"]
    if file_sha256(root / "manifest.json") != expected_hash:
        raise ValueError("Source cache manifest differs from the archived original bytes; identity cannot be rewritten for a new commit.")
    print("Preflight: validating original complete cache schema, identity and every channel hash (read-only).", flush=True)
    manifest = load_complete_cache(root)
    if manifest != archive["manifest"]:
        raise ValueError("Source cache manifest content differs from the archived original.")
    plan = split_plan(manifest, config)
    historical_hashes = manifest["identity"]["source"]["python_sources_lf_sha256"]
    for name in PROTECTED:
        if canonical_file_hash(REPOSITORY / name) != historical_hashes[name]:
            raise ValueError(f"Frozen receiver/cache/codec source changed since cache generation: {name}")
    environment = numeric_runtime(config, require_gpu=require_gpu)
    code = training_code_identity()
    source = dict(path=str(root), identity_id=manifest["identity_id"], identity=manifest["identity"],
                  manifest_sha256=expected_hash, archive_index_sha256=archive["index_sha256"],
                  archive_files=archive["files"], generator_code_identity=manifest["identity"]["source"])
    identity = dict(schema_version=1, config=config, source_cache=source, split_plan=plan,
                    training_code_identity=code, environment=environment)
    print(f"Preflight PASS: source cache {source['identity_id']}; train/validation/development=24/8/64; "
          f"current code {code['commit']}; elapsed={time.perf_counter()-started:.2f}s", flush=True)
    return dict(config=config, cache_dir=root, cache_manifest=manifest, source_cache=source,
                split_plan=plan, run_identity=identity, run_identity_id=digest(identity),
                environment=environment, code_identity=code)


def records_for(context, split):
    if split not in ("train", "validation", "development"):
        raise ValueError("Unknown feasibility split.")
    config = context["config"]
    if config["mode"] == "smoke" and split == "development":
        raise ValueError("Smoke may use train/validation only; development is not an early diagnostic.")
    seeds = context["split_plan"][split]
    source_split = "development" if split == "development" else "calibration"
    records = [record for record in context["cache_manifest"]["channels"]
               if record["split"] == source_split and record["seed"] in seeds]
    if config["mode"] == "smoke":
        records = [record for record in records if record["seed"] == seeds[0] and record["ebno_db"] == config["smoke"]["ebno_db"]]
    return sorted(records, key=lambda record: (record["seed"], record["ebno_db"]))


def feature_tensors(channel, config):
    """Allowlisted receiver inputs only. No bits, CRC/error outcomes, IDs or true H."""
    settings, features = config["features"], channel["features"]
    mean, variance, gram = features["ep5_mean"][0], features["ep5_variance"][0], features["gram"][0]
    residual, eta = features["residual"][0], features["eta"][0]
    diagonal = gram.diagonal(dim1=-2, dim2=-1).real
    if any(not torch.isfinite(value).all() for value in (mean, variance, gram, residual, eta)) or (variance < 0).any() or (diagonal < 0).any():
        raise ValueError("Invalid finite receiver-only features.")
    # Cached residual already passed its recorded cancellation-error tolerance.
    # The fixed epsilon also defines logs for zero or tolerated tiny negatives.
    logarithm = lambda value: value.clamp_min(settings["log_epsilon"]).log().clamp(*settings["log_clip"])
    nodes = torch.stack((mean.real, mean.imag, logarithm(variance), logarithm(diagonal),
                         logarithm(residual).unsqueeze(-1).expand_as(eta), eta), dim=-1).float()
    nodes = nodes.clamp(*settings["raw_clip"])
    denominator = (diagonal.clamp_min(settings["edge_epsilon"]).unsqueeze(-1)
                   * diagonal.clamp_min(settings["edge_epsilon"]).unsqueeze(-2)).sqrt()
    correlation = gram / denominator
    edges = torch.stack((correlation.real, correlation.imag, correlation.abs()), -1).float().clamp(*settings["edge_clip"])
    if nodes.ndim != 3 or nodes.shape[-1] != 6 or edges.shape != (*nodes.shape[:2], nodes.shape[1], 3):
        raise ValueError("Receiver node/edge axes differ from [RE,UE,feature] and [RE,UE,UE,edge].")
    return nodes.contiguous(), edges.contiguous()


def load_prepared(context, split):
    if split not in ("train", "validation"):
        raise ValueError("Training preloader cannot access development channels.")
    result, started = [], time.perf_counter()
    records = records_for(context, split)
    for index, record in enumerate(records, 1):
        channel = load_channel(context["cache_dir"], record, context["source_cache"]["identity_id"])
        nodes, edges = feature_tensors(channel, context["config"])
        result.append(dict(split=split, seed=record["seed"], ebno_db=record["ebno_db"], file=record["file"], sha256=record["sha256"],
                           nodes=nodes, edges=edges, llr=channel["methods"]["ep5"]["llr"][0].clone(), labels=coded_labels(channel)[0].clone()))
        print(f"Preload {split} {index}/{len(records)} seed={record['seed']} Eb/N0={record['ebno_db']:g} "
              f"elapsed={time.perf_counter()-started:.2f}s", flush=True)
    if not result:
        raise ValueError("Empty complete-codeword training/validation split.")
    return result


def fit_normalizer(train_items, config):
    if not train_items or any(item["split"] != "train" for item in train_items):
        raise ValueError("Normalization can only consume the train identities.")
    total = torch.zeros(6, dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for item in train_items:
        values = item["nodes"].reshape(-1, 6).double()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite train features.")
        total += values.sum(0)
        squares += values.square().sum(0)
        count += len(values)
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(config["features"]["normalization_epsilon"])
    return dict(mean=mean.tolist(), std=std.tolist(), count=count, feature_names=FEATURE_NAMES,
                train_seeds=sorted({item["seed"] for item in train_items}), fit_split="train",
                fit_sources=[dict(file=item["file"], sha256=item["sha256"]) for item in train_items],
                reduction="float64 train-only population moments over all selected complete RE/UE nodes")


def normalize_nodes(nodes, normalization, config):
    if normalization["feature_names"] != FEATURE_NAMES or normalization["fit_split"] != "train":
        raise ValueError("Invalid train-only feature normalization provenance.")
    mean = torch.as_tensor(normalization["mean"], dtype=nodes.dtype, device=nodes.device)
    std = torch.as_tensor(normalization["std"], dtype=nodes.dtype, device=nodes.device)
    if mean.shape != (6,) or std.shape != (6,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid feature normalization moments.")
    return ((nodes - mean) / std).clamp(*config["features"]["standardized_clip"])
