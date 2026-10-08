"""Immutable data, frozen references, and stable identities for mechanism controls."""

import copy
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from evaluation.cached_llr_common import (
    FEATURE_NAMES, feature_tensors, fit_normalizer, normalize_nodes, read_archive,
    run_paths as old_run_paths, split_plan as old_split_plan,
)
from evaluation.evaluate_paper_soft_output import coded_labels
from evaluation.paper_soft_common import (
    REPOSITORY, archive_evidence, canonical_file_hash, collect_seeds, digest,
    file_sha256, load_channel, load_complete_cache, numeric_runtime,
)
from training.train_cached_llr_scaling import training_manifest_digest


DEFAULT_CONFIG = "configs/evaluation/llr_mechanism_controls.yaml"
CANDIDATES = ["full_mlp", "mlp_without_residual", "mlp_without_eta", "mlp_without_both", "affine_all_features"]


def load_config(path=DEFAULT_CONFIG, mode="confirmation"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or mode not in ("smoke", "confirmation"):
        raise ValueError("Unknown mechanism-control schema or mode.")
    if config["candidates"] != CANDIDATES or config["features"]["names"] != FEATURE_NAMES:
        raise ValueError("The five prespecified controls and six-feature interface must remain fixed.")
    if config["protocol"] != dict(train_channels=24, validation_channels=8, development_channels=64, ebno_dbs=[-12, -10.5, -9]):
        raise ValueError("The historical train24/validation8 protocol is fixed.")
    confirmation = config["confirmation"]
    if (confirmation["channels"] != 512 or confirmation["ebno_dbs"] != [-12, -11.5, -11, -10.5, -10, -9.5, -9]
            or confirmation["checkpoint_selection"] != "best" or confirmation["fixed_budget"] is not True
            or confirmation["save_raw_tensors"] is not False or not isinstance(confirmation["start_seed"], int)
            or not 0 <= confirmation["start_seed"] < 2 ** 31):
        raise ValueError("Confirmation requires the prespecified fixed 512-channel, seven-point budget.")
    training, model, fit = config["training"], config["model"], config["constant_fit"]
    if (training["seeds"] != [42, 43, 44] or training["steps"] != 3000 or training["re_per_step"] != 256
            or training["validate_every"] != 100 or training["optimizer"] != "AdamW"
            or model["hidden"] != 32 or model["layers"] != 2
            or model["alpha_min"] != .05 or model["alpha_max"] != 8):
        raise ValueError("Training budget, architecture and positive scale bounds are prespecified.")
    original = json.loads((REPOSITORY / "configs/evaluation/cached_llr_scaling.yaml").read_text(encoding="utf-8"))
    if (config["features"] != original["features"] or config["model"] != original["model"]
            or fit != original["constant_fit"] or any(training[key] != original["training"][key]
                 for key in ("learning_rate", "weight_decay", "grad_clip", "validation_chunk"))):
        raise ValueError("Inherit the original feature processing, bounds and optimization settings unchanged.")
    if config["scale_distribution"] != dict(bin_edges=[.05, .1, .25, .5, .75, 1, 1.5, 2, 4, 8],
                                             convention="left_closed_right_open_last_closed"):
        raise ValueError("Scale distribution uses the same prespecified bins for every method.")
    noise = config["noise_conditioning"]
    if (noise["coordinate"] != "log_n0" or noise["interpolation"] != "linear_log_alpha" or noise["outside_range"] != "endpoint_hold"
            or any(not math.isfinite(noise[key]) or noise[key] <= 0 for key in ("n0_rtol", "n0_atol"))):
        raise ValueError("N0 conditioning requires fixed log/log interpolation and endpoint holding.")
    if config["targets"] != dict(bler=[.1, .05], interpolation="linear_ebno_log10_bler", outside_range="unresolved", multiple_crossings="unresolved"):
        raise ValueError("Both target BLERs require measured brackets; no extrapolation or favorable crossing selection.")
    if config["smoke"] != dict(training_steps=2, training_seed=42, ebno_db=-10.5):
        raise ValueError("Smoke uses old train/validation data only with the fixed two-step budget.")
    if (config["runtime"]["precision"] != "single" or config["runtime"]["torch_threads"] < 1
            or not 0 < config["bootstrap"]["confidence"] < 1 or config["bootstrap"]["replicates"] < 1
            or config["evaluation_chunk"] < 1):
        raise ValueError("Invalid runtime/precision/report configuration.")
    receiver = config.get("receiver", {})
    if (set(receiver) != {"re_chunk", "residual_atol", "residual_rtol", "ce_epsilon"}
            or not isinstance(receiver["re_chunk"], int) or isinstance(receiver["re_chunk"], bool)
            or receiver["re_chunk"] < 1 or any(not math.isfinite(receiver[key]) or receiver[key] <= 0
                for key in ("residual_atol", "residual_rtol", "ce_epsilon"))):
        raise ValueError("Receiver chunk and feature tolerances must have the fixed positive schema.")
    config["mode"] = mode
    if mode == "smoke":
        config["training"].update(seeds=[42], steps=2, validate_every=2)
    return config


def run_paths(config, run_id):
    paths = old_run_paths(config, run_id)
    for path in paths.values():
        for key in ("frozen_training_root", "frozen_archive", "historical_archive"):
            protected, output = Path(config[key]).resolve(), path.resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError("New outputs must not overlap frozen checkpoints or historical archives.")
    return paths


def frozen_evidence(config):
    root = Path(config["frozen_archive"])
    index = json.loads((root / "snapshot_index.json").read_text(encoding="utf-8"))
    files = {}
    for record in index["files"]:
        path = (root / record["archived"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Missing frozen indexed report: {record['archived']}")
        if record["sha256"] not in (file_sha256(path), canonical_file_hash(path)):
            raise ValueError(f"Frozen report checksum mismatch: {record['archived']}")
        files[record["archived"]] = record
    if not {"training_manifest.json", "summary.json", "summary.csv", "decision_summary.json", "channels.jsonl"}.issubset(files):
        raise ValueError("Frozen experiment archive is incomplete.")
    manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "complete" or manifest["content_sha256"] != training_manifest_digest(manifest)
            or digest(manifest["run_identity"]) != manifest["run_identity_id"]):
        raise ValueError("Frozen training identity/content checksum mismatch.")
    records = [record for record in manifest["records"] if record["candidate"] == "mlp"]
    if sorted(record["training_seed"] for record in records) != [42, 43, 44]:
        raise ValueError("Need exactly the three archived full MLP training seeds.")
    return dict(manifest=manifest, records=records, files=files, index_sha256=canonical_file_hash(root / "snapshot_index.json"))


def confirmation_plan(config, source_manifest, frozen):
    plan = old_split_plan(source_manifest, config)
    if plan["train"] != list(range(81000000, 81000024)) or plan["validation"] != list(range(81000024, 81000032)):
        raise ValueError("The actual source seeds differ from the prespecified training/validation protocol.")
    evidence = archive_evidence(config["historical_archive"])
    excluded = set(evidence["excluded_seeds"]) | set(source_manifest["seed_plan"]["excluded_seeds"])
    excluded.update(source_manifest["seed_plan"]["calibration"] + source_manifest["seed_plan"]["development"])
    for document in (source_manifest["identity"], frozen["manifest"]):
        for values in collect_seeds(document).values():
            excluded.update(values)
    # These three indexed historical archives are deliberately fixed. A later
    # report-only archive must not retrospectively change this predeclared split.
    origins = list(evidence["sources"])
    for root in (Path(config["source_archive"]), Path(config["frozen_archive"])):
        path = root / "channels.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing historical evaluation seed evidence: {path}")
        seeds = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                seeds.add(int(json.loads(line)["seed"]))
        excluded.update(seeds)
        origins.append(dict(path=str(path), canonical_sha256=canonical_file_hash(path), evaluation_seeds=sorted(seeds)))
    selected, candidate = [], config["confirmation"]["start_seed"]
    while len(selected) < config["confirmation"]["channels"]:
        if candidate >= 2 ** 31:
            raise ValueError("Exhausted valid independent channel seeds.")
        if candidate not in excluded:
            selected.append(candidate)
        candidate += 1
    plan.update(confirmation=selected, confirmation_excluded_seeds=sorted(excluded), confirmation_exclusion_sources=origins,
                confirmation_rule="Fixed budget; never filter channels by error, CRC, rescue, BLER or model output; all seven Eb/N0 points stay in one seed cluster.")
    return plan


def algorithm_identity(frozen):
    saved = frozen["manifest"]["run_identity"]["training_code_identity"]
    old = {**saved["python_sources_lf_sha256"], **saved["conditional_sources_lf_sha256"]}
    for name, expected in old.items():
        if canonical_file_hash(REPOSITORY / name) != expected:
            raise ValueError(f"Protected historical algorithm/config/test source changed: {name}")
    new_paths = [REPOSITORY / DEFAULT_CONFIG]
    for folder in ("models", "training", "evaluation", "tests"):
        new_paths += list((REPOSITORY / folder).glob("*llr_mechanism*"))
    current = {path.relative_to(REPOSITORY).as_posix(): canonical_file_hash(path)
               for path in sorted(new_paths) if path.is_file() and path.suffix in (".py", ".sh", ".yaml")}
    hashes = {**old, **current}
    git = lambda *args: subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()
    return dict(protected_algorithm_hashes=hashes, commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                dirty=bool(git("status", "--porcelain")),
                identity_policy="Commit is provenance only; algorithm/config hashes, environment, data and checkpoint identities remain mandatory after documentation-only commits.")


def validate_frozen_checkpoint(checkpoint, record, archived):
    manifest = archived["manifest"]
    expected = dict(schema_version=1, kind="cached_conditional_llr_scaling_v1", candidate="mlp",
                    training_seed=record["training_seed"], step=record["best"]["step"],
                    run_identity_id=manifest["run_identity_id"], source_cache_identity_id=manifest["source_cache_identity_id"],
                    normalization=manifest["normalization"], config=manifest["config"],
                    sampling_plan_sha256=record["sampling_plan_sha256"], parameter_count=record["parameter_count"])
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen MLP metadata differs from its original archived identity; identities must not be rewritten.")
    if (record["best"]["validation_bce_nats"] != min(row["validation"]["bce_nats"] for row in record["history"])
            or record["best"]["step"] not in [row["step"] for row in record["history"]]):
        raise ValueError("Frozen reference was not its original validation-selected best.")
    from models.conditional_llr_scaling import make_model
    model = make_model("mlp", manifest["config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if (sum(p.numel() for p in model.parameters()) != record["parameter_count"]
            or any(not torch.isfinite(value).all() for value in checkpoint["model_state"].values())):
        raise ValueError("Invalid frozen MLP state or parameter count.")
    return model


def frozen_references(config, archived, *, load_weights=True):
    root, manifest = Path(config["frozen_training_root"]).resolve(), archived["manifest"]
    records = []
    if load_weights and file_sha256(root / "manifest.json") != archived["files"]["training_manifest.json"]["sha256"]:
        raise ValueError("Server frozen training manifest differs from archived original bytes.")
    for record in archived["records"]:
        path = (root / record["best"]["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Frozen checkpoint path escapes its training directory.")
        if load_weights:
            if file_sha256(path) != record["best"]["sha256"]:
                raise ValueError("Frozen MLP checkpoint SHA256 differs from the archived best.")
            validate_frozen_checkpoint(torch.load(path, map_location="cpu", weights_only=True), record, archived)
        records.append(dict(method=f"frozen_reference_seed{record['training_seed']}", candidate="frozen_reference",
                            training_seed=record["training_seed"], checkpoint_type="frozen_best", step=record["best"]["step"],
                            parameter_count=record["parameter_count"], file=str(path), sha256=record["best"]["sha256"],
                            normalization=manifest["normalization"], original_run_identity_id=manifest["run_identity_id"],
                            original_record=record, original_config=manifest["config"]))
    return records


def load_frozen_models(context, device):
    archived = frozen_evidence(context["config"])
    result = []
    for descriptor in context["frozen_references"]:
        if file_sha256(descriptor["file"]) != descriptor["sha256"]:
            raise ValueError("Frozen checkpoint changed after preflight.")
        checkpoint = torch.load(descriptor["file"], map_location="cpu", weights_only=True)
        model = validate_frozen_checkpoint(checkpoint, descriptor["original_record"], archived)
        result.append(dict(descriptor, model=model.to(device).eval().requires_grad_(False), checkpoint_sha256=descriptor["sha256"]))
    return result


def validate_receiver_config(config, source_manifest):
    original = source_manifest["identity"]["experiment_config"]
    expected = dict(re_chunk=original["re_chunk"], **original["features"])
    if config["receiver"] != expected:
        raise ValueError("Confirmation receiver settings differ from the original cached frontend; re_chunk and all feature tolerances are frozen.")


def prepare(config, *, require_gpu=True):
    started, root = time.perf_counter(), Path(config["source_cache"])
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"STOP: historical source cache missing: {root}; it will not be rebuilt.")
    source, frozen = read_archive(config), frozen_evidence(config)
    expected = source["files"]["cache_manifest.json"]["sha256"]
    if file_sha256(root / "manifest.json") != expected:
        raise ValueError("Original source cache bytes/identity changed.")
    print("Preflight: strict original cache validation and frozen checkpoint hashes.", flush=True)
    manifest = load_complete_cache(root)
    if manifest != source["manifest"] or frozen["manifest"]["source_cache_identity_id"] != manifest["identity_id"]:
        raise ValueError("Source cache and frozen reference provenance disagree.")
    validate_receiver_config(config, manifest)
    references = frozen_references(config, frozen)
    prior = manifest["identity"]["reference"]["covariance"]
    if file_sha256(prior["path"]) != prior["sha256"]:
        raise ValueError("Frozen CE covariance SHA256 mismatch; no prior rebuilding is allowed.")
    environment = numeric_runtime(config, require_gpu=require_gpu)
    try:
        environment["matplotlib"] = importlib.metadata.version("matplotlib")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Matplotlib is required for the vector reports; no packages will be installed.") from error
    code, plan = algorithm_identity(frozen), confirmation_plan(config, manifest, frozen)
    data = dict(path=str(root), identity_id=manifest["identity_id"], identity=manifest["identity"], manifest_sha256=expected,
                archive_index_sha256=source["index_sha256"], frozen_archive_index_sha256=frozen["index_sha256"],
                frozen_training_manifest_sha256=frozen["files"]["training_manifest.json"]["sha256"])
    identity = dict(schema_version=1, config=config, source_cache=data, split_plan=plan, frozen_references=references,
                    protected_algorithm_hashes=code["protected_algorithm_hashes"], environment=environment)
    print(f"Preflight PASS: train24/validation8, independent confirmation={len(plan['confirmation'])} seeds; "
          f"elapsed={time.perf_counter()-started:.2f}s", flush=True)
    return dict(config=config, cache_dir=root, cache_manifest=manifest, source_cache=data, split_plan=plan,
                frozen_references=references, run_identity=identity, run_identity_id=digest(identity),
                environment=environment, code_identity=code)


def records_for(context, split):
    config, plan = context["config"], context["split_plan"]
    if split == "confirmation":
        if config["mode"] == "smoke":
            raise ValueError("Smoke cannot access independent confirmation channels.")
        return [dict(split=split, seed=seed, ebno_db=float(ebno)) for seed in plan[split] for ebno in config["confirmation"]["ebno_dbs"]]
    if split not in ("train", "validation"):
        raise ValueError("Old development data is not part of mechanism training, selection or smoke.")
    seeds = plan[split][:1] if config["mode"] == "smoke" else plan[split]
    records = [r for r in context["cache_manifest"]["channels"] if r["split"] == "calibration" and r["seed"] in seeds]
    if config["mode"] == "smoke" and split == "validation":
        records = [r for r in records if r["ebno_db"] == config["smoke"]["ebno_db"]]
    return sorted(records, key=lambda r: (r["seed"], r["ebno_db"]))


def load_training_items(context, split):
    if split not in ("train", "validation"):
        raise ValueError("Training cannot preload development or confirmation observations.")
    records, items = records_for(context, split), []
    for number, record in enumerate(records, 1):
        channel = load_channel(context["cache_dir"], record, context["source_cache"]["identity_id"])
        nodes, edges = feature_tensors(channel, context["config"])
        items.append(dict(split=split, seed=record["seed"], ebno_db=record["ebno_db"], file=record["file"], sha256=record["sha256"],
                          n0=float(channel["n0"]), nodes=nodes, edges=edges,
                          llr=channel["methods"]["ep5"]["llr"][0].clone(), labels=coded_labels(channel)[0].clone()))
        print(f"Preload {split} {number}/{len(records)} seed={record['seed']} Eb/N0={record['ebno_db']:g}", flush=True)
    if not items:
        raise ValueError("Missing complete training/validation observations.")
    return items
