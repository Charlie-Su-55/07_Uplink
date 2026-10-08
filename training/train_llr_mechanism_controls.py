"""Train matched-capacity mechanism controls on the original train24 cache only."""

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import torch
import torch.nn.functional as functional

from evaluation.cached_llr_common import fit_normalizer, normalize_nodes
from evaluation.evaluate_paper_soft_output import file_hash, fit_positive_scale
from models.conditional_llr_scaling import scale_llr
from models.llr_mechanism_controls import CANDIDATES, MASKED_FEATURES, interpolate_noise_alpha, make_model
from training.train_cached_llr_scaling import (
    elapsed_since, fit_constants, make_sampling_plan, require_split, save_checkpoint as save_base_checkpoint,
    synchronize, training_manifest_digest, validate, validate_fit_sources, write_manifest,
)


KIND = "llr_mechanism_controls_v1"
DEFAULT_CONFIG = "configs/evaluation/llr_mechanism_controls.yaml"


def fit_noise_per_bit(train_items, config):
    require_split(train_items, "train")
    settings = config["noise_conditioning"]
    if (settings["coordinate"], settings["interpolation"], settings["outside_range"]) != ("log_n0", "linear_log_alpha", "endpoint_hold"):
        raise ValueError("Require the prespecified log-N0/log-alpha interpolation and endpoint hold.")
    groups = {}
    sources = []
    for item in train_items:
        noise = float(item["n0"])
        if not math.isfinite(noise) or noise <= 0:
            raise ValueError("Actual cached N0 must be finite and positive.")
        groups.setdefault(item["ebno_db"], []).append(item)
        sources.append(dict(**{key: item[key] for key in ("seed", "ebno_db", "file", "sha256")}, n0=noise))
    if set(groups) != set(config["protocol"]["ebno_dbs"]) or len(groups) != 3:
        raise ValueError("Noise-conditioned fitting needs all three training Eb/N0 points.")
    knots = []
    for ebno, items in sorted(groups.items()):
        noise = float(items[0]["n0"])
        if any(not math.isclose(float(item["n0"]), noise, rel_tol=settings["n0_rtol"], abs_tol=settings["n0_atol"]) for item in items):
            raise ValueError("Actual N0 differs across training channels at the same Eb/N0.")
        signed = torch.cat([(item["llr"] * (2 * item["labels"] - 1)).reshape(-1, 4) for item in items])
        fitted = fit_positive_scale(signed, config["constant_fit"], per_bit=True)
        knots.append(dict(ebno_db=ebno, n0=noise, alpha=fitted["alpha"], fit=fitted,
                          train_seeds=sorted({item["seed"] for item in items})))
    knots.sort(key=lambda knot: knot["n0"])
    result = dict(fit_split="train", train_seeds=sorted({item["seed"] for item in train_items}),
                  fit_sources=sources, knots=knots, alpha_min=config["model"]["alpha_min"],
                  alpha_max=config["model"]["alpha_max"], coordinate="log_n0",
                  interpolation="linear_log_alpha_log_n0_endpoint_hold", settings=settings,
                  optimizer_settings=config["constant_fit"])
    for knot in knots:
        interpolate_noise_alpha(knot["n0"], result)
    return result


def fit_training_constants(train_items, config):
    result = fit_constants(train_items, config["constant_fit"])
    result["noise_per_bit"] = fit_noise_per_bit(train_items, config)
    return result


def save_checkpoint(path, model, config, context, normalization, candidate, seed, step, metrics, plan_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        save_base_checkpoint(temporary, model, config, context, normalization, candidate, seed, step, metrics, plan_hash)
        payload = torch.load(temporary, weights_only=True, map_location="cpu")
        payload.update(kind=KIND, code_identity=context["code_identity"], environment=context["environment"],
                       masked_feature_indices=list(MASKED_FEATURES[candidate]),
                       mask_stage="after shared train-only standardization and clipping; same six input columns")
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return dict(file=str(path), sha256=file_hash(path), step=step, validation_bce_nats=metrics["bce_nats"])


def train_candidate(candidate, training_seed, train_items, validation_items, config, context, normalization, output, artifact_root=None):
    require_split(train_items, "train")
    require_split(validation_items, "validation")
    if {item["seed"] for item in train_items} & {item["seed"] for item in validation_items}:
        raise ValueError("Mechanism train and validation channels overlap.")
    settings = config["training"]
    steps = config["smoke"]["training_steps"] if config["mode"] == "smoke" else settings["steps"]
    plan, plan_hash = make_sampling_plan(train_items, training_seed, steps, settings["re_per_step"])
    torch.manual_seed(training_seed)
    device = config["runtime"]["device"]
    model = make_model(candidate, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    tolerance = config["model"]["saturation_tolerance"]
    synchronize(device)
    started = time.monotonic()
    metrics = validate(model, validation_items, device, settings["validation_chunk"],
                       saturation_tolerance=tolerance, require_identity=True)
    history = [dict(step=0, training_loss=None, elapsed_seconds=elapsed_since(started, device), validation=metrics)]
    best_path, last_path = Path(output) / "best.pth", Path(output) / "last.pth"
    best = save_checkpoint(best_path, model, config, context, normalization, candidate, training_seed, 0, metrics, plan_hash)
    training_loss = 0.0
    print(f"train {candidate} seed={training_seed} step=0/{steps} validation_BCE={metrics['bce_nats']:.7g}", flush=True)
    for step, sample in enumerate(plan, 1):
        model.train()
        item, indices = train_items[sample["item_index"]], sample["re_indices"]
        nodes, edges = item["nodes"][indices].to(device), item["edges"][indices].to(device)
        llr, labels = item["llr"][indices].to(device), item["labels"][indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = functional.binary_cross_entropy_with_logits(scale_llr(llr, model(nodes, edges)), labels)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite mechanism-control training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        training_loss += float(loss.detach())
        if step % settings["validate_every"] == 0 or step == steps:
            metrics = validate(model, validation_items, device, settings["validation_chunk"], saturation_tolerance=tolerance)
            interval = step - history[-1]["step"]
            history.append(dict(step=step, training_loss=training_loss / interval,
                                elapsed_seconds=elapsed_since(started, device), validation=metrics))
            training_loss = 0.0
            if metrics["bce_nats"] < best["validation_bce_nats"]:
                best = save_checkpoint(best_path, model, config, context, normalization, candidate,
                                       training_seed, step, metrics, plan_hash)
            print(f"train {candidate} seed={training_seed} step={step}/{steps} validation_BCE={metrics['bce_nats']:.7g} "
                  f"best_step={best['step']} elapsed={elapsed_since(started, device):.1f}s", flush=True)
    last = save_checkpoint(last_path, model, config, context, normalization, candidate, training_seed, steps, metrics, plan_hash)
    if artifact_root is not None:
        for artifact in (best, last):
            artifact["file"] = Path(artifact["file"]).relative_to(artifact_root).as_posix()
    return dict(candidate=candidate, training_seed=training_seed, best=best, last=last,
                parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                masked_feature_indices=list(MASKED_FEATURES[candidate]), history=history,
                sampling_plan_sha256=plan_hash, elapsed_seconds=elapsed_since(started, device),
                sampling="Uniform train channel, then cached Eb/N0, then REs without replacement; matched plan for all controls")


def validate_training(directory, context, config):
    from evaluation.llr_mechanism_common import records_for

    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = dict(schema_version=1, kind=KIND, status="complete", config=config,
                             run_identity_id=context["run_identity_id"], run_identity=context["run_identity"],
                             source_cache_identity_id=context["cache_manifest"]["identity_id"],
                             source_cache=context["source_cache"], split_plan=context["split_plan"])
    if (any(manifest.get(key) != value for key, value in expected_manifest.items())
            or manifest.get("content_sha256") != training_manifest_digest(manifest)):
        raise ValueError("Incomplete or changed mechanism-control training identity/content.")
    selected = [dict(record, split="train") for record in records_for(context, "train")]
    validate_fit_sources(selected, manifest["normalization"], manifest["constants"])
    noise_fit = manifest["constants"]["noise_per_bit"]
    validate_fit_sources(selected, manifest["normalization"], noise_fit)
    if manifest["constants"]["settings"] != config["constant_fit"] or noise_fit["settings"] != config["noise_conditioning"]:
        raise ValueError("Changed fixed constant/noise fitting settings.")
    if noise_fit["optimizer_settings"] != config["constant_fit"] or {knot["ebno_db"] for knot in noise_fit["knots"]} != set(config["protocol"]["ebno_dbs"]):
        raise ValueError("Noise baseline does not use the three fixed training points.")
    for knot in noise_fit["knots"]:
        interpolate_noise_alpha(knot["n0"], noise_fit)
        if knot["alpha"] != knot["fit"]["alpha"]:
            raise ValueError("Noise interpolation knots differ from fitted training scales.")
        matching = [source for source in noise_fit["fit_sources"] if source["ebno_db"] == knot["ebno_db"]]
        if not matching or any(not math.isclose(float(source["n0"]), knot["n0"], rel_tol=noise_fit["settings"]["n0_rtol"],
                                                abs_tol=noise_fit["settings"]["n0_atol"]) for source in matching):
            raise ValueError("Noise knot differs from recorded actual training N0 values.")
    for variant, count in (("global_scale", 1), ("per_bit_scale", 4)):
        alpha = torch.tensor(manifest["constants"][variant]["alpha"], dtype=torch.float64)
        if alpha.shape != (count,) or not torch.isfinite(alpha).all() or (alpha < .05).any() or (alpha > 8).any():
            raise ValueError("Invalid train-only positive constant scale.")
    seeds = [config["smoke"]["training_seed"]] if config["mode"] == "smoke" else config["training"]["seeds"]
    expected = {(candidate, seed) for candidate in config["candidates"] for seed in seeds}
    actual = [(record["candidate"], record["training_seed"]) for record in manifest["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing/duplicate mechanism candidate and training-seed records.")
    steps = config["smoke"]["training_steps"] if config["mode"] == "smoke" else config["training"]["steps"]
    expected_steps = [0] + list(range(config["training"]["validate_every"], steps + 1, config["training"]["validate_every"]))
    if expected_steps[-1] != steps:
        expected_steps.append(steps)
    for record in manifest["records"]:
        if [row["step"] for row in record["history"]] != expected_steps or record["last"]["step"] != steps:
            raise ValueError("Incomplete fixed-budget training/validation history.")
        best_row = min(record["history"], key=lambda row: row["validation"]["bce_nats"])
        if record["best"]["step"] != best_row["step"] or record["best"]["validation_bce_nats"] != best_row["validation"]["bce_nats"]:
            raise ValueError("Best checkpoint was not selected solely by validation BCE.")
        for selection in ("best", "last"):
            artifact = record[selection]
            path = (directory / artifact["file"]).resolve()
            if not path.is_relative_to(directory) or file_hash(path) != artifact["sha256"]:
                raise ValueError("Mechanism checkpoint path/hash mismatch.")
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            expected_metadata = dict(schema_version=1, kind=KIND, candidate=record["candidate"], training_seed=record["training_seed"],
                                     step=artifact["step"], run_identity_id=context["run_identity_id"],
                                     source_cache_identity_id=context["cache_manifest"]["identity_id"], config=config,
                                     normalization=manifest["normalization"], code_identity=manifest["code_identity"],
                                     environment=manifest["environment"], sampling_plan_sha256=record["sampling_plan_sha256"],
                                     parameter_count=record["parameter_count"], masked_feature_indices=list(MASKED_FEATURES[record["candidate"]]))
            if any(checkpoint.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError("Mechanism checkpoint metadata mismatch.")
            model = make_model(record["candidate"], config)
            if not torch.equal(checkpoint["model_state"]["feature_mask"], model.feature_mask):
                raise ValueError("Checkpoint changed its prespecified feature mask.")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            if sum(parameter.numel() for parameter in model.parameters()) != record["parameter_count"] or any(not torch.isfinite(value).all() for value in checkpoint["model_state"].values()):
                raise ValueError("Invalid mechanism checkpoint parameter count or values.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "confirmation"), default="confirmation")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--reuse-complete", action="store_true")
    args = parser.parse_args(argv)
    from evaluation.llr_mechanism_common import load_config, load_training_items, prepare, run_paths

    config = load_config(args.config, mode=args.mode)
    context = prepare(config)
    if args.preflight:
        print(f"Mechanism preflight PASS identity={context['run_identity_id']}", flush=True)
        return 0
    directory = run_paths(config, args.run_id)["training"]
    if directory.exists():
        if not args.reuse_complete:
            raise FileExistsError("Mechanism training output exists; use a new run ID or --reuse-complete.")
        validate_training(directory, context, config)
        print(f"Reused complete mechanism training: {directory}", flush=True)
        return 0
    train_items, validation_items = load_training_items(context, "train"), load_training_items(context, "validation")
    normalization = fit_normalizer(train_items, config)
    constants = fit_training_constants(train_items, config)
    validate_fit_sources(train_items, normalization, constants)
    validate_fit_sources(train_items, normalization, constants["noise_per_bit"])
    for item in train_items + validation_items:
        item["nodes"] = normalize_nodes(item["nodes"], normalization, config)
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "manifest.json"
    manifest = dict(schema_version=1, kind=KIND, status="running", run_id=args.run_id, config=config,
                    run_identity=context["run_identity"], run_identity_id=context["run_identity_id"],
                    code_identity=context["code_identity"], environment=context["environment"],
                    source_cache=context["source_cache"], source_cache_identity_id=context["cache_manifest"]["identity_id"],
                    split_plan=context["split_plan"], normalization=normalization, constants=constants, records=[])
    write_manifest(manifest_path, manifest)
    started = time.monotonic()
    try:
        seeds = [config["smoke"]["training_seed"]] if args.mode == "smoke" else config["training"]["seeds"]
        for candidate in config["candidates"]:
            if candidate not in CANDIDATES:
                raise ValueError("Unknown mechanism candidate.")
            for training_seed in seeds:
                record = train_candidate(candidate, training_seed, train_items, validation_items, config, context,
                                         normalization, directory / candidate / f"seed_{training_seed}", artifact_root=directory)
                manifest["records"].append(record)
                write_manifest(manifest_path, manifest)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started)
        write_manifest(manifest_path, manifest)
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        write_manifest(manifest_path, manifest)
        raise
    print(f"Complete mechanism training: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
