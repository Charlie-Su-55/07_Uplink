"""Fit small positive conditional LLR scales using only cached training channels."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
import torch.nn.functional as functional

from evaluation.evaluate_paper_soft_output import file_hash, fit_positive_scale
from evaluation.evaluate_sionna_bler import save_json
from models.conditional_llr_scaling import CANDIDATES, bound_statistics, make_model, scale_llr


def require_split(items, split):
    if not items or any(item.get("split") != split for item in items):
        raise ValueError(f"Only nonempty {split} inputs are permitted here.")


def fit_constants(train_items, settings):
    require_split(train_items, "train")
    signed = torch.cat([(item["llr"] * (2 * item["labels"] - 1)).reshape(-1, 4)
                        for item in train_items])
    return dict(global_scale=fit_positive_scale(signed, settings),
                per_bit_scale=fit_positive_scale(signed, settings, per_bit=True),
                train_seeds=sorted({item["seed"] for item in train_items}),
                ebno_dbs=sorted({item["ebno_db"] for item in train_items}),
                fit_sources=[{key: item[key] for key in ("seed", "ebno_db", "file", "sha256")}
                             for item in train_items],
                fit_split="train", settings=settings)


def validate_fit_sources(train_items, normalization, constants):
    require_split(train_items, "train")
    expected_seeds = sorted({item["seed"] for item in train_items})
    expected_sources = {(item["file"], item["sha256"]) for item in train_items}
    for artifact in (normalization, constants):
        if (artifact.get("fit_split") != "train" or artifact.get("train_seeds") != expected_seeds
                or {(source["file"], source["sha256"]) for source in artifact.get("fit_sources", [])} != expected_sources):
            raise ValueError("Normalization and constant fits must use exactly the selected training inputs.")


def make_sampling_plan(train_items, training_seed, steps, re_per_step):
    require_split(train_items, "train")
    if steps < 1 or re_per_step < 1:
        raise ValueError("Training steps and RE count must be positive.")
    by_seed = {}
    for index, item in enumerate(train_items):
        if item["nodes"].shape[0] < re_per_step:
            raise ValueError("Requested RE count exceeds a complete cached channel.")
        by_seed.setdefault(item["seed"], []).append(index)
    seeds = sorted(by_seed)
    for indices in by_seed.values():
        indices.sort(key=lambda index: train_items[index]["ebno_db"])
    generator = torch.Generator(device="cpu").manual_seed(training_seed + 1907)
    plan, digest = [], hashlib.sha256()
    for step in range(steps):
        seed = seeds[int(torch.randint(len(seeds), (), generator=generator))]
        choices = by_seed[seed]
        index = choices[int(torch.randint(len(choices), (), generator=generator))]
        selected = torch.randperm(train_items[index]["nodes"].shape[0], generator=generator)[:re_per_step]
        plan.append(dict(item_index=index, re_indices=selected))
        digest.update(json.dumps([step, seed, train_items[index]["ebno_db"], train_items[index]["sha256"]]).encode())
        digest.update(selected.numpy().tobytes())
    return plan, digest.hexdigest()


@torch.inference_mode()
def validate(model, validation_items, device, chunk_size, *, saturation_tolerance=1e-5, require_identity=False):
    require_split(validation_items, "validation")
    model.eval()
    total_loss, raw_loss, count = 0.0, 0.0, 0
    lower_count, upper_count, near_lower_count, near_upper_count = 0, 0, 0, 0
    alpha_sum, alpha_squares, bit_errors = 0.0, 0.0, 0
    observed_min, observed_max = float("inf"), 0.0
    per_channel = []
    for item in validation_items:
        channel_loss, channel_count = 0.0, 0
        for start in range(0, item["nodes"].shape[0], chunk_size):
            indices = slice(start, start + chunk_size)
            alpha = model(item["nodes"][indices].to(device), item["edges"][indices].to(device))
            llr, labels = item["llr"][indices].to(device), item["labels"][indices].to(device)
            scaled = scale_llr(llr, alpha)
            if require_identity and (not torch.equal(alpha, torch.ones_like(alpha)) or not torch.equal(scaled, llr)):
                raise ValueError("Fresh conditional scale must exactly reproduce alpha=1 and raw EP LLRs.")
            bit_errors += int(((scaled > 0) != labels.bool()).sum())
            alpha_sum += float(alpha.double().sum())
            alpha_squares += float(alpha.double().square().sum())
            loss = functional.binary_cross_entropy_with_logits(scaled.double(), labels.double(), reduction="sum")
            raw = functional.binary_cross_entropy_with_logits(llr.double(), labels.double(), reduction="sum")
            channel_loss += float(loss)
            channel_count += labels.numel()
            raw_loss += float(raw)
            bounds = bound_statistics(alpha, model.alpha_min, model.alpha_max, saturation_tolerance)
            lower_count += bounds["lower_count"]
            upper_count += bounds["upper_count"]
            near_lower_count += bounds["near_lower_count"]
            near_upper_count += bounds["near_upper_count"]
            observed_min, observed_max = min(observed_min, bounds["min"]), max(observed_max, bounds["max"])
        per_channel.append(dict(seed=item["seed"], ebno_db=item["ebno_db"], bce_nats=channel_loss / channel_count))
        total_loss += channel_loss
        count += channel_count
    alpha_mean = alpha_sum / count
    return dict(bce_nats=total_loss / count, raw_bce_nats=raw_loss / count, coded_bits=count,
                coded_bit_errors=bit_errors, coded_ber=bit_errors / count,
                alpha_mean=alpha_mean, alpha_std=max(alpha_squares / count - alpha_mean ** 2, 0.0) ** 0.5,
                near_lower_boundary_count=near_lower_count, near_upper_boundary_count=near_upper_count,
                near_lower_boundary_fraction=near_lower_count / count, near_upper_boundary_fraction=near_upper_count / count,
                lower_boundary_fraction=lower_count / count, upper_boundary_fraction=upper_count / count,
                saturation_tolerance=saturation_tolerance, initial_identity_checked=require_identity,
                per_channel=per_channel, alpha_min=observed_min, alpha_max=observed_max,
                lower_boundary_count=lower_count, upper_boundary_count=upper_count,
                hard_decisions_unchanged=True)


def save_checkpoint(path, model, config, context, normalization, candidate, training_seed, step, metrics, plan_hash):
    payload = dict(schema_version=1, kind="cached_conditional_llr_scaling_v1", candidate=candidate,
                   training_seed=training_seed, step=step, run_identity_id=context["run_identity_id"],
                   source_cache_identity_id=context["cache_manifest"]["identity_id"],
                   normalization=normalization, config=config, validation=metrics,
                   sampling_plan_sha256=plan_hash, parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                   model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if any(not torch.equal(value, reloaded["model_state"][name]) for name, value in payload["model_state"].items()):
            raise ValueError("Checkpoint serialization changed model weights.")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return dict(file=str(path), sha256=file_hash(path), step=step, validation_bce_nats=metrics["bce_nats"])


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def elapsed_since(started, device):
    synchronize(device)
    return time.monotonic() - started


def train_candidate(candidate, training_seed, train_items, validation_items, config, context, normalization, output, artifact_root=None):
    require_split(train_items, "train")
    require_split(validation_items, "validation")
    if {item["seed"] for item in train_items} & {item["seed"] for item in validation_items}:
        raise ValueError("Training and validation channels overlap.")
    settings = config["training"]
    steps = config["smoke"]["training_steps"] if config["mode"] == "smoke" else settings["steps"]
    plan, plan_hash = make_sampling_plan(train_items, training_seed, steps, settings["re_per_step"])
    torch.manual_seed(training_seed)
    device = config["runtime"]["device"]
    model = make_model(candidate, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    synchronize(device)
    started = time.monotonic()
    tolerance = config["model"].get("saturation_tolerance", 1e-5)
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
        alpha = model(nodes, edges)
        loss = functional.binary_cross_entropy_with_logits(scale_llr(llr, alpha), labels)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite cached-scale training loss.")
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
                parameter_count=sum(parameter.numel() for parameter in model.parameters()), history=history,
                sampling_plan_sha256=plan_hash, elapsed_seconds=elapsed_since(started, device),
                sampling="Uniform channel seed, then uniform cached Eb/N0, then REs without replacement; same plan for every candidate.")


def training_manifest_digest(manifest):
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_manifest(path, manifest):
    manifest["content_sha256"] = training_manifest_digest(manifest)
    save_json(path, manifest)


def reuse_manifest(directory, context, config):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("content_sha256") != training_manifest_digest(manifest)
            or manifest.get("run_identity_id") != context["run_identity_id"]
            or manifest.get("run_identity") != context["run_identity"]
            or manifest.get("source_cache_identity_id") != context["cache_manifest"]["identity_id"]
            or manifest.get("source_cache") != context["source_cache"]
            or manifest.get("split_plan") != context["split_plan"] or manifest.get("config") != config):
        raise ValueError("Refusing incomplete or mismatched conditional-scale training reuse.")
    from evaluation.cached_llr_common import records_for
    selected = [dict(record, split="train") for record in records_for(context, "train")]
    validate_fit_sources(selected, manifest["normalization"], manifest["constants"])
    for variant, size in (("global_scale", 1), ("per_bit_scale", 4)):
        fitted = manifest["constants"][variant]
        alpha = torch.tensor(fitted["alpha"], dtype=torch.float64)
        if (alpha.shape != (size,) or not torch.isfinite(alpha).all()
                or (alpha < config["model"]["alpha_min"]).any() or (alpha > config["model"]["alpha_max"]).any()):
            raise ValueError("Invalid cached train-only constant scale.")
    if manifest["constants"]["settings"] != config["constant_fit"]:
        raise ValueError("Constant fit changed its fixed optimization settings.")
    seeds = [config["smoke"]["training_seed"]] if config["mode"] == "smoke" else config["training"]["seeds"]
    expected = {(candidate, seed) for candidate in config["candidates"] for seed in seeds}
    actual = [(record["candidate"], record["training_seed"]) for record in manifest["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Training manifest has missing/duplicate candidates or seeds.")
    steps = config["smoke"]["training_steps"] if config["mode"] == "smoke" else config["training"]["steps"]
    for record in manifest["records"]:
        if record["last"]["step"] != steps or record["best"]["step"] not in [row["step"] for row in record["history"]]:
            raise ValueError("Missing actual best/last training steps.")
        if record["best"]["validation_bce_nats"] != min(row["validation"]["bce_nats"] for row in record["history"]):
            raise ValueError("Best selection was not based on validation BCE.")
        for selection in ("best", "last"):
            artifact = record[selection]
            path = (directory / artifact["file"]).resolve()
            if not path.is_relative_to(directory) or file_hash(path) != artifact["sha256"]:
                raise ValueError("Conditional-scale checkpoint path/hash mismatch.")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            expected_metadata = dict(schema_version=1, kind="cached_conditional_llr_scaling_v1",
                                     candidate=record["candidate"], training_seed=record["training_seed"],
                                     step=artifact["step"], run_identity_id=context["run_identity_id"],
                                     source_cache_identity_id=context["cache_manifest"]["identity_id"],
                                     normalization=manifest["normalization"], config=config,
                                     sampling_plan_sha256=record["sampling_plan_sha256"], parameter_count=record["parameter_count"])
            if any(checkpoint.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError("Conditional-scale checkpoint metadata mismatch.")
            model = make_model(record["candidate"], config)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            if any(not torch.isfinite(value).all() for value in checkpoint["model_state"].values()):
                raise ValueError("Non-finite cached checkpoint weights.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/cached_llr_scaling.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), default="development")
    parser.add_argument("--reuse-complete", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    from evaluation.cached_llr_common import (
        fit_normalizer, load_config, load_prepared, normalize_nodes, prepare, run_paths,
    )
    config = load_config(args.config, mode=args.mode)
    context = prepare(config)
    if args.preflight:
        print(f"Cached-scale preflight PASS identity={context['run_identity_id']}", flush=True)
        return 0
    directory = run_paths(config, args.run_id)["training"]
    if directory.exists():
        if not args.reuse_complete:
            raise FileExistsError(f"Training output exists; use a new run ID or --reuse-complete: {directory}")
        reuse_manifest(directory, context, config)
        print(f"Reused complete training: {directory}", flush=True)
        return 0
    train_items = load_prepared(context, "train")
    validation_items = load_prepared(context, "validation")
    normalization = fit_normalizer(train_items, config)
    constants = fit_constants(train_items, config["constant_fit"])
    validate_fit_sources(train_items, normalization, constants)
    for item in train_items + validation_items:
        item["nodes"] = normalize_nodes(item["nodes"], normalization, config)
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "manifest.json"
    manifest = dict(schema_version=1, status="running", run_id=args.run_id, config=config,
                    run_identity=context["run_identity"], run_identity_id=context["run_identity_id"],
                    source_cache=context["source_cache"], source_cache_identity_id=context["cache_manifest"]["identity_id"],
                    split_plan=context["split_plan"], normalization=normalization, constants=constants, records=[])
    write_manifest(manifest_path, manifest)
    started = time.monotonic()
    try:
        seeds = [config["smoke"]["training_seed"]] if args.mode == "smoke" else config["training"]["seeds"]
        for candidate in config["candidates"]:
            if candidate not in CANDIDATES:
                raise ValueError("Unknown training candidate.")
            for training_seed in seeds:
                output = directory / candidate / f"seed_{training_seed}"
                record = train_candidate(candidate, training_seed, train_items, validation_items,
                                         config, context, normalization, output, artifact_root=directory)
                manifest["records"].append(record)
                write_manifest(manifest_path, manifest)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started)
        write_manifest(manifest_path, manifest)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        write_manifest(manifest_path, manifest)
        raise
    print(f"Complete cached-scale training: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
