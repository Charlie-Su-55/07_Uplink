"""Fixed contracts, provenance and atomic channel storage for Mode-B soft outputs."""

import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile

import torch

from evaluation.evaluate_sionna_bler import save_json
from link_level.detector_adapter import llrs_to_native_order
from link_level.partner_a1_adapter import file_sha256, validate_neural_checkpoint
from link_level.sionna_ce import distribution_id, validate_covariance_artifact
from link_level.sionna_reference import load_config, validate_reference_config


DEFAULT_CONFIG = "configs/evaluation/paper_soft_output_dev.yaml"
METHODS = ("sionna_lmmse", "ep5", "gt_best", "gt_last", "detr_best", "detr_last")
SCHEMA_VERSION = 1
LLR_SIGN = "log P(bit=1)/P(bit=0); hard bit = llr > 0"
LLR_MAPPING = "llr[B,Ndata,UE,Qm] -> permute(0,2,1,3).reshape(B,UE,1,G); codec squeezes axis2; no clip/rescale"
REPOSITORY = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def canonical_file_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_experiment_config(path=DEFAULT_CONFIG, mode="development"):
    text = Path(path).read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        config = load_config(path)
    config = copy.deepcopy(config)
    if config.get("schema_version") != SCHEMA_VERSION or mode not in ("smoke", "development"):
        raise ValueError("Unknown soft-output configuration schema/mode.")
    if config["batch_size"] != 1 or config["re_chunk"] < 1:
        raise ValueError("Require one complete slot/channel and positive RE chunk.")
    expected = {"gt_best": "gt_ep", "gt_last": "gt_ep", "detr_best": "detr_ep", "detr_last": "detr_ep"}
    if {name: item["arch"] for name, item in config["checkpoints"].items()} != expected:
        raise ValueError("Expected the four explicit GT/DETR best/last checkpoint aliases.")
    if set(config["splits"]) != {"calibration", "development"}:
        raise ValueError("Need distinct calibration/development splits.")
    for split in config["splits"].values():
        if not isinstance(split["channels"], int) or split["channels"] < 1 or not 0 <= split["start_seed"] < 2 ** 31:
            raise ValueError("Invalid channel count/seed candidate.")
    ebnos = config["ebno_dbs"]
    if not ebnos or any(not math.isfinite(value) for value in ebnos) or len(set(ebnos)) != len(ebnos):
        raise ValueError("Eb/N0 points must be finite and unique.")
    fit = config["calibration"]
    if (fit["method"] != "bounded_derivative_bisection" or fit["precision"] != "float64"
            or not 0 < fit["alpha_min"] < 1 < fit["alpha_max"]
            or not all(math.isfinite(fit[key]) and fit[key] > 0 for key in
                       ("alpha_min", "alpha_max", "alpha_tolerance", "gradient_tolerance"))
            or not isinstance(fit["max_iterations"], int) or fit["max_iterations"] < 1):
        raise ValueError("Invalid predeclared positive-scale fitting settings.")
    if (config["runtime"]["precision"] != "single" or config["runtime"]["torch_threads"] < 1
            or not 0 < config["bootstrap"]["confidence"] < 1 or config["bootstrap"]["replicates"] < 1):
        raise ValueError("Invalid precision/numeric/bootstrap settings.")
    for value in config["features"].values():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Feature tolerances must be finite and positive.")
    config["mode"] = mode
    if mode == "smoke":
        config["ebno_dbs"] = [-10.5]
        for split in config["splits"].values():
            split["channels"] = 1
    return config


def run_paths(config, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or run_id in (".", ".."):
        raise ValueError("run-id must be a simple name without directory separators.")
    return {name: Path(config["outputs"][name + "_root"]) / run_id
            for name in ("cache", "results", "calibration", "logs")}


def collect_seeds(value):
    categories = {name: set() for name in ("train_seeds", "validation_seeds", "calibration_seeds", "evaluation_seeds")}

    def walk(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in categories and isinstance(item, list):
                    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in item):
                        raise ValueError(f"Invalid archived seed list: {key}")
                    categories[key].update(item)
                elif key in ("seed", "channel_seed", "evaluation_seed") and isinstance(item, int):
                    categories["evaluation_seeds"].add(item)
                elif key in ("seeds", "channel_seeds") and isinstance(item, list) and all(isinstance(seed, int) for seed in item):
                    categories["evaluation_seeds"].update(item)
                elif isinstance(item, (dict, list)):
                    walk(item)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    walk(item)
    walk(value)
    return {key: sorted(seeds) for key, seeds in categories.items()}


def archive_evidence(snapshot_dir):
    root = Path(snapshot_dir)
    index = json.loads((root / "snapshot_index.json").read_text(encoding="utf-8"))
    sources, excluded, reference = [], set(), None
    indexed = {item["archived"]: item for item in index["sources"]}
    for relative in indexed:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Missing indexed archived evidence: {relative}")
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        document = json.loads(path.read_text(encoding="utf-8"))
        current_hash = file_sha256(path)
        if relative in indexed:
            expected = indexed[relative]["archived_sha256"]
            if current_hash != expected and canonical_file_hash(path) != expected:
                raise ValueError(f"Archived evidence hash mismatch: {relative}")
        seeds = collect_seeds(document)
        for values in seeds.values():
            excluded.update(values)
        if any(seeds.values()):
            sources.append(dict(path=str(path), sha256=current_hash, canonical_sha256=canonical_file_hash(path), seeds=seeds))
        if relative == "training_history/paper_reference_gt_ep_smoke300.json":
            reference = document["metadata"]
    if not reference or not sources:
        raise ValueError("Incomplete archived training and evaluation evidence.")
    for path in sorted((root / "training_history").glob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        if not metadata.get("train_seeds") or not metadata.get("validation_seeds"):
            raise ValueError(f"Missing full historical channel seed lists: {path}")
    return dict(index=index, reference=reference, excluded_seeds=sorted(excluded), sources=sources,
                snapshot_index_sha256=canonical_file_hash(root / "snapshot_index.json"))


def seed_plan(config, evidence, covariance_seeds=(), checkpoint_metadata=()):
    used = set(evidence["excluded_seeds"]) | set(covariance_seeds)
    runtime_sources = []
    for metadata in checkpoint_metadata:
        seeds = set(metadata["train_seeds"]) | set(metadata["validation_seeds"])
        used.update(seeds)
        runtime_sources.append(dict(path=metadata.get("path"), train_seeds=metadata["train_seeds"],
                                    validation_seeds=metadata["validation_seeds"]))
    plan = dict(excluded_seeds=sorted(used), exclusion_sources=evidence["sources"],
                runtime_covariance_seeds=list(covariance_seeds), runtime_checkpoints=runtime_sources,
                independent_channel_definition="Unique channel seed; repeated Eb/N0 points share a channel cluster.")
    for name in ("calibration", "development"):
        specification = config["splits"][name]
        candidate, selected = specification["start_seed"], []
        while len(selected) < specification["channels"]:
            if candidate >= 2 ** 31:
                raise ValueError("Exhausted valid seed candidates.")
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
            candidate += 1
        plan[name] = selected
    return plan


def numeric_runtime(config, *, require_gpu=True):
    runtime = config["runtime"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(runtime["torch_threads"])
    torch.use_deterministic_algorithms(runtime["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = runtime["deterministic"]
    versions = {"python": platform.python_version()}
    for name in ("torch", "sionna", "numpy", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    differences = [f"{name}: expected {runtime['expected_' + name]}, found {versions[name]}"
                   for name in ("sionna", "torch") if versions[name] != runtime["expected_" + name]]
    for difference in differences:
        print("Environment difference: " + difference, flush=True)
    if versions["sionna"] != runtime["expected_sionna"]:
        raise RuntimeError("Use the existing Sionna 2.0.1 server environment; no dependency changes are performed.")
    if require_gpu and (not runtime["device"].startswith("cuda") or not torch.cuda.is_available()):
        raise RuntimeError("Full Mode-B cache generation requires the GPU server.")
    return dict(versions=versions, cuda=torch.version.cuda,
                gpus=[torch.cuda.get_device_name(number) for number in range(torch.cuda.device_count())],
                device=runtime["device"], deterministic=torch.are_deterministic_algorithms_enabled(),
                torch_threads=torch.get_num_threads(), allow_tf32=runtime["allow_tf32"],
                cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"], differences=differences,
                historical_bitwise_reproduction_claimed=False)


def source_identity():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPOSITORY).decode().split("\0")
    paths += [str(path.relative_to(REPOSITORY)).replace("\\", "/")
              for path in (REPOSITORY / "evaluation").glob("*paper_soft*.py")]
    hashes = {name: canonical_file_hash(REPOSITORY / name) for name in sorted(set(paths))
              if name.endswith(".py") and not name.startswith("tests/")}
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
                branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=REPOSITORY, text=True).strip(),
                dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPOSITORY, text=True).strip()),
                python_sources_lf_sha256=hashes)


def preflight(config):
    environment = numeric_runtime(config)
    evidence = archive_evidence(config["snapshot_dir"])
    reference_config = copy.deepcopy(load_config(config["reference_config"]))
    reference_config["general"].update(device=config["runtime"]["device"], precision=config["runtime"]["precision"])
    reference_config["channel_estimation"].update(mode="practical", covariance_path=config["covariance_path"])
    validate_reference_config(reference_config)
    if distribution_id(reference_config) != evidence["reference"]["distribution_id"]:
        raise ValueError("Frozen Mode-B distribution differs from archived training.")
    expected_hashes = {item["path"]: item["sha256"] for item in evidence["index"]["local_only_artifacts"] if item.get("exists")}
    paths = [config["covariance_path"]] + [item["path"] for item in config["checkpoints"].values()]
    for path in paths:
        if path not in expected_hashes or file_sha256(path) != expected_hashes[path]:
            raise ValueError(f"Server artifact SHA256 differs from snapshot_index.json: {path}")
    prior = torch.load(config["covariance_path"], map_location="cpu", weights_only=True)
    validate_covariance_artifact(prior, reference_config)
    prior_metadata = {key: value for key, value in prior.items() if key not in ("cov_mat_freq", "cov_mat_time")}
    prior_metadata.update(path=config["covariance_path"], sha256=expected_hashes[config["covariance_path"]])
    metadata = {}
    for name, specification in config["checkpoints"].items():
        checkpoint = torch.load(specification["path"], map_location="cpu", weights_only=True)
        validate_neural_checkpoint(checkpoint, specification["arch"], reference_config, prior_metadata["sha256"], [])
        metadata[name] = {key: value for key, value in checkpoint.items() if key not in ("model_state", "optimizer_state")}
        metadata[name].update(path=specification["path"], sha256=expected_hashes[specification["path"]])
    plan = seed_plan(config, evidence, prior["calibration_seeds"], metadata.values())
    evaluation_seeds = plan["calibration"] + plan["development"]
    for name, specification in config["checkpoints"].items():
        validate_neural_checkpoint(metadata[name], specification["arch"], reference_config, prior_metadata["sha256"], evaluation_seeds)
        if metadata[name]["grid"] != evidence["reference"]["grid"] or metadata[name]["codec"] != evidence["reference"]["codec"]:
            raise ValueError(f"Checkpoint grid/codec differs from archived profile: {name}")
    identity = dict(schema_version=SCHEMA_VERSION, experiment_config=config, environment=environment, source=source_identity(),
                    reference=dict(config=reference_config, grid=evidence["reference"]["grid"],
                                   codec=evidence["reference"]["codec"], covariance=prior_metadata),
                    distribution_id=distribution_id(reference_config), checkpoints=metadata,
                    snapshot_index_sha256=evidence["snapshot_index_sha256"], seed_plan=plan,
                    methods=list(METHODS), llr_sign=LLR_SIGN, llr_mapping=LLR_MAPPING,
                    Ruu="raw N0*I; CE whitening uses N0+sum_UE err_var, then identity covariance",
                    ebno="ebnodb2no(Eb/N0,Qm,payload/G,actual frozen pilot/data/CP ResourceGrid)")
    print(f"Preflight PASS: calibration={len(plan['calibration'])}, development={len(plan['development'])} unique channels; "
          f"checkpoint steps=" + str({name: item["step"] for name, item in metadata.items()}), flush=True)
    return dict(identity=identity, identity_id=digest(identity), seed_plan=plan)


def expected_cases(identity):
    return [(split, ebno, seed) for split in ("calibration", "development")
            for ebno in identity["experiment_config"]["ebno_dbs"] for seed in identity["seed_plan"][split]]


def channel_name(split, ebno, seed):
    token = f"{ebno:g}".replace("-", "m").replace(".", "p")
    return f"channels/{split}_ebno_{token}_seed_{seed}.pt"


def cpu_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tensors(item) for item in value]
    return value


def tensor_schema(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, torch.Tensor):
            result[name] = dict(shape=list(item.shape), dtype=str(item.dtype), bytes=item.numel() * item.element_size())
        elif isinstance(item, dict):
            result.update(tensor_schema(item, name + "."))
    return result


def validate_channel(payload, identity_id):
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "complete" or payload.get("identity_id") != identity_id:
        raise ValueError("Incomplete/foreign cache channel schema or identity.")
    if payload.get("split") not in ("calibration", "development"):
        raise ValueError("Invalid channel split.")
    coded, info, indices = payload["coded_bits"], payload["info_bits"], payload["data_indices"]
    if coded.ndim != 3 or info.ndim != 3 or coded.shape[:2] != info.shape[:2] or coded.shape[0] != 1:
        raise ValueError("Cache requires complete one-channel codewords.")
    batch_size, users, coded_length = coded.shape
    data_count = len(indices)
    if coded_length != 4 * data_count or not torch.equal(payload["native_data_ordinals"], torch.arange(data_count)):
        raise ValueError("Incomplete codeword/native data ordinal order.")
    if indices.ndim != 1 or indices.dtype != torch.int64 or (indices[1:] <= indices[:-1]).any() or (indices < 0).any():
        raise ValueError("Invalid original data_indices order.")
    if coded.dtype != torch.float32 or info.dtype != torch.float32 or not torch.all((coded == 0) | (coded == 1)) or not torch.all((info == 0) | (info == 1)):
        raise ValueError("Original bits must be binary float32 tensors.")
    if payload["n0"].shape != torch.Size([]) or payload["n0"].dtype != torch.float32 or payload["n0"].item() <= 0:
        raise ValueError("Invalid actual noise variance.")
    if set(payload["methods"]) != set(METHODS):
        raise ValueError("Incomplete receiver set.")
    for name, record in payload["methods"].items():
        if record["llr"].shape != (batch_size, data_count, users, 4) or record["llr"].dtype != torch.float32:
            raise ValueError(f"Wrong full LLR shape/dtype: {name}")
        if record["decoder_input_llr"].dtype != torch.float32 or not torch.equal(llrs_to_native_order(record["llr"]), record["decoder_input_llr"]):
            raise ValueError(f"Wrong decoder input ordering/values: {name}")
        if (record["decoded_payload_bits"].shape != info.shape or record["decoded_payload_bits"].dtype != torch.float32
                or not torch.all((record["decoded_payload_bits"] == 0) | (record["decoded_payload_bits"] == 1))
                or record["crc_ok"].shape != info.shape[:2] or record["crc_ok"].dtype != torch.bool
                or record["block_error"].dtype != torch.bool or record["payload_bit_errors"].dtype != torch.int64
                or record["total_payload_bits_per_ue"].dtype != torch.int64):
            raise ValueError(f"Invalid per-TB output: {name}")
        errors = (record["decoded_payload_bits"] != info).sum(-1)
        if (not torch.equal(errors, record["payload_bit_errors"]) or not torch.equal(errors > 0, record["block_error"])
                or not torch.equal(torch.full_like(errors, info.shape[-1]), record["total_payload_bits_per_ue"])):
            raise ValueError(f"Inconsistent per-TB counts: {name}")
    for key, shape in dict(z=(1, data_count, users), gram=(1, data_count, users, users),
                           ep5_mean=(1, data_count, users), ep5_variance=(1, data_count, users),
                           received_energy=(1, data_count), residual=(1, data_count), eta=(1, data_count, users)).items():
        feature = payload["features"][key]
        expected_dtype = torch.complex64 if key in ("z", "gram", "ep5_mean") else torch.float32
        if tuple(feature.shape) != shape or feature.dtype != expected_dtype:
            raise ValueError(f"Wrong feature shape/dtype: {key}")
    def check_tensors(node):
        for item in node.values():
            if isinstance(item, torch.Tensor) and (item.device.type != "cpu" or not torch.isfinite(item).all()):
                raise ValueError("Cache tensors must be finite CPU tensors.")
            if isinstance(item, dict):
                check_tensors(item)
    check_tensors(payload)
    if payload["metadata"]["tensor_shapes"] != tensor_schema({key: value for key, value in payload.items() if key != "metadata"}):
        raise ValueError("Cache tensor shape/dtype manifest mismatch.")
    axes = payload["metadata"].get("tensor_axes", {})
    if set(axes) != set(payload["metadata"]["tensor_shapes"]) or any(
            not isinstance(names, list) or len(names) != len(payload["metadata"]["tensor_shapes"][name]["shape"])
            or any(not isinstance(axis, str) or not axis for axis in names) for name, names in axes.items()):
        raise ValueError("Cache tensor axis manifest mismatch.")
    return payload


def load_channel(cache_dir, record, identity_id):
    if record.get("status") != "complete":
        raise ValueError("Channel was not marked complete.")
    root = Path(cache_dir).resolve()
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root) or path.suffix != ".pt" or file_sha256(path) != record["sha256"]:
        raise ValueError("Channel cache path/hash mismatch.")
    if path.stat().st_size != record["bytes"]:
        raise ValueError("Channel cache byte count mismatch.")
    payload = validate_channel(torch.load(path, map_location="cpu", weights_only=True), identity_id)
    if any(payload[key] != record[key] for key in ("split", "ebno_db", "seed")):
        raise ValueError("Channel index and payload coordinates differ.")
    return payload


def load_complete_cache(cache_dir):
    root = Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("Cache manifest is incomplete or has an unsupported schema.")
    if digest(manifest["identity"]) != manifest["identity_id"]:
        raise ValueError("Cache identity digest mismatch.")
    expected = set(expected_cases(manifest["identity"]))
    actual = [(record["split"], record["ebno_db"], record["seed"]) for record in manifest["channels"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Cache has missing/duplicate/unexpected channels.")
    plan = manifest["identity"]["seed_plan"]
    if manifest["seed_plan"] != plan:
        raise ValueError("Cache seed plan differs from its identity.")
    for split in ("calibration", "development"):
        seeds = plan[split]
        if (any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 31 for seed in seeds)
                or len(seeds) != len(set(seeds))
                or len(seeds) != manifest["identity"]["experiment_config"]["splits"][split]["channels"]):
            raise ValueError("Invalid complete cache seed list/count.")
    if set(plan["calibration"]) & set(plan["development"]):
        raise ValueError("Calibration/development seed leakage.")
    excluded = set(plan["excluded_seeds"]) | set(plan.get("runtime_covariance_seeds", []))
    for values in collect_seeds(plan.get("exclusion_sources", []) + plan.get("runtime_checkpoints", [])).values():
        excluded.update(values)
    if excluded.intersection(plan["calibration"] + plan["development"]):
        raise ValueError("Cache seeds overlap historical exclusions.")
    reference = manifest["identity"]["reference"]
    grid, codec = reference["grid"], reference["codec"]
    for record in manifest["channels"]:
        payload = load_channel(root, record, manifest["identity_id"])
        if (payload["coded_bits"].shape != (1, grid["num_ues"], codec["coded_bits_per_ue"])
                or payload["info_bits"].shape != (1, grid["num_ues"], codec["info_bits_per_ue"])
                or len(payload["data_indices"]) != grid["num_data_symbols"] or codec["bits_per_symbol"] != 4
                or (payload["data_indices"] >= grid["num_ofdm_symbols"] * grid["fft_size"]).any()):
            raise ValueError("Channel tensor shapes differ from the identity grid/codec.")
        if "pilot_symbols" in grid:
            expected_indices = torch.tensor([symbol * grid["fft_size"] + carrier
                                            for symbol in range(grid["num_ofdm_symbols"]) if symbol not in grid["pilot_symbols"]
                                            for carrier in range(grid["fft_size"])], dtype=torch.int64)
            if not torch.equal(payload["data_indices"], expected_indices):
                raise ValueError("Channel data_indices differ from the frozen pilot/data grid.")
    return manifest


def atomic_torch_save(path, payload, verify=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite channel cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if verify is not None:
            verify(temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return dict(sha256=file_sha256(path), bytes=path.stat().st_size)
