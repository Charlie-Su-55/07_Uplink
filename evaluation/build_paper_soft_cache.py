"""Build compact, complete-codeword Mode-B caches with one shared receiver frontend."""

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from data.preprocessing.whitening import CovarianceAwareFrontEnd
from evaluation.paper_soft_common import (
    DEFAULT_CONFIG, LLR_MAPPING, LLR_SIGN, METHODS, SCHEMA_VERSION,
    atomic_torch_save, channel_name, cpu_tensors, digest, expected_cases,
    load_channel, load_complete_cache, load_experiment_config, preflight,
    run_paths, save_json, tensor_schema, validate_channel,
)
from link_level.detector_adapter import llrs_to_native_order, make_detector_batch, whiten_diagonal_uncertainty
from link_level.partner_a1_adapter import load_neural_checkpoint
from link_level.sionna_reference import SionnaReferenceLink


@torch.no_grad()
def shared_statistics(batch, ordinals, ce_epsilon):
    if batch["metadata"]["csi"] != "practical":
        raise ValueError("Soft-output cache requires practical CSI.")
    batch_size, symbols, carriers, antennas = batch["y"].shape
    users = batch["h_hat"].shape[-1]
    indices = batch["data_indices"][ordinals.to(batch["data_indices"].device)]
    received = batch["y"].reshape(batch_size, symbols * carriers, antennas)[:, indices].unsqueeze(1)
    estimate = batch["h_hat"].reshape(batch_size, symbols * carriers, antennas, users)[:, indices].unsqueeze(1)
    error = batch["err_var"].reshape_as(batch["h_hat"]).reshape(batch_size, symbols * carriers, antennas, users)[:, indices].unsqueeze(1)
    received_white, estimate_white = whiten_diagonal_uncertainty(received, estimate, error, batch["n0"])
    identity = torch.eye(antennas, dtype=estimate.dtype, device=estimate.device).expand(batch_size, -1, -1)
    statistics = CovarianceAwareFrontEnd("single")(received_white, estimate_white, identity)
    error_power = error.sum(-2)
    eta = error_power / (estimate.abs().square().sum(-2) + error_power + ce_epsilon)
    return dict(z=statistics["z"][:, 0], gram=statistics["gram"][:, 0],
                received_energy=received_white.abs().square().sum(-1)[:, 0], eta=eta[:, 0])


def posterior_residual(statistics, mean, variance, antennas, absolute_tolerance, relative_tolerance):
    matched = (mean.conj() * statistics["z"]).sum(-1).real
    projected = torch.einsum("...k,...kl,...l->...", mean.conj(), statistics["gram"], mean).real
    uncertainty = (statistics["gram"].diagonal(dim1=-2, dim2=-1).real * variance).sum(-1)
    residual = (statistics["received_energy"] - 2 * matched + projected + uncertainty) / antennas
    magnitude = (statistics["received_energy"] + 2 * matched.abs() + projected.abs() + uncertainty.abs()) / antennas
    tolerance = absolute_tolerance + relative_tolerance * magnitude
    if not torch.isfinite(residual).all() or (variance < 0).any() or (residual < -tolerance).any():
        raise ValueError("Invalid posterior residual; negative values exceed the declared float32 tolerance.")
    return residual


def tb_record(llr, decoder_input, decoded, crc, info):
    errors = (decoded != info).sum(-1)
    return cpu_tensors(dict(llr=llr, decoder_input_llr=decoder_input, decoded_payload_bits=decoded,
                            crc_ok=crc.bool(), payload_bit_errors=errors, block_error=errors > 0,
                            total_payload_bits_per_ue=torch.full_like(errors, info.shape[-1]),
                            llr_sign=LLR_SIGN, decoder_mapping=LLR_MAPPING, export_scaling="none", export_clipping="none"))


def axes_metadata():
    axes = dict(n0=[], data_indices=["data_RE"], native_data_ordinals=["data_RE"],
                coded_bits=["batch", "UE", "codeword_bit"], info_bits=["batch", "UE", "TB_input_bit"])
    for name in ("z", "ep5_mean", "ep5_variance", "eta"):
        axes["features." + name] = ["batch", "data_RE", "UE"]
    axes["features.gram"] = ["batch", "data_RE", "UE", "UE"]
    for name in ("received_energy", "residual"):
        axes["features." + name] = ["batch", "data_RE"]
    for method in METHODS:
        axes[f"methods.{method}.llr"] = ["batch", "data_RE", "UE", "Qm_bit"]
        axes[f"methods.{method}.decoder_input_llr"] = ["batch", "UE", "stream_per_UE", "codeword_bit"]
        axes[f"methods.{method}.decoded_payload_bits"] = ["batch", "UE", "TB_input_bit"]
        for name in ("crc_ok", "payload_bit_errors", "block_error", "total_payload_bits_per_ue"):
            axes[f"methods.{method}.{name}"] = ["batch", "UE"]
    return axes


@torch.inference_mode()
def build_channel(link, models, ep, split, ebno, seed, identity_id, settings, check_codec=False):
    started = time.perf_counter()
    sample = link.transmit(1, ebno, seed)
    batch = make_detector_batch(link, sample, "practical")
    if check_codec:
        link.check_codec_identity(sample)
    native = link.receive(sample, "practical")
    batch_size, users, _, coded_length = native["llr"].shape
    data_count = len(batch["data_indices"])
    if batch_size != 1 or coded_length != 4 * data_count or set(models) != set(METHODS[2:]):
        raise ValueError("Expected one complete codeword per UE and all four checkpoints.")
    native_llr = native["llr"].reshape(batch_size, users, data_count, 4).permute(0, 2, 1, 3)
    methods = {"sionna_lmmse": tb_record(native_llr, native["llr"], native["decoded"], native["crc_ok"], sample["info"])}
    parts = {name: [] for name in METHODS[1:]}
    feature_parts = {name: [] for name in ("z", "gram", "received_energy", "eta", "ep5_mean", "ep5_variance", "residual")}
    chunks = 0
    for start in range(0, data_count, settings["re_chunk"]):
        ordinals = torch.arange(start, min(start + settings["re_chunk"], data_count))
        statistics = shared_statistics(batch, ordinals, settings["features"]["ce_epsilon"])
        matched = statistics["z"].reshape(-1, users)
        gram = statistics["gram"].reshape(-1, users, users)
        posterior = ep(matched, gram, return_iterations=(5,))
        parts["ep5"].append(posterior["llr"].reshape(1, -1, users, 4).cpu())
        mean = posterior["x_hat"].reshape(1, -1, users)
        variance = posterior["posterior_variance"].reshape(1, -1, users)
        statistics.update(ep5_mean=mean, ep5_variance=variance)
        statistics["residual"] = posterior_residual(
            statistics, mean, variance, batch["y"].shape[-1],
            settings["features"]["residual_atol"], settings["features"]["residual_rtol"])
        for name, model in models.items():
            output = model(matched, gram, return_iterations=(5,))["llr"]
            if output.shape != posterior["llr"].shape or not torch.isfinite(output).all():
                raise ValueError(f"Invalid full-codeword neural LLR chunk: {name}")
            parts[name].append(output.reshape(1, -1, users, 4).cpu())
        for name in feature_parts:
            feature_parts[name].append(statistics[name].cpu())
        chunks += 1
    for name, values in parts.items():
        llr = torch.cat(values, 1)
        decoder_input = llrs_to_native_order(llr.to(link.device))
        decoded, crc = link.codec.decode(decoder_input)
        methods[name] = tb_record(llr, decoder_input, decoded, crc, sample["info"])
    payload = cpu_tensors(dict(schema_version=SCHEMA_VERSION, status="complete", identity_id=identity_id,
        split=split, ebno_db=float(ebno), seed=int(seed), n0=batch["n0"],
        data_indices=batch["data_indices"], native_data_ordinals=torch.arange(data_count),
        coded_bits=batch["coded_bits"], info_bits=batch["bits"], methods=methods,
        features={name: torch.cat(values, 1) for name, values in feature_parts.items()}))
    payload["metadata"] = dict(tensor_shapes=tensor_schema(payload), tensor_axes=axes_metadata(),
        llr_sign=LLR_SIGN, llr_mapping=LLR_MAPPING,
        data_order="data_indices[n]=OFDM_symbol*FFT_size+subcarrier; coded_bits[b,u,n*4+q]",
        inference_inputs="y,Hhat,err_var,N0 only; no true channel, realized noise, or labels in feature computation",
        residual="(s-2*Re(mu^H*z)+Re(mu^H*G*mu)+sum diag(G)*nu)/M; EP5 posterior; no clamp",
        residual_tolerance=dict(absolute=settings["features"]["residual_atol"], relative=settings["features"]["residual_rtol"],
            scale="(s+2*abs(Re(mu^H*z))+abs(Re(mu^H*G*mu))+abs(sum diag(G)*nu))/M"),
        eta="sum_Rx err_var / (sum_Rx |Hhat|^2 + sum_Rx err_var + epsilon); candidate feature, not exact posterior",
        feature_settings=settings["features"], shared_frontend_chunks=chunks, single_transmit=True,
        csi="one cached practical estimate reused by native LMMSE and all chunk receivers",
        codec_identity_checked=check_codec, elapsed_seconds=time.perf_counter() - started)
    return validate_channel(payload, identity_id)


@torch.inference_mode()
def verify_cache_decode(path, codec, device, identity_id):
    payload = validate_channel(torch.load(path, map_location="cpu", weights_only=True), identity_id)
    for method, values in payload["methods"].items():
        bits, crc = codec.decode(values["decoder_input_llr"].to(device))
        if not torch.equal(bits.cpu(), values["decoded_payload_bits"]) or not torch.equal(crc.cpu(), values["crc_ok"]):
            raise ValueError(f"Reloaded raw LLR decode mismatch: {method}")
    return payload


def run(args):
    started = time.perf_counter()
    config = load_experiment_config(args.config, args.mode)
    paths = run_paths(config, args.run_id)
    checked = preflight(config)
    identity, identity_id = checked["identity"], checked["identity_id"]
    if args.preflight:
        print(json.dumps(dict(identity_id=identity_id, seed_plan=checked["seed_plan"],
                              paths={key: str(value) for key, value in paths.items()}), indent=2), flush=True)
        return 0
    root, manifest_path = paths["cache"], paths["cache"] / "manifest.json"
    if root.exists():
        if not args.reuse_cache:
            raise FileExistsError(f"Cache exists: {root}; use a new run-id or explicit --reuse-cache.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("identity_id") != identity_id or manifest.get("identity") != identity:
            raise ValueError("Existing cache identity differs; no files were reused or overwritten.")
        for record in manifest["channels"]:
            load_channel(root, record, identity_id)
        if manifest.get("status") == "complete":
            load_complete_cache(root)
            print(f"Reused complete cache {root}: no waveform generation or model execution; elapsed={time.perf_counter()-started:.2f}s", flush=True)
            return 0
    else:
        root.mkdir(parents=True, exist_ok=False)
        manifest = dict(schema_version=SCHEMA_VERSION, status="running", identity_id=identity_id, identity=identity,
                        seed_plan=checked["seed_plan"], run_id=args.run_id, channels=[],
                        tensor_axes=axes_metadata(), independent_channels={key: len(checked["seed_plan"][key])
                            for key in ("calibration", "development")},
                        planned_channel_ebno_cases=len(expected_cases(identity)),
                        limitations="Calibration/development only, not a final independent paper test set.")
        save_json(manifest_path, manifest)
    desired = [case for case in expected_cases(identity) if args.split == "all" or case[0] == args.split]
    existing = {(record["split"], record["ebno_db"], record["seed"]): record for record in manifest["channels"]}
    if len(existing) != len(manifest["channels"]) or not set(existing) <= set(expected_cases(identity)):
        raise ValueError("Existing cache has duplicate/unexpected channel records.")
    pending = [case for case in desired if case not in existing]
    if not pending:
        print(f"Requested split already complete: {args.split}; no generation or model execution", flush=True)
        return 0
    try:
        link = SionnaReferenceLink(identity["reference"]["config"])
        if (link.grid_metadata != identity["reference"]["grid"] or link.codec.metadata() != identity["reference"]["codec"]
                or link.covariance_metadata["sha256"] != identity["reference"]["covariance"]["sha256"]):
            raise ValueError("Actual native grid/codec/prior differs from the checked archive identity.")
        manifest["native_reference"] = link.metadata()
        model_config = copy.deepcopy(link.cfg)
        model_config["modulation"] = {"bits_per_symbol": link.codec.qm}
        from detectors.classical.ep import ExpectationPropagationDetector
        ep = ExpectationPropagationDetector(model_config, num_iterations=5, damping=.5)
        models = {}
        seeds = checked["seed_plan"]["calibration"] + checked["seed_plan"]["development"]
        for name, specification in config["checkpoints"].items():
            models[name], metadata = load_neural_checkpoint(specification["path"], specification["arch"], link, seeds)
            if any(metadata[key] != identity["checkpoints"][name][key] for key in ("sha256", "step", "distribution_id")):
                raise ValueError(f"Checkpoint identity changed after preflight: {name}")
        for number, (split, ebno, seed) in enumerate(pending, 1):
            channel_started = time.perf_counter()
            relative = channel_name(split, ebno, seed)
            path = root / relative
            if path.exists():
                raise FileExistsError(f"Unindexed channel file exists: {path}; select a new run-id; no partial file is silently trusted.")
            print(f"Cache {number}/{len(pending)} split={split} Eb/N0={ebno:g} seed={seed} start; elapsed={time.perf_counter()-started:.2f}s", flush=True)
            payload = build_channel(link, models, ep, split, ebno, seed, identity_id, config, check_codec=(number == 1))
            saved = atomic_torch_save(path, payload,
                verify=lambda temporary: verify_cache_decode(temporary, link.codec, link.device, identity_id))
            record = dict(split=split, ebno_db=ebno, seed=seed, status="complete", file=relative, **saved,
                          n0=payload["n0"].item(), elapsed_seconds=time.perf_counter()-channel_started,
                          raw_decode_reload_verified=True, shared_frontend_chunks=payload["metadata"]["shared_frontend_chunks"])
            manifest["channels"].append(record)
            manifest["status"] = "complete" if len(manifest["channels"]) == len(expected_cases(identity)) else "running"
            manifest["storage_bytes"] = sum(item["bytes"] for item in manifest["channels"])
            manifest["elapsed_seconds"] = time.perf_counter() - started
            manifest["tensor_shapes"] = payload["metadata"]["tensor_shapes"]
            save_json(manifest_path, manifest)
            print(f"Cache {number}/{len(pending)} complete: {saved['bytes']/2**20:.2f} MiB, "
                  f"channel_elapsed={record['elapsed_seconds']:.2f}s total_elapsed={manifest['elapsed_seconds']:.2f}s", flush=True)
        if manifest["status"] == "complete":
            load_complete_cache(root)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter()-started)
        save_json(manifest_path, manifest)
        raise
    print(f"Cache {manifest['status']}: {manifest_path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), default="development")
    parser.add_argument("--split", choices=("all", "calibration", "development"), default="all")
    parser.add_argument("--preflight", action="store_true", help="Check server artifacts, runtime and disjoint seeds without generating samples.")
    parser.add_argument("--reuse-cache", action="store_true", help="Explicitly reuse only complete channels with exactly matching identities.")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
