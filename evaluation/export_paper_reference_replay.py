"""Export complete, paired Mode-B replay cases without changing the frozen pipeline.

Run from the repository root with ``python -m evaluation.export_paper_reference_replay``.
Only a NEW output directory is written. Existing results and the A1 bundle are read-only.
"""

import argparse
import copy
import csv
import inspect
import json
import math
import os
from pathlib import Path
import sys

import torch

from evaluation.compare_paper_reference_a1 import evaluate_neural, selected_bits, selected_native_llrs
from evaluation.evaluate_sionna_bler import block_statistics, provenance, save_json
from link_level.detector_adapter import ClassicalDetectorAdapter, llrs_to_native_order, make_detector_batch
from link_level.partner_a1_adapter import (
    PartnerA1Adapter, a1_inputs, file_sha256, load_neural_checkpoint, verify_bundle,
)
from link_level.sionna_ce import distribution_id
from link_level.sionna_reference import SionnaReferenceLink


DEFAULT_CASES = ((-11., 20260957), (-10.5, 20260966), (-10., 20260989),
                 (-11., 20260959), (-10.5, 20260991), (-10., 20260947),
                 (-11., 20260986), (-10.5, 20260933))
METHODS = ("sionna_lmmse", "custom_lmmse", "ep5", "gt_ep", "detr_ep",
           "a1_ce_k64", "a1_ce_k256", "a1_ce_native_lmmse")
UNKNOWN_CRC = "not directly established by exported runtime metadata"
COMPARATOR = Path(__file__).with_name("compare_paper_reference_a1.py")
COMPARATOR_SHA256 = "3ebfc4382c0611d0bd15ee8efe1682293da5353689e1459f0981267364c01092"
LLR_CONVENTION = "log P(bit=1)/P(bit=0); hard bit = llr > 0"
BIT_MAPPING = (
    "data_indices[n] = t*fft_size+f, in native ResourceGrid data order; "
    "coded_bits[b,u,n*Qm+q] corresponds to QAM data symbol [b,u,0,n], label bit q; "
    "evaluator_llr[b,n,u,q] -> permute(0,2,1,3).reshape(B,UE,1,Ndata*Qm); "
    "codec.decode squeezes the singleton stream axis before TBDecoder. "
    "No clipping, scaling or bit permutation is added by this exporter."
)


def cpu_tree(value):
    """Own compact storage without changing shape, dtype, order or values."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_tree(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Replay metadata must be primitive, got {type(value)}")


def tensor_schema(tree, axes, prefix=""):
    result = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(tensor_schema(value, axes, name))
        elif isinstance(value, torch.Tensor):
            if name not in axes or len(axes[name]) != value.ndim:
                raise ValueError(f"Missing/wrong tensor axes: {name}")
            if value.device.type != "cpu":
                raise ValueError(f"Non-CPU replay tensor: {name}")
            result[name] = dict(shape=list(value.shape), axes=axes[name], dtype=str(value.dtype),
                                logical_bytes=value.numel() * value.element_size())
    return result


def capture_inputs(sample, batch):
    """Select all data REs once; keep RX and per-UE stream singleton axes explicit."""
    b, t, f, m = batch["y"].shape
    k = batch["h_hat"].shape[-1]
    indices = batch["data_indices"]
    ordinals = torch.arange(indices.numel(), device=indices.device)

    def y_data(grid):
        return grid.reshape(b, t * f, m)[:, indices].unsqueeze(1)

    def h_data(grid):
        return grid.reshape(b, t * f, m, k)[:, indices].unsqueeze(1)

    clean = sample["y_clean"][:, 0].permute(0, 2, 3, 1)
    noise = (sample["y"] - sample["y_clean"])[:, 0].permute(0, 2, 3, 1)
    values = dict(n0=batch["n0"], data_indices=indices, native_data_ordinals=ordinals,
                  data_coordinates=torch.stack((indices // f, indices % f), -1),
                  y=y_data(batch["y"]), h_hat=h_data(batch["h_hat"]),
                  err_var=h_data(batch["err_var"]), h_true=h_data(batch["h_true"]),
                  Ruu=batch["Ruu"], y_clean=y_data(clean), realized_noise=y_data(noise),
                  qam_symbols=sample["x"].flatten(-2)[..., indices],
                  coded_bits=batch["coded_bits"], info_bits=batch["bits"],
                  a1_inputs=a1_inputs(batch, ordinals, "sionna_diagonal"))
    axes = dict(n0=[], data_indices=["data_RE"], native_data_ordinals=["data_RE"],
                data_coordinates=["data_RE", "coordinate(t,f)"],
                Ruu=["batch", "Rx_antenna", "Rx_antenna"],
                qam_symbols=["batch", "UE", "stream_per_UE", "data_RE"],
                coded_bits=["batch", "UE", "codeword_bit"], info_bits=["batch", "UE", "TB_input_bit"],
                **{"a1_inputs.ruu": ["batch", "Rx_antenna", "Rx_antenna"]})
    for name in ("y", "y_clean", "realized_noise", "a1_inputs.y"):
        axes[name] = ["batch", "receiver", "data_RE", "Rx_antenna"]
    for name in ("h_hat", "err_var", "h_true", "a1_inputs.h"):
        axes[name] = ["batch", "receiver", "data_RE", "Rx_antenna", "UE"]
    return cpu_tree(values), {"inputs." + key: value for key, value in axes.items()}


def method_record(name, evaluator_llr, decoder_input, decoded, crc, info, natural_llr=None):
    if not torch.equal(llrs_to_native_order(evaluator_llr), decoder_input.cpu()):
        raise ValueError(f"LLR ordering/value mismatch: {name}")
    errors = (decoded.cpu() != info.cpu()).sum(-1)
    native_axes = ["batch", "UE", "stream_per_UE", "codeword_bit"]
    evaluator_axes = ["batch", "data_RE", "UE", "Qm_bit"]
    natural_is_native = natural_llr is not None
    record = dict(detector_llr=natural_llr if natural_is_native else evaluator_llr,
                  evaluator_llr=evaluator_llr, decoder_input_llr=decoder_input,
                  decoded_payload_bits=decoded, crc_ok=crc.bool(), payload_bit_errors=errors,
                  block_error=errors > 0, total_payload_bits_per_ue=torch.full_like(errors, info.shape[-1]),
                  llr_convention=LLR_CONVENTION, mapping=BIT_MAPPING,
                  exporter_clipping="none", exporter_scaling="none",
                  detector_representation="existing ClassicalDetectorAdapter [B,UE,1,G]" if natural_is_native else "existing evaluator",
                  detector_internal_processing="existing APP/logsumexp readout; no final LLR clip or rescale",
                  decoder_internal_processing="unchanged; actual llr_max and decoder attributes in manifest.codec.runtime")
    axes = dict(detector_llr=native_axes if natural_is_native else evaluator_axes,
                evaluator_llr=evaluator_axes, decoder_input_llr=native_axes,
                decoded_payload_bits=["batch", "UE", "TB_input_bit"])
    for key in ("crc_ok", "payload_bit_errors", "block_error", "total_payload_bits_per_ue"):
        axes[key] = ["batch", "UE"]
    if name.startswith("a1_ce_"):
        # PartnerA1Adapter removes only the receiver axis from the original package output.
        record["package_llr"] = evaluator_llr.unsqueeze(1)
        axes["package_llr"] = ["batch", "receiver", "data_RE", "UE", "Qm_bit"]
        record["package_to_evaluator"] = "package_llr[:,0]; receiver count is one"
        if name in ("a1_ce_k64", "a1_ce_k256"):
            record["detector_internal_processing"] = "original A1 RB readout: probability floor 1e-20, internal LLR clamp [-20,20]; no export changes"
    return cpu_tree(record), {f"methods.{name}.{key}": value for key, value in axes.items()}


@torch.inference_mode()
def verify_serialized_decodes(path, codec, device):
    replay = torch.load(path, map_location="cpu", weights_only=True)
    for name, values in replay["methods"].items():
        bits, crc = codec.decode(values["decoder_input_llr"].to(device))
        if not torch.equal(bits.cpu(), values["decoded_payload_bits"]) or not torch.equal(crc.cpu(), values["crc_ok"]):
            raise ValueError(f"Serialized LLR decode mismatch: {name}")
    return replay


def assert_tensor_tree_equal(first, second, prefix="inputs"):
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError(f"Determinism keys differ at {prefix}")
        for key in first:
            assert_tensor_tree_equal(first[key], second[key], prefix + "." + key)
    elif isinstance(first, torch.Tensor):
        if first.dtype != second.dtype or first.shape != second.shape or not torch.equal(first, second):
            raise ValueError(f"Non-deterministic regeneration: {prefix}")
    elif first != second:
        raise ValueError(f"Non-deterministic metadata: {prefix}")


def json_attribute(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite runtime attribute")
        return value
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() > 32768 or value.is_complex():
            raise ValueError("large/complex attribute omitted")
        return json_attribute(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return [json_attribute(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_attribute(item) for key, item in value.items()}
    if callable(value):
        return dict(callable_identity=f"{value.__module__}.{value.__qualname__}")
    if hasattr(value, "tolist"):
        if getattr(value, "size", 0) > 32768:
            raise ValueError("large array omitted")
        return json_attribute(value.tolist())
    raise TypeError(type(value).__name__)


def runtime_record(obj, fields):
    if obj is None:
        return dict(available=False)
    result = dict(available=True, class_identity=f"{type(obj).__module__}.{type(obj).__qualname__}",
                  attributes={}, unavailable={})
    try:
        source = inspect.getsourcefile(type(obj))
    except TypeError:
        source = None
    if source and Path(source).is_file():
        result.update(source_path=source, source_sha256=file_sha256(source))
    for name in fields.split():
        try:
            result["attributes"][name] = json_attribute(getattr(obj, name))
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            result["unavailable"][name] = f"{type(exc).__name__}: {exc}"
    return result


def codec_manifest(codec):
    enc, dec = codec.encoder, codec.decoder
    encoder_fields = ("k n tb_size k_padding num_cbs coderate num_tx cw_lengths cw_lengths_sum cw_lengths_max "
                      "_target_tb_size _target_coderate _num_coded_bits _num_bits_per_symbol _num_layers "
                      "_n_rnti _n_id _channel_type _codeword_index _use_scrambler _tb_crc_length _cb_crc_length _cb_size "
                      "_cw_lengths_min _output_perm _output_perm_inv precision device")
    records = dict(encoder=runtime_record(enc, encoder_fields),
                   decoder=runtime_record(dec, "k n tb_size _num_cbs _output_perm_inv precision device"))
    for name in ("tb_crc_encoder", "cb_crc_encoder"):
        records[name] = runtime_record(getattr(enc, name, None), "crc_degree crc_length crc_pol k n")
    records["scrambler"] = runtime_record(getattr(enc, "scrambler", None),
                                         "n_rnti n_id _n_rnti _n_id _channel_type _codeword_index _c_init _binary")
    records["ldpc_encoder"] = runtime_record(getattr(enc, "ldpc_encoder", None),
        "k n k_ldpc n_ldpc num_bits_per_symbol out_int out_int_inv _bg _z _k_b _i_ls _coderate")
    records["ldpc_decoder"] = runtime_record(getattr(dec, "_decoder", None),
        "num_iter llr_max _num_iter _llr_max _cn_update _vn_update _hard_out _return_infobits _prune_pcm precision device")
    for name in ("_descrambler", "_tb_crc_decoder", "_cb_crc_decoder"):
        records[name] = runtime_record(getattr(dec, name, None), "crc_degree crc_length _binary")
    return dict(summary=codec.metadata(), runtime=records, info_bits_transport_block_crc_membership=UNKNOWN_CRC,
                attribute_evidence="Values read from these actual instantiated objects; absent fields are not inferred.",
                bit_mapping=BIT_MAPPING, llr_convention=LLR_CONVENTION,
                decoder_input="[B,UE,1,G]; grid_llrs_to_codewords -> squeeze(2).contiguous() -> TBDecoder",
                payload_error_definition="Compare decoder output to original TBEncoder input, in original order.")


def parse_case(value):
    try:
        ebno, seed = value.split(":")
        case = float(ebno), int(seed)
        if not math.isfinite(case[0]) or not 0 <= case[1] < 2 ** 31:
            raise ValueError()
        return case
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use --case=-11:20260957 (finite Eb/N0 and nonnegative seed).") from exc


def select_source_cases(source, cases):
    if (source.get("status") != "complete" or source.get("coded_bler_evaluated") is not True
            or source.get("selected_data_re") != 2304 or "sionna_diagonal" not in source.get("policies", [])
            or source.get("classical_and_gt_detr_ce_policy") != "sionna_diagonal"):
        raise ValueError("Source must be the completed full-codeword practical-CSI comparison.")
    if len(set(cases)) != len(cases):
        raise ValueError("Duplicate requested replay cases.")
    selected = []
    for ebno, seed in cases:
        matches = [row for point in source["points"] if point["ebno_db"] == ebno and point["status"] == "complete"
                   for row in point["channels"] if row["seed"] == seed]
        if len(matches) != 1:
            raise ValueError(f"Expected one source row for Eb/N0={ebno}, seed={seed}.")
        row = matches[0]
        if row["data_ordinals"] != list(range(2304)) or row["sampling_seed"] != seed + 3000007:
            raise ValueError("Source RE ordering/sampling seed does not match the frozen evaluator.")
        for method in METHODS:
            if "coded" not in row["receivers"].get(method, {}):
                raise ValueError(f"Source lacks coded results for {method}.")
        selected.append(row)
    return selected


def source_mismatches(source_row, receivers):
    differences = []
    for method in METHODS:
        expected, actual = source_row["receivers"][method], receivers[method]
        for metric, value in expected["coded"].items():
            if metric in ("bler", "ber"):
                continue  # Derived ratios; integer numerators and denominators are checked.
            if actual["coded"].get(metric) != value:
                differences.append(f"{method}.coded.{metric}: expected {value}, got {actual['coded'].get(metric)}")
        for metric in ("bits", "bit_errors"):
            if expected["soft"][metric] != actual["soft"][metric]:
                differences.append(f"{method}.soft.{metric}: expected {expected['soft'][metric]}, got {actual['soft'][metric]}")
    return differences


def diagnostics_selection(cases, rows, mode):
    if mode == "all":
        return set(cases)
    chosen, roles = set(), set()
    if mode == "representative":
        for case, row in zip(cases, rows):
            gt = row["receivers"]["gt_ep"]["coded"]["block_errors"]
            a1 = row["receivers"]["a1_ce_k256"]["coded"]["block_errors"]
            role = "rescue" if gt > a1 else "harm" if gt < a1 else "tie"
            if role in ("rescue", "harm") and role not in roles:
                chosen.add(case)
                roles.add(role)
    return chosen


@torch.inference_mode()
def a1_population_diagnostics(partner, values, sampling_seed, predictions):
    """Call original read-only APIs. No sampler/readout mathematics are copied."""
    runtime = partner.runtime
    if not all(callable(getattr(runtime, key, None)) for key in ("detector_for", "sample_population")):
        return None, "Original runtime does not expose detector_for/sample_population."
    if set(values) != {"y", "h", "ruu"}:
        raise ValueError("A1 inference accepts only y/h/ruu.")
    values = {key: value.to(partner.device) for key, value in values.items()}
    detector = runtime.detector_for(partner.model, values["y"], values["h"], sampling_seed)
    population = runtime.sample_population(detector, detector.prepare(**values), prefixes=(64,))
    required = {"candidate_tokens", "physical_energy", "path_log_q", "rao_blackwell_probability",
                "rao_blackwell_probability_prefixes"}
    if not required <= population.keys() or 64 not in population["rao_blackwell_probability_prefixes"]:
        return None, "Original population API does not expose all requested fields/prefix64."
    shape = tuple(values["y"].shape[:3])
    llr256 = detector.readout(population, shape=shape, device=partner.device).cpu()
    prefix = population["rao_blackwell_probability_prefixes"][64]
    llr64 = detector.readout(dict(rao_blackwell_probability=prefix), shape=shape, device=partner.device).cpu()
    for key, value in (("a1_ce_k256", llr256), ("a1_ce_k64", llr64)):
        if not torch.equal(value[:, 0], predictions[key]):
            raise ValueError(f"Repeated original A1 diagnostic readout differs: {key}")
    tokens = population["candidate_tokens"]
    points = partner.model.table.points.detach().cpu()
    result = dict(candidate_tokens=tokens, candidate_symbols=points[tokens.cpu().long()],
                  candidate_energies=population["physical_energy"], log_q=population["path_log_q"],
                  rb_probabilities=population["rao_blackwell_probability"], k64_prefix_rb_probabilities=prefix,
                  k64_prefix_llr=llr64, k256_llr=llr256, constellation_points=points,
                  constellation_bit_labels=partner.model.table.bit_table,
                  sampling_seed=sampling_seed, exact_readout_matches_evaluation=True,
                  flattened_RE_order="(batch, receiver, native_data_ordinal), contiguous; batch=receiver=1",
                  k64_semantics="First 64 members of this K256 population; original prefix RB accumulator/readout.",
                  omitted_population_fields=sorted(set(population) - required))
    axes = dict(candidate_tokens=["flattened_RE", "candidate_256", "UE"],
                candidate_symbols=["flattened_RE", "candidate_256", "UE"],
                candidate_energies=["flattened_RE", "candidate_256"], log_q=["flattened_RE", "candidate_256"],
                rb_probabilities=["flattened_RE", "UE", "constellation_symbol"],
                k64_prefix_rb_probabilities=["flattened_RE", "UE", "constellation_symbol"],
                k64_prefix_llr=["batch", "receiver", "data_RE", "UE", "Qm_bit"],
                k256_llr=["batch", "receiver", "data_RE", "UE", "Qm_bit"],
                constellation_points=["constellation_symbol"],
                constellation_bit_labels=["constellation_symbol", "Qm_bit"])
    result = cpu_tree(result)
    result["tensor_shapes_and_axes"] = tensor_schema(result, axes)
    return result, None


def save_tensor_file(path, payload, root):
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)
    return dict(path=path.relative_to(root).as_posix(), sha256=file_sha256(path), bytes=path.stat().st_size)


def disk_usage(root, manifest):
    case_sizes = [item["file"]["bytes"] for item in manifest["cases"]]
    diag_sizes = [item["diagnostics"]["file"]["bytes"] for item in manifest["cases"]
                  if "file" in item.get("diagnostics", {})]
    mean_case = sum(case_sizes) / len(case_sizes) if case_sizes else None
    mean_diag = sum(diag_sizes) / len(diag_sizes) if diag_sizes else None
    measured = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return dict(measured_directory_bytes=measured, completed_cases=len(case_sizes),
                mean_case_pt_bytes=mean_case, mean_diagnostic_pt_bytes=mean_diag,
                estimated_eight_cases_pt_bytes=8 * mean_case if mean_case is not None else None,
                estimated_eight_cases_plus_two_diagnostics_pt_bytes=(8 * mean_case + 2 * mean_diag)
                if mean_case is not None and mean_diag is not None else None,
                basis="Actual torch.save file sizes; full estimate is 8 cases + 2 diagnostic populations. "
                      "JSON/CSV/README overhead excluded from projection; missing diagnostics are not guessed.")


def write_index(root, manifest):
    with (root / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ebno_db", "seed", "a1_sampling_seed", "n0", "method", "payload_bit_errors",
                         "TB_errors", "crc_failures", "source_counts_match", "path", "sha256"))
        for case in manifest["cases"]:
            for method, values in case["receivers"].items():
                counts = values["coded"]
                writer.writerow((case["ebno_db"], case["seed"], case["a1_sampling_seed"], case["n0"], method,
                                 counts["bit_errors"], counts["block_errors"], counts["crc_failures"],
                                 not case["source_mismatches"], case["file"]["path"], case["file"]["sha256"]))
    manifest["storage"] = disk_usage(root, manifest)
    save_json(root / "manifest.json", manifest)


def verify_protected_files(hashes, a1_root, bundle_identity):
    for path, expected in hashes.items():
        if file_sha256(path) != expected:
            raise ValueError(f"Protected input changed: {path}")
    actual = verify_bundle(a1_root)
    for field in ("manifest_sha256", "model"):
        if actual[field] != bundle_identity[field]:
            raise ValueError(f"External A1 bundle changed: {field}")


@torch.inference_mode()
def export_case(link, classical, partner, models, case, source_row, chunk_size):
    ebno, seed = case
    sample = link.transmit(1, ebno, seed)
    batch = make_detector_batch(link, sample, "practical")
    inputs, axes = capture_inputs(sample, batch)
    ordinals = inputs["native_data_ordinals"]
    classical_outputs = classical.evaluate(sample, batch)
    predictions = {name: selected_native_llrs(out["llr"], ordinals) for name, out in classical_outputs.items()}
    neural, neural_timing = evaluate_neural(models, batch, ordinals, chunk_size)
    predictions.update(neural)
    partner_outputs, a1_timing = partner.evaluate(batch, ordinals, "sionna_diagonal", source_row["sampling_seed"])
    predictions.update({"a1_ce_k64": partner_outputs["raw_k64"], "a1_ce_k256": partner_outputs["raw_k256"],
                        "a1_ce_native_lmmse": partner_outputs["lmmse"]})
    methods, receivers = {}, {}
    bits = selected_bits(batch, ordinals)
    for name in METHODS:
        llr = predictions[name]
        if name in classical_outputs:
            received = classical_outputs[name]
            decoder_input = received["llr"]
        else:
            decoder_input = llrs_to_native_order(llr.to(link.device))
            decoded, crc = link.codec.decode(decoder_input)
            received = dict(decoded=decoded, crc_ok=crc)
        methods[name], method_axes = method_record(
            name, llr, decoder_input, received["decoded"], received["crc_ok"], sample["info"],
            received.get("llr"))
        axes.update(method_axes)
        receivers[name] = dict(soft=partner.metrics.soft_metrics(llr, bits), coded=block_statistics(sample, received))
    replay = dict(schema_version=1, ebno_db=ebno, seed=seed, a1_sampling_seed=source_row["sampling_seed"],
                  inputs=inputs, methods=methods,
                  metadata=dict(bit_mapping=BIT_MAPPING, llr_convention=LLR_CONVENTION,
                                source_comparison_row=source_row, timing=dict(neural=neural_timing, a1=a1_timing)))
    replay["metadata"]["tensor_shapes_and_axes"] = tensor_schema(replay, axes)
    # Release the first full grid before regenerating: only selected data tensors are kept on CPU.
    del sample, batch, classical_outputs, received, decoder_input, partner_outputs, neural
    repeated_sample = link.transmit(1, ebno, seed)
    repeated_batch = make_detector_batch(link, repeated_sample, "practical")
    repeated, _ = capture_inputs(repeated_sample, repeated_batch)
    assert_tensor_tree_equal(inputs, repeated)
    replay["metadata"]["input_regeneration_bitwise_equal"] = True
    return replay, receivers, predictions


README = """Mode-B practical-CSI deterministic replay

Load: case = torch.load('cases/<name>.pt', map_location='cpu', weights_only=True)
All tensor leaves are CPU tensors, with original dtypes. Axes/shapes/bytes are in
case['metadata']['tensor_shapes_and_axes']. No precision reduction is performed.
Inputs y/h_hat/err_var/h_true have explicit singleton receiver axes. qam_symbols
keeps the singleton stream-per-UE axis. A1 inputs are ONLY inputs['a1_inputs']:
y and h are whitened by sqrt(N0 + sum_UE err_var), ruu is identity. Original Ruu
is N0*I. Truth/labels are diagnostic exports and never enter A1 inference.

Each methods[name] includes evaluator_llr and decoder_input_llr, per-UE error
counts, block-error booleans, CRC outcomes and decoded TB input bits. LLRs are
log P1/P0; hard bits are llr>0. decoder_input_llr has axes [B,UE,1,G]. The codec
squeezes only axis 2 before TBDecoder. No external clipping/scaling is applied.
Use the original NRTransportBlockCodec with manifest codec.summary fields and
Ndata=G/Qm, UE=16, on the recorded Sionna version; codec.decode(decoder_input_llr)
must reproduce decoded_payload_bits and crc_ok. Every exported file is reloaded
and decoded in the exporting environment to verify this property.

manifest.json records actual runtime codec/CRC/LDPC attributes, versions,
config, model/prior hashes and SHA256 for EVERY generated .pt. Some codec fields
are private runtime attributes and may be absent on other Sionna versions.
No claim about CRC membership in info_bits is inferred from names or constants.

Each physical input is regenerated twice and compared bitwise. All integer
error counts (including per-UE TB errors) are compared to the source JSON;
disagreements fail the export and leave a manifest with status=failed. Historical
raw tensors were not stored in that JSON: count agreement does not establish
bitwise identity with the historical run. Current/original environments are
recorded separately. Accept only a manifest with status=complete.

Optional A1 populations use original detector_for/prepare/sample_population/
readout APIs. Repeated K256 and exact-prefix K64 readouts must match exported
LLRs bitwise. Candidate symbols use the original constellation lookup; original
tokens/energies/log q/RB probabilities are also retained. Large unrequested
conditional per-candidate moments are omitted explicitly, not downcast.

storage in manifest.json projects 8 case files plus 2 diagnostic files from
actual saved sizes. Results are local-only and ignored by Git. No model training,
upload, weight copying, checkpoint selection or source-result modification occurs.
"""


def run(args):
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise FileExistsError("Replay output directory already exists; choose a new --output-dir.")
    if root.is_relative_to(Path(args.a1_root).resolve()):
        raise ValueError("Replay output must not be inside the read-only A1 bundle.")
    if file_sha256(COMPARATOR) != COMPARATOR_SHA256:
        raise ValueError("Frozen comparator identity changed; review before exporting.")
    source_path = Path(args.source_comparison)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cases = args.case or list(DEFAULT_CASES)
    source_rows = select_source_cases(source, cases)
    cfg = copy.deepcopy(source["reference"]["config"])
    if cfg["mode"] != "paper_reference" or cfg["channel_estimation"]["mode"] != "practical":
        raise ValueError("Source is not Mode-B practical CSI.")
    cfg["general"]["device"] = args.device
    cfg["channel_estimation"]["covariance_path"] = args.ce_covariance
    hashes = {str(COMPARATOR): file_sha256(COMPARATOR), str(source_path): file_sha256(source_path),
              args.ce_covariance: file_sha256(args.ce_covariance)}
    if hashes[args.ce_covariance] != source["reference"]["covariance"]["sha256"]:
        raise ValueError("CE prior SHA256 differs from source comparison.")
    for name, path in (("gt_ep", args.gt_checkpoint), ("detr_ep", args.detr_checkpoint)):
        hashes[path] = file_sha256(path)
        if hashes[path] != source["neural_checkpoints"][name]["sha256"]:
            raise ValueError(f"{name} checkpoint SHA256 differs from source comparison.")
    bundle_identity = verify_bundle(args.a1_root)
    for field in ("manifest_sha256", "model"):
        if bundle_identity[field] != source["a1"][field]:
            raise ValueError(f"A1 {field} differs from source comparison.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    partner = PartnerA1Adapter(args.a1_root, args.device)
    link = SionnaReferenceLink(cfg)
    if (link.codec.qm, link.codec.coded_bits, link.codec.info_bits) != (4, 9216, 3104):
        raise ValueError("Frozen Qm/G/TB contract changed.")
    partner.check_constellation(link)
    reference = link.metadata()
    if distribution_id(cfg) != source["distribution_id"]:
        raise ValueError("Source distribution identity differs.")
    for field in ("grid", "codec"):
        if reference[field] != source["reference"][field]:
            raise ValueError(f"Source {field} differs from regenerated configuration.")
    models, checkpoints = {}, {}
    for name, path in (("gt_ep", args.gt_checkpoint), ("detr_ep", args.detr_checkpoint)):
        models[name], checkpoints[name] = load_neural_checkpoint(path, name, link, [seed for _, seed in cases])
    chunk_size = int(source["provenance"]["arguments"]["re_chunk"])
    classical = ClassicalDetectorAdapter(link, custom_ce_policy="sionna_diagonal", re_chunk=chunk_size)
    diagnostics_cases = diagnostics_selection(cases, source_rows, args.diagnostics)
    manifest = dict(schema_version=1, status="running", purpose="selected full-codeword replay export",
                    source_comparison=dict(path=str(source_path.resolve()), sha256=hashes[str(source_path)],
                                           provenance=source["provenance"]),
                    provenance=provenance(args), protected_inputs_sha256=hashes,
                    exporter=dict(path=__file__, sha256=file_sha256(__file__)),
                    a1=partner.metadata, neural_checkpoints=checkpoints,
                    covariance=dict(path=str(Path(args.ce_covariance).resolve()), sha256=hashes[args.ce_covariance]),
                    distribution_id=distribution_id(cfg), reference=reference, codec=codec_manifest(link.codec),
                    requested_cases=cases, re_chunk=chunk_size, diagnostics_mode=args.diagnostics,
                    ce_policy="sionna_diagonal", Ruu_semantics="original Ruu=N0*I; A1 ruu=I after diagonal CE whitening",
                    a1_input_semantics="a1_inputs.y=y/sqrt(n0+sum_UE err_var); a1_inputs.h=Hhat/sqrt(...)",
                    n0_semantics="Existing link.noise_variance uses ebnodb2no(Eb/N0,Qm,k/G,actual ResourceGrid); "
                                 "includes the frozen pilot/data/CP accounting; numerical n0 is saved per case.",
                    cases=[], scientific_limitations=[
                        "Historical JSON has error counts, not raw tensors; this verifies counts and new repeatability.",
                        "A1 training distribution and evaluation-seed independence remain unverified."])
    root.mkdir(parents=True, exist_ok=False)
    (root / "cases").mkdir()
    (root / "diagnostics").mkdir()
    (root / "README.txt").write_text(README, encoding="utf-8")
    write_index(root, manifest)
    try:
        for case, source_row in zip(cases, source_rows):
            ebno, seed = case
            label = f"ebno_{ebno:g}_seed_{seed}".replace("-", "m").replace(".", "p")
            print(f"Exporting {label}: all data REs, all eight receivers", flush=True)
            replay, receivers, predictions = export_case(link, classical, partner, models, case, source_row, chunk_size)
            differences = source_mismatches(source_row, receivers)
            replay["metadata"].update(source_comparison_path=str(source_path.resolve()),
                                      source_comparison_sha256=hashes[str(source_path)],
                                      distribution_id=manifest["distribution_id"], source_mismatches=differences)
            path = root / "cases" / (label + ".pt")
            record = dict(ebno_db=ebno, seed=seed, a1_sampling_seed=source_row["sampling_seed"],
                          n0=replay["inputs"]["n0"].item(), receivers=receivers, source_mismatches=differences,
                          file=save_tensor_file(path, replay, root),
                          input_regeneration_bitwise_equal=True, serialized_decode_verified=False)
            manifest["cases"].append(record)
            verify_serialized_decodes(path, link.codec, link.device)
            record["serialized_decode_verified"] = True
            if differences:
                raise ValueError("Replay differs from source comparison: " + "; ".join(differences))
            if case in diagnostics_cases:
                population, reason = a1_population_diagnostics(partner, replay["inputs"]["a1_inputs"],
                                                               source_row["sampling_seed"], predictions)
                if population is None:
                    record["diagnostics"] = dict(status="unavailable", reason=reason)
                else:
                    population.update(ebno_db=ebno, seed=seed, replay_file=record["file"])
                    record["diagnostics"] = dict(status="saved", file=save_tensor_file(
                        root / "diagnostics" / (label + "_a1_population.pt"), population, root))
            else:
                record["diagnostics"] = dict(status="not_selected")
            manifest["codec"] = codec_manifest(link.codec)  # Include lazy runtime attributes after decoding.
            write_index(root, manifest)
            print(f"Verified {label}; case file {record['file']['bytes'] / 2**20:.2f} MiB", flush=True)
            del replay, predictions
        verify_protected_files(hashes, args.a1_root, bundle_identity)
        manifest.update(status="complete", protected_inputs_unchanged=True)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            verify_protected_files(hashes, args.a1_root, bundle_identity)
        except Exception as exc:
            manifest.update(status="failed", protected_input_error=str(exc))
            raise
        finally:
            write_index(root, manifest)
    print(f"Saved {root / 'manifest.json'}", flush=True)
    print(json.dumps(manifest["storage"], indent=2), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-comparison", default="results/paper_reference/a1_gt_detr_coded_100_v1.json")
    parser.add_argument("--output-dir", default="results/paper_reference/replay_a1_gt_v1")
    parser.add_argument("--case", action="append", type=parse_case, help="Repeat --case=EBNO:SEED; default: all eight requested cases.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--a1-root", default="../A1_PARTNER_MINIMAL")
    parser.add_argument("--ce-covariance", default="data/cache/paper_reference_ft_cov.pt")
    parser.add_argument("--gt-checkpoint", default="ckp/paper_reference_gt_ep_smoke300/best.pth")
    parser.add_argument("--detr-checkpoint", default="ckp/paper_reference_detr_ep_smoke300/best.pth")
    parser.add_argument("--diagnostics", choices=("representative", "all", "none"), default="representative")
    args = parser.parse_args(argv)
    # Also cover lazy external imports during inference, not only load_runtime().
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return run(args)
    finally:
        sys.dont_write_bytecode = previous


if __name__ == "__main__":
    raise SystemExit(main())
