"""Small paired, full-codeword pilot for frozen full_mlp and R3P5 reference6000."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from evaluation.evaluate_llr_mechanism_controls import (
    evaluate_observation, link_and_ep, load_receivers, stream_observation,
)
from evaluation.evaluate_paper_soft_output import (
    coded_labels, llrs_to_native_order, payload_statistics, soft_statistics,
)
from evaluation.evaluate_sionna_bler import save_json
from evaluation.llr_mechanism_common import load_config, prepare
from link_level.detector_adapter import make_detector_batch
from models.r3p5_reference6000_adapter import R3P5Reference6000Adapter


DEFAULT_TRAINING = "ckp/llr_mechanism_controls/mechanism_20261008_confirm1"
DEFAULT_OUTPUT = "results/r3p5_reference6000_pilot/validation8_v1"
METHODS = ("native_lmmse", "ep5_raw", "noise_per_bit_train24",
           "full_mlp_seed42_best", "full_mlp_seed43_best", "full_mlp_seed44_best",
           "r3p5_reference6000_native_k64_direct_rb")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_data(value: torch.Tensor, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    b = value.shape[0]
    flat = value.reshape(b, -1, *value.shape[3:])
    selected = flat[:, indices.to(flat.device)]
    return selected.unsqueeze(1).to(device).contiguous()


def add_pilot_receiver(channel, link, adapter, seed):
    batch = make_detector_batch(link, channel["_sample"], "practical")
    indices = batch["data_indices"]
    y = selected_data(batch["y"], indices, adapter.device)
    h = selected_data(batch["h_hat"], indices, adapter.device)
    error = selected_data(batch["err_var"], indices, adapter.device)
    if y.shape != (1, 1, 2304, 256) or h.shape != (1, 1, 2304, 256, 16):
        raise ValueError(f"Mode-B external input mismatch: y={tuple(y.shape)} H={tuple(h.shape)}")

    llr, audit = adapter.predict(y, h, error, batch["n0"], seed=seed + 3_000_007)
    llr = llr[:, 0].cpu()  # [B, complete data RE, UE stream, Qm]
    labels = coded_labels(channel)
    if llr.shape != labels.shape or not bool(torch.isfinite(llr).all()):
        raise ValueError("External model did not return complete finite codeword LLRs")
    if tuple(llrs_to_native_order(llr).shape) != tuple(channel["methods"]["ep5"]["decoder_input_llr"].shape):
        raise ValueError("External codeword ordering shape differs from EP5 contract")
    decoded, crc = link.codec.decode(llrs_to_native_order(llr).to(link.device))
    outcomes, totals = payload_statistics(channel["info_bits"], decoded.cpu(), crc.cpu())
    metrics = soft_statistics(llr, labels)
    raw_block = channel["methods"]["ep5"]["block_error"]
    row = dict(method=METHODS[-1], candidate="external_frozen_transfer", training_seed=None,
               checkpoint_type="frozen_reference6000", step=6000, parameter_count=4_247_644,
               seed=int(channel["seed"]), ebno_db=float(channel["ebno_db"]),
               **totals, **metrics,
               rescue_vs_raw=int((raw_block & ~outcomes["block_error"]).sum()),
               harm_vs_raw=int((~raw_block & outcomes["block_error"]).sum()),
               per_ue={key: value.tolist() for key, value in outcomes.items()
                       if key != "decoded_payload_bits"},
               inference_seed=audit["native_seed"],
               decoded_payload_identity=hashlib.sha256(
                   outcomes["decoded_payload_bits"].contiguous().numpy().tobytes()).hexdigest())
    return row, {"external_seconds": sum(x["inference"]["wall_seconds"] for x in audit["native_audit"]),
                 "native_audit": audit["native_audit"]}


def load_sample_once(link, ep, seed, ebno, config, check_codec):
    captured = {}
    original = link.transmit

    def capture(*args, **kwargs):
        sample = original(*args, **kwargs)
        captured["sample"] = sample
        return sample

    link.transmit = capture
    try:
        channel, timings = stream_observation(link, ep, seed, ebno, config, check_codec=check_codec)
    finally:
        link.transmit = original
    if "sample" not in captured:
        raise RuntimeError("The shared transmission was not captured")
    channel["_sample"] = captured["sample"]
    return channel, timings


def write_csv(path, report):
    rows = [row for case in report["cases"] for row in case["receivers"]]
    fields = ("seed", "ebno_db", "method", "training_seed", "blocks", "block_errors",
              "tb_bler", "payload_bits", "payload_bit_errors", "payload_ber", "crc_failures",
              "undetected_block_errors", "coded_bits", "coded_bit_errors", "coded_ber", "bce_nats",
              "rescue_vs_raw", "harm_vs_raw")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def summarize(report):
    grouped = {}
    for case in report["cases"]:
        for row in case["receivers"]:
            key = (row["ebno_db"], row["method"])
            total = grouped.setdefault(key, dict(method=row["method"], ebno_db=row["ebno_db"],
                                                  blocks=0, block_errors=0, payload_bits=0,
                                                  payload_bit_errors=0, crc_failures=0,
                                                  undetected_block_errors=0, coded_bits=0,
                                                  coded_bit_errors=0, bce_sum_nats=0.0,
                                                  rescue_vs_raw=0, harm_vs_raw=0))
            for field in ("blocks", "block_errors", "payload_bits", "payload_bit_errors",
                          "crc_failures", "undetected_block_errors", "coded_bits", "coded_bit_errors"):
                total[field] += int(row[field])
            total["rescue_vs_raw"] += int(row["rescue_vs_raw"])
            total["harm_vs_raw"] += int(row["harm_vs_raw"])
            total["bce_sum_nats"] += float(row["bce_sum_nats"])
    rows = []
    for total in grouped.values():
        total["tb_bler"] = total["block_errors"] / total["blocks"]
        total["payload_ber"] = total["payload_bit_errors"] / total["payload_bits"]
        total["coded_ber"] = total["coded_bit_errors"] / total["coded_bits"]
        total["bce_nats"] = total["bce_sum_nats"] / total["coded_bits"]
        rows.append(total)
    report["summary"] = sorted(rows, key=lambda x: (x["ebno_db"], x["method"]))


def run(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {out}")
    seeds = [int(x) for x in args.seeds.split(",") if x]
    ebnos = [float(x) for x in args.ebno_dbs.split(",") if x]
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in range(81_000_024, 81_000_032) for s in seeds):
        raise ValueError("This diagnostic pilot is restricted to the eight old validation seeds 81000024..81000031")
    if not ebnos or len(ebnos) != len(set(ebnos)):
        raise ValueError("Eb/N0 list must be nonempty and duplicate-free")

    config = load_config(args.config, mode="confirmation")
    context = prepare(config, require_gpu=True)
    trained, all_receivers, training_artifacts = load_receivers(args.training_root, context, config, load_models=True)
    receiver_by_name = {r["method"]: r for r in all_receivers}
    selected = [receiver_by_name[name] for name in METHODS[:-1]]
    if len(selected) != len(METHODS) - 1:
        raise ValueError("Required native, EP5, frozen noise-per-bit and three MLP receiver checkpoints are missing")
    link, ep = link_and_ep(context, config)
    adapter = R3P5Reference6000Adapter(device=config["runtime"]["device"])
    points, bit_labels = adapter.receiver.constellation()
    actual = link.codec.mapper(bit_labels.to(link.device).reshape(1, 1, -1)).reshape(-1).cpu()
    adapter.check_constellation(points, bit_labels)
    torch.testing.assert_close(actual, points, atol=2e-6, rtol=2e-6)

    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", kind="exploratory_paired_transfer_pilot", schema_version=1,
                  split="old_validation", seeds=seeds, ebno_dbs=ebnos,
                  independent_channels=len(seeds), channel_ebno_observations=len(seeds) * len(ebnos),
                  warning="Diagnostic pilot only. No tuning or checkpoint selection. R3P5 training seeds are undisclosed; independence is unverified. Do not treat as the 512-channel confirmation.",
                  receiver_methods=list(METHODS), training_root=str(Path(args.training_root).resolve()),
                  training_artifacts=training_artifacts, environment=context["environment"],
                  package=dict(root=str(adapter.root), archive_sha256=adapter.archive_sha256,
                               weights_sha256=adapter.metadata["weights_sha256"], model_sha256=adapter.metadata["model_file_sha256"],
                               source_manifest_sha256=sha256(adapter.root / "BUNDLE_MANIFEST.json"),
                               interface="native K64 Direct-RB; one full 2304-data-RE window; positive log(P1/P0)"),
                  code_commit=subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
                  code_dirty=True, cases=[])
    save_json(out / "summary.json", report)
    started = time.perf_counter()
    count = len(seeds) * len(ebnos)
    for number, (seed, ebno) in enumerate(((s, e) for s in seeds for e in ebnos), 1):
        print(f"Pilot {number}/{count} seed={seed} Eb/N0={ebno:g} START", flush=True)
        channel, timings = load_sample_once(link, ep, seed, ebno, config, check_codec=(number == 1))
        rows = evaluate_observation(channel, selected, link.codec, config, trained["normalization"], timings)
        external_row, external_timing = add_pilot_receiver(channel, link, adapter, seed)
        rows.append(external_row)
        if {row["method"] for row in rows} != set(METHODS):
            raise ValueError("Paired receiver set is incomplete")
        case = dict(seed=seed, ebno_db=ebno, independent_cluster=seed,
                    shared_transmission=True, complete_codeword=True,
                    timings={**timings, **external_timing}, receivers=rows)
        report["cases"].append(case)
        del channel["_sample"]
        summarize(report)
        report["elapsed_seconds"] = time.perf_counter() - started
        save_json(out / "summary.json", report)
        write_csv(out / "summary.csv", report)
        print("  " + " ".join(f"{r['method']}={r['block_errors']}/{r['blocks']}"
                              for r in rows), flush=True)
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    summarize(report)
    save_json(out / "summary.json", report)
    write_csv(out / "summary.csv", report)
    print(f"Pilot complete: {out / 'summary.json'}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/llr_mechanism_controls.yaml")
    parser.add_argument("--training-root", default=DEFAULT_TRAINING)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="81000024,81000025,81000026,81000027,81000028,81000029,81000030,81000031")
    parser.add_argument("--ebno-dbs", default="-12,-11.5,-11,-10.5,-10,-9.5,-9")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
