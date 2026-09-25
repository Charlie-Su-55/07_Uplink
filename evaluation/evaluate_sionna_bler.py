"""Coded BLER versus per-UE payload Eb/N0, independent of legacy evaluators."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

from link_level.sionna_reference import DEFAULT_CONFIG, load_config, SionnaReferenceLink, power_diagnostics


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive.")
    return value


def ebno_grid(value):
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated Eb/N0 values.") from exc
    if not values or not all(math.isfinite(x) for x in values) or values != sorted(set(values)):
        raise argparse.ArgumentTypeError("Eb/N0 values must be finite, unique and increasing.")
    return values


def common_arguments(parser, default_output):
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Override device; 256Rx requires CUDA.")
    parser.add_argument("--table", type=int, choices=(1, 2), help="Override nr.mcs_table.")
    parser.add_argument("--mcs", type=int, help="Override nr.mcs_index.")
    parser.add_argument("--bp-iters", type=positive_int)
    parser.add_argument("--seed", type=int, help="Override general.seed.")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--overwrite", action="store_true")


def config_from_args(args):
    cfg = load_config(args.config)
    for arg, section, field in (("device", "general", "device"), ("seed", "general", "seed"),
                                 ("table", "nr", "mcs_table"), ("mcs", "nr", "mcs_index"),
                                 ("bp_iters", "nr", "num_bp_iter")):
        value = getattr(args, arg)
        if value is not None:
            cfg[section][field] = value
    if int(cfg["general"]["seed"]) < 0:
        raise ValueError("Seed must be nonnegative.")
    return cfg


def output_path(args):
    path = Path(args.output)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {path}. Choose a new path or pass --overwrite.")
    return path


def save_json(path, report):
    """Atomic checkpoints, with no NaN/Infinity in research artifacts."""
    contents = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def provenance(args):
    def git(*command):
        try:
            return subprocess.check_output(["git", *command], text=True, stderr=subprocess.DEVNULL,
                                           cwd=Path(__file__).resolve().parents[1]).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {"python": platform.python_version()}
    for package in ("torch", "sionna", "numpy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = git("status", "--porcelain")
    result = dict(arguments=vars(args), versions=versions, git_commit=git("rev-parse", "HEAD"),
                  git_branch=git("branch", "--show-current"),
                  git_dirty=None if status is None else bool(status), git_status=status)
    import torch
    result["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return result


def block_statistics(sample, received):
    """One independent channel cluster contains K simultaneous TBs."""
    bad_bits = received["decoded"] != sample["info"]
    bad_blocks = bad_bits.any(dim=-1)
    crc_bad = ~received["crc_ok"].bool()
    return dict(
        channels=int(bad_blocks.shape[0]), blocks=int(bad_blocks.numel()),
        bits=int(bad_bits.numel()), block_errors=int(bad_blocks.sum().item()),
        bit_errors=int(bad_bits.sum().item()), crc_failures=int(crc_bad.sum().item()),
        undetected_block_errors=int((bad_blocks & ~crc_bad).sum().item()),
        crc_failures_without_payload_error=int((~bad_blocks & crc_bad).sum().item()),
        per_ue_block_errors=bad_blocks.sum(0).cpu().tolist(),
        per_channel_block_errors=bad_blocks.sum(-1).cpu().tolist(),
        per_channel_bit_errors=bad_bits.sum((-1, -2)).cpu().tolist())


def add_statistics(point, counts):
    for key in ("channels", "blocks", "bits", "block_errors", "bit_errors", "crc_failures",
                "undetected_block_errors", "crc_failures_without_payload_error"):
        point[key] = point.get(key, 0) + counts[key]
    previous = point.setdefault("per_ue_block_errors", [0] * len(counts["per_ue_block_errors"]))
    point["per_ue_block_errors"] = [a + b for a, b in zip(previous, counts["per_ue_block_errors"])]
    for key in ("per_channel_block_errors", "per_channel_bit_errors"):
        point.setdefault(key, []).extend(counts[key])
    point["bler"] = point["block_errors"] / point["blocks"]
    point["ber"] = point["bit_errors"] / point["bits"]


def wilson_upper(successes, trials):
    """Upper endpoint of an approximate two-sided 95% binomial interval."""
    z = 1.959963984540054
    p = successes / trials
    return (p + z * z / (2 * trials) + z * math.sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials))) / (1 + z * z / trials)


def assess_acceptance(points, num_ues, low_bler=0.01, min_channels=100):
    """Conservative screening; never fit away a floor or smooth measured BLER.

    Simultaneous UEs are correlated. Use channel-level pairing for increases,
    and P(any TB error in a channel) as a conservative proxy upper bound for
    mean per-TB BLER. The Wilson interval is approximate, not certification.
    """
    ordered = sorted((p for p in points if p.get("status") == "complete"), key=lambda p: p["ebno_db"])
    if len(ordered) < 2:
        return {"status": "unresolved", "reason": "Need at least two completed Eb/N0 points."}
    brackets, increases = [], []
    for left, right in zip(ordered, ordered[1:]):
        if left["bler"] >= 0.1 and right["bler"] <= 0.1 and left["bler"] > right["bler"]:
            brackets.append([left["ebno_db"], right["ebno_db"]])
        a, b = left["per_channel_block_errors"], right["per_channel_block_errors"]
        if len(a) != len(b):
            raise ValueError("Acceptance requires paired channel counts at every Eb/N0.")
        delta = [(y - x) / num_ues for x, y in zip(a, b)]
        se = statistics.stdev(delta) / math.sqrt(len(delta)) if len(delta) > 1 else float("inf")
        if statistics.mean(delta) > max(3 * se, 1 / (num_ues * len(delta))):
            increases.append([left["ebno_db"], right["ebno_db"]])
    high = ordered[-1]
    n = high["channels"]
    upper = wilson_upper(sum(x > 0 for x in high["per_channel_block_errors"]), n)
    enough = min(p["channels"] for p in ordered) >= min_channels
    passed = enough and bool(brackets) and not increases and high["bler"] <= low_bler and upper <= low_bler
    reasons = []
    if not enough:
        reasons.append("Too few independent channels for acceptance.")
    if not brackets:
        reasons.append("BLER=0.1 is not bracketed by measured points; extend the Eb/N0 range.")
    if increases:
        reasons.append("Paired increases exceed the 3-SE screening threshold; investigate/repeat.")
    if high["bler"] > low_bler or upper > low_bler:
        reasons.append("High-Eb/N0 endpoint has not established low BLER; increase support or investigate a floor.")
    return dict(status="pass" if passed else "unresolved", reasons=reasons,
                measured_crossing_brackets=brackets, target_bler=0.1, low_bler_threshold=low_bler,
                minimum_channels=min_channels, high_endpoint_bler=high["bler"],
                high_endpoint_channel_any_error_wilson_upper95=upper,
                observed_monotone=all(b["bler"] <= a["bler"] for a, b in zip(ordered, ordered[1:])),
                significant_paired_increases=increases,
                uncertainty_unit="independent channel, not UE",
                neural_training_gate="reference_pass_required; persistent floor requires PHY investigation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser, "results/sionna_reference/bler.json")
    parser.add_argument("--channels", type=positive_int, default=1000, help="Independent channels per Eb/N0.")
    parser.add_argument("--ebno-dbs", type=ebno_grid, default=ebno_grid("-20,-16,-12,-8,-4,0,4,8,12"),
                        help="Increasing comma list; for negatives use --ebno-dbs=-20,-10,0.")
    parser.add_argument("--require-acceptance", action="store_true", help="Exit 2 unless the measured grid passes.")
    args = parser.parse_args(argv)
    path = output_path(args)
    cfg = config_from_args(args)
    link = SionnaReferenceLink(cfg)
    report = dict(schema_version=1, status="running", experiment="sionna_ebno_reference",
                  provenance=provenance(args), reference=link.metadata(), points=[],
                  sampling="Fixed channel budget; identical per-batch seeds across Eb/N0. Batch size affects draws.")
    save_json(path, report)
    try:
        for ebno in args.ebno_dbs:
            point = dict(status="running", ebno_db=ebno, n0=float(link.noise_variance(ebno).item()), batch_seeds=[])
            report["points"].append(point)
            for offset in range(0, args.channels, args.batch_size):
                size = min(args.batch_size, args.channels - offset)
                seed = int(cfg["general"]["seed"]) + offset
                sample = link.transmit(size, ebno, seed)
                if "codec_identity" not in report:
                    report["codec_identity"] = link.check_codec_identity(sample)
                if offset == 0:
                    point["first_batch_power_diagnostic"] = power_diagnostics(sample)
                    tolerance = 1e-5 if link.precision == "single" else 1e-10
                    if point["first_batch_power_diagnostic"]["reconstruction_relative_rms"] > tolerance:
                        raise RuntimeError("Physical antenna transmission and H_eff*x disagree.")
                received = link.receive(sample)
                add_statistics(point, block_statistics(sample, received))
                point["batch_seeds"].append({"first_channel": offset, "batch_size": size, "seed": seed})
                del sample, received
                save_json(path, report)
                if point["channels"] == args.channels or offset % (10 * args.batch_size) == 0:
                    print(f"Eb/N0={ebno:g} dB N0={point['n0']:.6g} channels={point['channels']}/{args.channels} "
                          f"BLER={point['bler']:.6g} ({point['block_errors']}/{point['blocks']})", flush=True)
            point["status"] = "complete"
            save_json(path, report)
        report["acceptance"] = assess_acceptance(report["points"], link.num_ues)
        report["acceptance"]["first_acceptance_case"] = (link.num_ues == 16 and link.num_rx == 256
            and link.codec.mcs.table == 1 and link.codec.mcs.index == 10)
        report["status"] = "complete"
        save_json(path, report)
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_json(path, report)
        raise
    print(f"Acceptance: {report['acceptance']['status']}; saved {path}", flush=True)
    return 2 if args.require_acceptance and report["acceptance"]["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
