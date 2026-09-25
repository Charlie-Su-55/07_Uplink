"""Mode B paired perfect/practical CSI and existing classical detector evaluation."""

import argparse

from evaluation.evaluate_sionna_bler import (
    add_statistics, assess_acceptance, block_statistics, common_arguments,
    config_from_args, ebno_grid, output_path, positive_int, provenance, save_json,
)
from link_level.detector_adapter import (
    ClassicalDetectorAdapter, compare_lmmse_outputs, make_detector_batch, validate_detector_request,
)
from link_level.sionna_ce import csi_diagnostics, distribution_id
from link_level.sionna_reference import PAPER_CONFIG, SionnaReferenceLink, power_diagnostics


def run(args):
    path = output_path(args)
    detectors = tuple(args.detectors.split(","))
    validate_detector_request(detectors, args.csi, args.custom_ce_policy)
    cfg = config_from_args(args)
    if cfg.get("mode") != "paper_reference":
        raise ValueError("This evaluator requires mode: paper_reference.")
    # Practical runs always include the requested same-realization oracle control.
    csi_modes = ("perfect", "practical") if args.csi in ("practical", "both") else ("perfect",)
    cfg["channel_estimation"]["mode"] = "practical" if "practical" in csi_modes else "perfect"
    if args.ce_covariance:
        cfg["channel_estimation"]["covariance_path"] = args.ce_covariance
    link = SionnaReferenceLink(cfg)
    if link.covariance_metadata is not None:
        eval_seeds = {int(cfg["general"]["seed"]) + i for i in range(0, args.channels, args.batch_size)}
        if eval_seeds.intersection(link.covariance_metadata["calibration_seeds"]):
            raise ValueError("Evaluation and CE calibration seeds overlap.")
    adapter = ClassicalDetectorAdapter(link, detectors, args.custom_ce_policy, args.re_chunk)
    report = dict(schema_version=1, status="running", experiment="paper_reference",
                  distribution_id=distribution_id(cfg), reference=link.metadata(), provenance=provenance(args),
                  requested_csi=args.csi, csi_modes=list(csi_modes), detectors=list(detectors), custom_ce_policy=args.custom_ce_policy,
                  existing_checkpoints_distribution="legacy-distribution", neural_enabled=False,
                  sampling="One transmit per batch/EbNo; same y, Hhat, err_var, N0, grid and bits for every detector; paired seeds across EbNo",
                  points=[])
    save_json(path, report)
    try:
        for ebno in args.ebno_dbs:
            point = dict(status="running", ebno_db=ebno, n0=float(link.noise_variance(ebno).item()),
                         receivers={}, diagnostics=[], batch_seeds=[])
            report["points"].append(point)
            for offset in range(0, args.channels, args.batch_size):
                size = min(args.batch_size, args.channels - offset)
                seed = int(cfg["general"]["seed"]) + offset
                sample = link.transmit(size, ebno, seed)
                if "codec_identity" not in report:
                    report["codec_identity"] = link.check_codec_identity(sample)
                if offset == 0:
                    point["first_batch_power_diagnostic"] = power_diagnostics(sample)
                    if point["first_batch_power_diagnostic"]["reconstruction_relative_rms"] > (1e-5 if link.precision == "single" else 1e-10):
                        raise RuntimeError("Physical waveform and effective channel reconstruction disagree.")
                batch_diagnostics = dict(first_channel=offset, batch_size=size, seed=seed, csi={})
                for csi in csi_modes:
                    batch = make_detector_batch(link, sample, csi)
                    batch_diagnostics["csi"][csi] = csi_diagnostics(batch)
                    outputs = adapter.evaluate(sample, batch)
                    comparison = compare_lmmse_outputs(outputs)
                    if comparison is not None:
                        batch_diagnostics["csi"][csi]["lmmse_comparison"] = comparison
                    for detector, output in outputs.items():
                        key = csi + "/" + detector
                        counts = point["receivers"].setdefault(key, {})
                        add_statistics(counts, block_statistics(sample, output))
                    del batch, outputs
                point["diagnostics"].append(batch_diagnostics)
                point["batch_seeds"].append(dict(first_channel=offset, batch_size=size, seed=seed))
                del sample
                save_json(path, report)
                if offset % (10 * args.batch_size) == 0 or offset + size == args.channels:
                    rates = " ".join(f"{key}={value['bler']:.5g}" for key, value in point["receivers"].items())
                    print(f"Eb/N0={ebno:g} channels={offset + size}/{args.channels} BLER {rates}", flush=True)
            point["status"] = "complete"
            save_json(path, report)
        # Retain Level-0's conservative screening as a report, not a neural gate
        # override. A requested 100-channel characterization can be unresolved.
        report["screening"] = {}
        for key in report["points"][0]["receivers"]:
            points = [dict(p["receivers"][key], ebno_db=p["ebno_db"], status=p["status"]) for p in report["points"]]
            report["screening"][key] = assess_acceptance(points, link.num_ues)
        report["status"] = "complete"
        save_json(path, report)
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_json(path, report)
        raise
    print(f"Saved {path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser, "results/paper_reference/bler.json")
    parser.set_defaults(config=PAPER_CONFIG)
    parser.add_argument("--csi", choices=("perfect", "practical", "both"), default="perfect",
                        help="Practical includes a paired perfect-CSI control; both is an explicit paired-run alias.")
    parser.add_argument("--ce-covariance", help="Independent Mode B covariance artifact for practical CSI.")
    parser.add_argument("--channels", type=positive_int, default=100)
    parser.add_argument("--ebno-dbs", type=ebno_grid, default=ebno_grid("-24,-22,-20,-18,-16,-14,-12,-10,-8"))
    parser.add_argument("--detectors", default="sionna_lmmse", help="Comma list: sionna_lmmse,custom_lmmse,ep5.")
    parser.add_argument("--custom-ce-policy", choices=("require_explicit", "sionna_diagonal"), default="require_explicit")
    parser.add_argument("--re-chunk", type=positive_int, default=128)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
