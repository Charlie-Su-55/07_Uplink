"""Power, channel ordering, and LMMSE diagnostics for the coded reference."""

import argparse

from evaluation.evaluate_sionna_bler import (
    block_statistics, common_arguments, config_from_args, output_path,
    positive_int, provenance, save_json,
)
from link_level.sionna_reference import (
    SionnaReferenceLink, power_diagnostics, compare_classical_lmmse,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser, "results/sionna_reference/power.json")
    parser.add_argument("--channels", type=positive_int, default=2)
    parser.add_argument("--ebno-db", type=float, default=0.0)
    parser.add_argument("--compare-classical", action="store_true",
                        help="Compare the existing LMMSE core on 16 REs using exact N0 I.")
    args = parser.parse_args(argv)
    path = output_path(args)
    cfg = config_from_args(args)
    link = SionnaReferenceLink(cfg)
    report = dict(schema_version=1, status="running", experiment="sionna_reference_power",
                  provenance=provenance(args), reference=link.metadata(), batches=[])
    save_json(path, report)
    try:
        for offset in range(0, args.channels, args.batch_size):
            sample = link.transmit(min(args.batch_size, args.channels - offset), args.ebno_db,
                                   int(cfg["general"]["seed"]) + offset)
            if "codec_identity" not in report:
                report["codec_identity"] = link.check_codec_identity(sample)
            received = link.receive(sample)
            diagnostic = power_diagnostics(sample)
            diagnostic["counts"] = block_statistics(sample, received)
            if args.compare_classical:
                diagnostic["classical_lmmse_comparison"] = compare_classical_lmmse(link, sample, received)
            report["batches"].append(diagnostic)
            save_json(path, report)
            if diagnostic["reconstruction_relative_rms"] > (1e-5 if link.precision == "single" else 1e-10):
                raise RuntimeError("H_eff*x does not reconstruct the actually transmitted waveform.")
            print(f"seed={sample['seed']} N0={diagnostic['n0']:.6g} "
                  f"noise/N0={diagnostic['measured_noise_over_n0']} "
                  f"reconstruction={diagnostic['reconstruction_relative_rms']:.3g}", flush=True)
            del sample, received
        report["status"] = "complete"
        save_json(path, report)
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_json(path, report)
        raise
    print(f"Saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
