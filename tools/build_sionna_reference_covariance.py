"""Build an independent effective-channel covariance prior for Mode B only."""

import argparse
from pathlib import Path
import tempfile

from evaluation.evaluate_sionna_bler import common_arguments, config_from_args, output_path, positive_int, provenance
from link_level.sionna_reference import PAPER_CONFIG, SionnaReferenceLink
from link_level.sionna_ce import COVARIANCE_KIND, channel_second_moments, distribution_id, distribution_spec, regularize_covariance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser, "data/cache/paper_reference_ft_cov.pt")
    parser.set_defaults(config=PAPER_CONFIG, seed=4000000)
    parser.add_argument("--channels", type=positive_int, default=200)
    parser.add_argument("--shrinkage", type=float, default=1e-3)
    args = parser.parse_args(argv)
    path = output_path(args)
    if not 0 < args.shrinkage < 1:
        parser.error("--shrinkage must lie in (0,1).")
    cfg = config_from_args(args)
    cfg["channel_estimation"]["mode"] = "perfect"
    if cfg.get("mode") != "paper_reference":
        raise ValueError("This builder accepts only paper_reference configurations.")
    link = SionnaReferenceLink(cfg)
    import torch
    from sionna.phy import config

    freq_sum, time_sum, seeds = None, None, []
    with torch.no_grad():
        for offset in range(0, args.channels, args.batch_size):
            size = min(args.batch_size, args.channels - offset)
            seed = int(cfg["general"]["seed"]) + offset
            config.seed = seed
            raw, h_eff = link.sample_channel(size)
            rf, rt = channel_second_moments(h_eff)
            rf, rt = rf.to(torch.complex128) * size, rt.to(torch.complex128) * size
            freq_sum = rf if freq_sum is None else freq_sum + rf
            time_sum = rt if time_sum is None else time_sum + rt
            seeds.append(seed)
            del raw, h_eff
            if offset % (10 * args.batch_size) == 0 or offset + size == args.channels:
                print(f"Calibration channels {offset + size}/{args.channels}", flush=True)
    artifact = dict(kind=COVARIANCE_KIND, distribution_id=distribution_id(cfg),
                    distribution=distribution_spec(cfg), channels=args.channels, batch_size=args.batch_size,
                    calibration_seeds=seeds, shrinkage=args.shrinkage, provenance=provenance(args),
                    prior="E[h_i conj(h_j)] of effective H; pooled channels/UEs/Rx; absolute power retained",
                    cov_mat_freq=regularize_covariance(freq_sum / args.channels, args.shrinkage).cpu(),
                    cov_mat_time=regularize_covariance(time_sum / args.channels, args.shrinkage).cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(artifact, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Saved {path}; distribution={artifact['distribution_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
