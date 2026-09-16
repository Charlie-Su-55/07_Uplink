import argparse
import json
from pathlib import Path

import torch
import yaml
from sionna.phy import config as sionna_config

from data.dataset import UplinkUMADataset
from detectors.classical.ep import ExpectationPropagationDetector
from detectors.classical.lmmse import LMMSESoftDetector
from models.graph.csi_robust_gt import CSIRobustGraphTransformer
from models.graph.gt_ep_detector import GraphTransformerEPDetector


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def whiten(y, h, ruu):
    chol = torch.linalg.cholesky(ruu)
    if chol.shape[0] == 1 and y.shape[0] > 1:
        chol = chol.expand(y.shape[0], -1, -1)
    y_white = torch.linalg.solve_triangular(chol, y.unsqueeze(-1), upper=False).squeeze(-1)
    h_white = torch.linalg.solve_triangular(chol, h, upper=False)
    return y_white, h_white


def sufficient_statistics(y, h):
    z = torch.einsum("bmk,bm->bk", h.conj(), y)
    gram = torch.einsum("bmk,bml->bkl", h.conj(), h)
    return z, gram


def errors(llr, bits):
    return ((llr > 0) != (bits > 0.5)).sum().item()


def fixed_broken(reference, method, bits):
    truth = bits > 0.5
    ref_wrong = (reference > 0) != truth
    method_wrong = (method > 0) != truth
    fixed = (ref_wrong & ~method_wrong).sum().item()
    broken = (~ref_wrong & method_wrong).sum().item()
    return fixed, broken


def bootstrap(rows, baseline, method, n_boot=10000, seed=12345):
    base = torch.tensor([x[baseline] for x in rows], dtype=torch.float64)
    test = torch.tensor([x[method] for x in rows], dtype=torch.float64)
    diff = base - test

    g = torch.Generator().manual_seed(seed)
    n = len(rows)
    samples = []

    for start in range(0, n_boot, 1000):
        count = min(1000, n_boot - start)
        idx = torch.randint(0, n, (count, n), generator=g)
        samples.append(diff[idx].mean(dim=1))

    samples = torch.cat(samples)

    return {
        "delta": diff.mean().item(),
        "low": torch.quantile(samples, 0.025).item(),
        "high": torch.quantile(samples, 0.975).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sgt_5db.yaml")
    parser.add_argument("--gt-ep-checkpoint", default="ckp/gt_ep_256rx_16ue_lsH_estR_5db/best.pth")
    parser.add_argument("--crgt-checkpoint", default="ckp/csi_robust_gt_256rx_16ue_lsH_estR_5db/best.pth")
    parser.add_argument("--channels", type=int, default=500)
    parser.add_argument("--re-per-channel", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=20260910)
    parser.add_argument("--output", default="results/gt_ep_eval/gt_ep_500ch_128re_5db.json")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    cfg = load_yaml(train_cfg["system_config"])
    cfg["link"]["rx_snr_db"] = float(train_cfg.get("training", {}).get("snr_db", 5.0))

    device = cfg["general"]["device"]

    sionna_config.device = device
    sionna_config.precision = cfg["general"]["precision"]
    sionna_config.seed = args.eval_seed
    torch.manual_seed(args.eval_seed)
    torch.set_float32_matmul_precision("high")

    dataset = UplinkUMADataset(cfg)

    ep = ExpectationPropagationDetector(cfg, num_iterations=5, damping=0.5)
    lmmse = LMMSESoftDetector(cfg)

    gt_ep = GraphTransformerEPDetector(
        cfg=cfg,
        num_users=16,
        num_iterations=5,
        damping=0.5,
        d_model=128,
        num_heads=8,
        num_layers=4,
        edge_dim=32,
        ffn_dim=256,
        dropout=0.05,
        max_logit_correction=4.0,
    ).to(device)

    ckpt_gt = torch.load(args.gt_ep_checkpoint, map_location=device, weights_only=False)
    gt_ep.load_state_dict(ckpt_gt["model_state"])
    gt_ep.eval()

    crgt = CSIRobustGraphTransformer(
        num_rx=256,
        num_users=16,
        d_model=128,
        num_heads=8,
        num_layers=8,
        edge_dim=32,
        ffn_dim=256,
        dropout=0.05,
        correction_scale=1.0,
    ).to(device)

    ckpt_crgt = torch.load(args.crgt_checkpoint, map_location=device, weights_only=False)
    crgt.load_state_dict(ckpt_crgt["model_state"])
    crgt.eval()

    names = ["lmmse", "ep5", "crgt_ep5", "gt_ep", "trueH_ep5"]
    total_errors = {name: 0 for name in names}

    rows = []
    total_bits = 0

    gt_fixed = 0
    gt_broken = 0
    gt_vs_crgt_fixed = 0
    gt_vs_crgt_broken = 0

    ep_valid_sum = torch.zeros(5, dtype=torch.float64)
    gt_valid_sum = torch.zeros(5, dtype=torch.float64)
    gt_corr_sum = torch.zeros(5, dtype=torch.float64)

    with torch.no_grad():
        for channel in range(1, args.channels + 1):
            batch = dataset.sample(1)
            data_idx = list(dataset.channel.resource_grid.data_symbols)

            y_full = batch["y"][:, data_idx]
            h_ls_full = batch["h_hat_ls"][:, data_idx]
            h_true_full = batch["h_true"][:, data_idx]
            bits_full = batch["bits"].reshape(-1, 16, 4).float()

            num_re = bits_full.shape[0]
            count = min(args.re_per_channel, num_re)
            idx = torch.randperm(num_re, device=device)[:count]

            y = y_full.reshape(-1, 256)[idx]
            h_ls = h_ls_full.reshape(-1, 256, 16)[idx]
            h_true = h_true_full.reshape(-1, 256, 16)[idx]
            bits = bits_full[idx]

            llr_lmmse = lmmse(y_full, h_ls_full, batch["ruu_hat"])["llr"].reshape(-1, 16, 4)[idx].float()

            y_w, h_ls_w = whiten(y, h_ls, batch["ruu_hat"])
            _, h_true_w = whiten(y, h_true, batch["ruu_hat"])

            z, gram = sufficient_statistics(y_w, h_ls_w)
            z_true, gram_true = sufficient_statistics(y_w, h_true_w)

            ep_out = ep(z, gram, return_iterations=(5,))
            llr_ep = ep_out["llr"].float()

            crgt_out = crgt(y_w, h_ls_w)
            llr_crgt = ep(crgt_out["z_refined"], crgt_out["gram_refined"], return_iterations=(5,))["llr"].float()

            gt_out = gt_ep(z, gram, return_iterations=(5,))
            llr_gt = gt_out["llr"].float()

            llr_true = ep(z_true, gram_true, return_iterations=(5,))["llr"].float()

            outputs = {
                "lmmse": llr_lmmse,
                "ep5": llr_ep,
                "crgt_ep5": llr_crgt,
                "gt_ep": llr_gt,
                "trueH_ep5": llr_true,
            }

            nbits = bits.numel()
            total_bits += nbits

            row = {"channel": channel}

            for name, llr in outputs.items():
                err = errors(llr, bits)
                total_errors[name] += err
                row[name] = err / nbits

            fixed, broken = fixed_broken(llr_ep, llr_gt, bits)
            gt_fixed += fixed
            gt_broken += broken

            fixed, broken = fixed_broken(llr_crgt, llr_gt, bits)
            gt_vs_crgt_fixed += fixed
            gt_vs_crgt_broken += broken

            ep_valid_sum += torch.tensor(ep_out["valid_update_fraction"], dtype=torch.float64)
            gt_valid_sum += gt_out["valid_update_fraction"].double().cpu()
            gt_corr_sum += gt_out["correction_rms"].double().cpu()

            rows.append(row)

            if channel == 1 or channel % 25 == 0:
                ber_now = {name: total_errors[name] / total_bits for name in names}
                print(
                    f"{channel:4d}/{args.channels} | "
                    f"LMMSE={ber_now['lmmse']:.6e} | "
                    f"EP5={ber_now['ep5']:.6e} | "
                    f"CRGT={ber_now['crgt_ep5']:.6e} | "
                    f"GT-EP={ber_now['gt_ep']:.6e} | "
                    f"TrueH={ber_now['trueH_ep5']:.6e}"
                )

    ber = {name: total_errors[name] / total_bits for name in names}

    comparisons = {
        "gt_ep_vs_ep5": bootstrap(rows, "ep5", "gt_ep", seed=args.eval_seed + 1),
        "gt_ep_vs_crgt": bootstrap(rows, "crgt_ep5", "gt_ep", seed=args.eval_seed + 2),
        "gt_ep_vs_lmmse": bootstrap(rows, "lmmse", "gt_ep", seed=args.eval_seed + 3),
        "crgt_vs_ep5": bootstrap(rows, "ep5", "crgt_ep5", seed=args.eval_seed + 4),
    }

    ep_valid = ep_valid_sum / args.channels
    gt_valid = gt_valid_sum / args.channels
    gt_corr = gt_corr_sum / args.channels

    result = {
        "gt_ep_checkpoint_step": ckpt_gt.get("step", -1),
        "crgt_checkpoint_step": ckpt_crgt.get("step", -1),
        "channels": args.channels,
        "re_per_channel": args.re_per_channel,
        "total_bits": total_bits,
        "ber": ber,
        "relative_gain_gt_ep_vs_ep5": (ber["ep5"] - ber["gt_ep"]) / ber["ep5"],
        "relative_gain_gt_ep_vs_crgt": (ber["crgt_ep5"] - ber["gt_ep"]) / ber["crgt_ep5"],
        "relative_gain_gt_ep_vs_lmmse": (ber["lmmse"] - ber["gt_ep"]) / ber["lmmse"],
        "gt_ep_fixed_ep5_bits": gt_fixed,
        "gt_ep_broken_ep5_bits": gt_broken,
        "gt_ep_net_fixed_ep5_bits": gt_fixed - gt_broken,
        "gt_ep_fixed_crgt_bits": gt_vs_crgt_fixed,
        "gt_ep_broken_crgt_bits": gt_vs_crgt_broken,
        "gt_ep_net_fixed_crgt_bits": gt_vs_crgt_fixed - gt_vs_crgt_broken,
        "ep5_valid_update_fraction": ep_valid.tolist(),
        "gt_ep_valid_update_fraction": gt_valid.tolist(),
        "gt_ep_correction_rms": gt_corr.tolist(),
        "bootstrap": comparisons,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 120)
    print("FINAL INDEPENDENT EVALUATION")
    print("=" * 120)
    print(f"GT-EP checkpoint step : {ckpt_gt.get('step', -1)}")
    print(f"CR-GT checkpoint step : {ckpt_crgt.get('step', -1)}")
    print()
    print(f"LS-H + Est-R LMMSE       : {ber['lmmse']:.8e}")
    print(f"LS-H + Est-R EP5         : {ber['ep5']:.8e}")
    print(f"LS-H + Est-R CR-GT + EP5 : {ber['crgt_ep5']:.8e}")
    print(f"LS-H + Est-R GT-EP       : {ber['gt_ep']:.8e}")
    print(f"TrueH + Est-R EP5        : {ber['trueH_ep5']:.8e}")

    print()
    print(f"GT-EP vs LMMSE : {100 * (ber['lmmse'] - ber['gt_ep']) / ber['lmmse']:+.3f}%")
    print(f"GT-EP vs EP5   : {100 * (ber['ep5'] - ber['gt_ep']) / ber['ep5']:+.3f}%")
    print(f"GT-EP vs CR-GT : {100 * (ber['crgt_ep5'] - ber['gt_ep']) / ber['crgt_ep5']:+.3f}%")

    print()
    print(f"GT-EP vs EP5 fixed/broken/net : {gt_fixed} / {gt_broken} / {gt_fixed - gt_broken}")
    print(f"GT-EP vs CRGT fixed/broken/net: {gt_vs_crgt_fixed} / {gt_vs_crgt_broken} / {gt_vs_crgt_fixed - gt_vs_crgt_broken}")

    print()
    print("EP5 site validity :", [round(x, 4) for x in ep_valid.tolist()])
    print("GT-EP site validity:", [round(x, 4) for x in gt_valid.tolist()])
    print("GT correction RMS :", [round(x, 4) for x in gt_corr.tolist()])

    print()
    print("PAIRED CHANNEL BOOTSTRAP")
    for name, item in comparisons.items():
        print(f"{name:<18s}: ΔBER={item['delta']:+.6e} 95%CI=[{item['low']:+.6e}, {item['high']:+.6e}]")

    print()
    print(f"Saved: {output}")
    print("=" * 120)


if __name__ == "__main__":
    main()