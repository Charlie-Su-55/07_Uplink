import torch

from evaluation.metrics import bit_error_rate


@torch.no_grad()
def evaluate_snr_sweep(dataset, detector, snr_db_list, num_batches=10, batch_size=1):
    results = []

    for snr_db in snr_db_list:
        dataset.link.rx_snr_db = float(snr_db)
        total_errors = 0
        total_bits = 0
        post_sinr_sum = 0.0
        num_post_sinr = 0

        for _ in range(num_batches):
            batch = dataset.sample(batch_size)
            data_idx = list(dataset.channel.resource_grid.data_symbols)

            y = batch["y"][:, data_idx]
            h = batch["h_hat_lmmse"][:, data_idx]
            ruu = batch["ruu_hat"]
            bits = batch["bits"]

            out = detector(y, h, ruu)
            llr = out["llr"]
            bits_hat = (llr > 0).to(bits.dtype)

            total_errors += (bits_hat != bits).sum().item()
            total_bits += bits.numel()

            post_sinr = 1.0 / out["no_eff"]
            post_sinr_sum += post_sinr.sum().item()
            num_post_sinr += post_sinr.numel()

        ber = total_errors / total_bits
        mean_post_sinr = post_sinr_sum / num_post_sinr
        mean_post_sinr_db = 10.0 * torch.log10(torch.tensor(mean_post_sinr)).item()

        results.append({
            "snr_db": float(snr_db),
            "ber": float(ber),
            "bit_errors": int(total_errors),
            "total_bits": int(total_bits),
            "mean_post_sinr_db": float(mean_post_sinr_db),
        })

        print(f"SNR={snr_db:>6.1f} dB | BER={ber:.6e} | errors={total_errors:>7d}/{total_bits:<9d} | mean post-SINR={mean_post_sinr_db:>7.2f} dB")

    return results