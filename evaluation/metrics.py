import torch


def hard_bits_from_llr(llr):
    return (llr > 0).to(torch.int32)


def bit_error_rate(llr, bits):
    bits_hat = hard_bits_from_llr(llr)
    bits_ref = bits.to(torch.int32)
    return (bits_hat != bits_ref).float().mean()


def per_stream_ber(llr, bits):
    bits_hat = hard_bits_from_llr(llr)
    bits_ref = bits.to(torch.int32)
    return (bits_hat != bits_ref).float().mean(dim=(0, 1, 2, 4))


def symbol_mse(x_hat, x):
    return (x_hat - x).abs().square().mean()