"""Eb/N0 accounting for the independent reference (no received-power input).

Imports remain lazy so CLI help and accounting tests work on the audit laptop.
The implementation targets Sionna 2.0.1; no compatibility path for Sionna 0.x.
"""

import importlib.metadata
import math


def require_sionna2():
    try:
        version = importlib.metadata.version("sionna")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Reference requires the server's Sionna 2.0.1 environment.") from exc
    if version.split(".")[0] != "2":
        raise RuntimeError(f"Reference targets Sionna 2.0.1; found {version}. Use the GPU server.")
    return version


def effective_payload_rate(info_bits, coded_bits):
    if (not isinstance(info_bits, int) or not isinstance(coded_bits, int)
            or isinstance(info_bits, bool) or isinstance(coded_bits, bool)
            or not 0 < info_bits <= coded_bits):
        raise ValueError("Expected positive integer payload <= coded bits per UE.")
    return info_bits / coded_bits


def ebno_to_noise(ebno_db, bits_per_symbol, info_bits, coded_bits,
                  resource_grid, *, precision="single", device="cpu"):
    """Complex AWGN variance per Rx/RE, including the actual grid's CP overhead.

    Each transmitter is one UE with one unit-energy stream. In particular,
    num_tx does not divide the energy; num_streams_per_tx must be one.
    """
    if not math.isfinite(float(ebno_db)):
        raise ValueError("Eb/N0 must be finite.")
    if bits_per_symbol not in (2, 4, 6, 8):
        raise ValueError("Expected QPSK/16/64/256-QAM.")
    rate = effective_payload_rate(info_bits, coded_bits)
    if resource_grid is None or resource_grid.num_streams_per_tx != 1:
        raise ValueError("Use a real ResourceGrid with one stream per UE.")
    if int(resource_grid.num_data_symbols) * bits_per_symbol != coded_bits:
        raise ValueError("Coded length does not fill the transmitted data grid.")
    require_sionna2()
    from sionna.phy.utils import ebnodb2no

    no = ebnodb2no(float(ebno_db), bits_per_symbol, rate,
                   resource_grid=resource_grid, precision=precision, device=device)
    value = float(no.item())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Eb/N0 produces non-positive/non-finite N0 in the selected precision.")
    return no
