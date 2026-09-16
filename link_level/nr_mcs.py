#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

from sionna.phy.nr.utils import decode_mcs_index


@dataclass(frozen=True)
class NRMCS:
    table: int
    index: int
    bits_per_symbol: int
    target_coderate: float
    spectral_efficiency: float


def get_pusch_mcs(table: int, index: int, device: str = "cpu") -> NRMCS:
    if table not in {1, 2}:
        raise ValueError("This experiment uses 3GPP MCS Table 1 or Table 2 only.")

    qm, rate = decode_mcs_index(
        mcs_index=index,
        table_index=table,
        is_pusch=True,
        transform_precoding=False,
        pi2bpsk=False,
        check_index_validity=True,
        device=device,
    )

    qm = int(qm.item())
    rate = float(rate.item())

    return NRMCS(
        table=int(table),
        index=int(index),
        bits_per_symbol=qm,
        target_coderate=rate,
        spectral_efficiency=qm * rate,
    )


def selected_mcs():
    return [
        get_pusch_mcs(table, mcs)
        for table in (1, 2)
        for mcs in (1, 4, 5, 11)
    ]


if __name__ == "__main__":
    print("3GPP NR PUSCH MCS | transform precoding disabled")
    print("=" * 72)

    for x in selected_mcs():
        modulation = {
            2: "QPSK",
            4: "16-QAM",
            6: "64-QAM",
            8: "256-QAM",
        }.get(x.bits_per_symbol, f"{1 << x.bits_per_symbol}-QAM")

        print(
            f"T{x.table} MCS{x.index:2d} | "
            f"{modulation:7s} | "
            f"Qm={x.bits_per_symbol} | "
            f"R={x.target_coderate:.6f} | "
            f"SE={x.spectral_efficiency:.4f}"
        )