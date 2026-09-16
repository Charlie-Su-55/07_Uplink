from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class UplinkResourceGridSpec:
    num_ofdm_symbols: int
    fft_size: int
    subcarrier_spacing: float
    cyclic_prefix_length: int
    omitted_symbols: tuple
    dmrs_symbols: tuple
    data_symbols: tuple

    @classmethod
    def from_config(cls, cfg):
        c = cfg["ofdm"]
        grid = cls(num_ofdm_symbols=int(c["num_ofdm_symbols"]), fft_size=int(c["fft_size"]),
                   subcarrier_spacing=float(c["subcarrier_spacing_hz"]),
                   cyclic_prefix_length=int(c["cyclic_prefix_length"]),
                   omitted_symbols=tuple(c["omitted_symbols"]),
                   dmrs_symbols=tuple(c["dmrs_symbols"]),
                   data_symbols=tuple(c["data_symbols"]))
        grid.validate()
        return grid

    @property
    def num_subcarriers(self):
        return self.fft_size

    @property
    def ofdm_symbol_duration(self):
        return (1.0 + self.cyclic_prefix_length / self.fft_size) / self.subcarrier_spacing

    def validate(self):
        valid = set(range(self.num_ofdm_symbols))
        omitted, dmrs, data = set(self.omitted_symbols), set(self.dmrs_symbols), set(self.data_symbols)
        assert omitted <= valid and dmrs <= valid and data <= valid, "Invalid OFDM symbol index."
        assert not (omitted & dmrs or omitted & data or dmrs & data), "OFDM symbol sets overlap."
        assert omitted | dmrs | data == valid, "Every OFDM symbol must be assigned exactly once."

    def symbol_type_map(self, device="cpu"):
        mask = torch.full((self.num_ofdm_symbols,), -1, dtype=torch.int64, device=device)
        mask[list(self.omitted_symbols)] = 0
        mask[list(self.dmrs_symbols)] = 1
        mask[list(self.data_symbols)] = 2
        return mask