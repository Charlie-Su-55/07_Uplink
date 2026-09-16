import torch
from sionna.phy.mapping import QAMSource


class UplinkDataGenerator:
    def __init__(self, cfg, resource_grid):
        mcfg, gcfg = cfg["modulation"], cfg["general"]
        if mcfg["type"].lower() != "qam":
            raise ValueError("Only QAM is supported in the current implementation.")

        self.num_bits_per_symbol = int(mcfg["bits_per_symbol"])
        self.num_streams = int(gcfg["num_ues"]) * int(gcfg["streams_per_ue"])
        self.grid = resource_grid
        self.device = gcfg["device"]
        self.source = QAMSource(self.num_bits_per_symbol, return_bits=True,
                                precision=gcfg["precision"], device=self.device)

    def sample_symbols(self, shape):
        return self.source(list(shape))

    def sample(self, batch_size):
        batch_size = int(batch_size)
        num_data_symbols = len(self.grid.data_symbols)
        shape = [batch_size, num_data_symbols, self.grid.num_subcarriers, self.num_streams]

        x_data, bits = self.sample_symbols(shape)
        x_grid = torch.zeros(batch_size, self.grid.num_ofdm_symbols, self.grid.num_subcarriers,
                             self.num_streams, dtype=x_data.dtype, device=x_data.device)
        x_grid[:, list(self.grid.data_symbols)] = x_data

        return {
            "bits": bits,
            "x_data": x_data,
            "x_grid": x_grid,
        }