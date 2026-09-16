import torch


class InterleavedDMRS:
    def __init__(self, cfg, resource_grid):
        self.grid = resource_grid
        self.num_streams = int(cfg["general"]["num_ues"]) * int(cfg["general"]["streams_per_ue"])
        self.num_subcarriers = resource_grid.num_subcarriers
        self.symbols = tuple(resource_grid.dmrs_symbols)
        self.base_pilot_amplitude = float(cfg["dmrs"]["pilot_amplitude"])
        self.pilot_amplitude = self.base_pilot_amplitude * (self.num_streams ** 0.5)
        self.scheme = cfg["dmrs"]["scheme"]

        if self.scheme != "interleaved_comb":
            raise ValueError(f"Unsupported DMRS scheme: {self.scheme}")
        if self.num_subcarriers % self.num_streams != 0:
            raise ValueError("num_subcarriers must be divisible by num_streams for interleaved comb DMRS.")

        self.pilot_indices = [tuple(range(k, self.num_subcarriers, self.num_streams)) for k in range(self.num_streams)]
        self.num_pilots_per_stream = self.num_subcarriers // self.num_streams

    def sample(self, batch_size, dtype, device):
        x_dmrs = torch.zeros(int(batch_size), self.grid.num_ofdm_symbols, self.num_subcarriers,
                             self.num_streams, dtype=dtype, device=device)
        pilot = torch.tensor(self.pilot_amplitude, dtype=dtype, device=device)

        for s in self.symbols:
            for k, idx in enumerate(self.pilot_indices):
                x_dmrs[:, s, list(idx), k] = pilot

        return {"x_dmrs": x_dmrs}