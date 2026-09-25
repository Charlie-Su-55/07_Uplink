from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import PanelArray, UMa

from data.channels.topology import generate_uma_topology
from data.link.resource_grid import UplinkResourceGridSpec


def _build_panel_array(cfg, carrier_frequency, precision, device):
    return PanelArray(num_rows_per_panel=int(cfg["num_rows_per_panel"]),
                      num_cols_per_panel=int(cfg["num_cols_per_panel"]),
                      polarization=cfg["polarization"], polarization_type=cfg["polarization_type"],
                      antenna_pattern=cfg["antenna_pattern"], carrier_frequency=carrier_frequency,
                      element_vertical_spacing=float(cfg["element_vertical_spacing_lambda"]),
                      element_horizontal_spacing=float(cfg["element_horizontal_spacing_lambda"]),
                      precision=precision, device=device)


class UMAChannelProvider:
    def __init__(self, cfg, resource_grid=None):
        self.cfg = cfg
        gcfg, tcfg = cfg["general"], cfg["topology"]
        self.device = gcfg["device"]
        self.precision = gcfg["precision"]
        self.num_ues = int(gcfg["num_ues"])
        self.carrier_frequency = float(gcfg["carrier_frequency_hz"])
        self.resource_grid = resource_grid if resource_grid is not None else UplinkResourceGridSpec.from_config(cfg)

        self.bs_array = _build_panel_array(cfg["bs_array"], self.carrier_frequency, self.precision, self.device)
        self.ut_array = _build_panel_array(cfg["ut_array"], self.carrier_frequency, self.precision, self.device)

        self.channel_model = UMa(carrier_frequency=self.carrier_frequency, o2i_model=tcfg["o2i_model"],
                                 ut_array=self.ut_array, bs_array=self.bs_array, direction="uplink",
                                 enable_pathloss=bool(tcfg["enable_pathloss"]),
                                 enable_shadow_fading=bool(tcfg["enable_shadow_fading"]),
                                 always_generate_lsp=bool(tcfg["always_generate_lsp"]),
                                 precision=self.precision, device=self.device)

        self.channel_generator = GenerateOFDMChannel(self.channel_model, self.resource_grid,
                                                     normalize_channel=bool(cfg["channel"]["normalize_channel"]),
                                                     precision=self.precision, device=self.device)
        self._batch_size = None

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if self._batch_size is not None and batch_size != self._batch_size:
            self.channel_model.reset_topology()

        topology = generate_uma_topology(batch_size, self.num_ues, self.cfg, self.device)
        los = False if self.cfg["topology"]["force_nlos"] else None
        self.channel_model.set_topology(topology["ut_loc"], topology["bs_loc"],
                                        topology["ut_orientations"], topology["bs_orientations"],
                                        topology["ut_velocities"], topology["in_state"], los=los)
        self._batch_size = batch_size
        h_freq = self.channel_generator(batch_size=batch_size)
        return h_freq, topology