from data.channels.uma import UMAChannelProvider
from data.link.stream_mapping import RankOneStreamMapper
from data.link.modulation import UplinkDataGenerator
from data.link.power_control import FractionalPowerController
from data.link.uplink_link import UplinkLink
from data.link.interference import UplinkInterferenceGenerator
from data.link.covariance import CovarianceEstimator
from data.link.dmrs import InterleavedDMRS
from data.link.channel_estimation import LSLinearChannelEstimator, LMMSEChannelEstimator


class UplinkUMADataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.channel = UMAChannelProvider(cfg)
        self.stream_mapper = RankOneStreamMapper(num_tx_antennas=self.channel.ut_array.num_ant, mode=cfg["stream_mapping"]["mode"], device=cfg["general"]["device"], precision=cfg["general"]["precision"])
        self.data_generator = UplinkDataGenerator(cfg, self.channel.resource_grid)
        self.power_controller = FractionalPowerController(cfg)
        self.link = UplinkLink(cfg, self.channel.resource_grid, self.stream_mapper)
        self.interference = UplinkInterferenceGenerator(cfg, protect_dmrs=True) if cfg["interference"]["enabled"] else None
        self.covariance_estimator = CovarianceEstimator(cfg, self.channel.resource_grid)
        self.dmrs = InterleavedDMRS(cfg, self.channel.resource_grid)
        self.channel_estimator_ls = LSLinearChannelEstimator(cfg, self.channel.resource_grid, self.dmrs)
        self.channel_estimator_lmmse = LMMSEChannelEstimator(
            cfg,
            self.channel.resource_grid,
            self.dmrs,
            covariance_path="data/cache/uma_lmmse_ft_cov.pt",
            order="f-t",
        )

    def sample(self, batch_size):
        h_raw, topology = self.channel.sample(batch_size)
        h_propagation = self.stream_mapper(h_raw)

        pc = self.power_controller(h_propagation)
        h_true = pc["h_powered"]

        tx = self.data_generator.sample(batch_size)
        dmrs = self.dmrs.sample(batch_size, tx["x_grid"].dtype, tx["x_grid"].device)
        x_grid = tx["x_grid"] + dmrs["x_dmrs"]

        clean = self.link.clean_signal(h_raw, x_grid, pc["tx_power"])
        noise_var = self.link.noise_variance(clean["y_clean"])

        if self.interference is not None:
            int_out = self.interference.sample(batch_size, noise_var)
            link = self.link.finalize(clean["y_clean"], noise_var, int_out["interference"], int_out["ruu_interference"])
        else:
            int_out = None
            link = self.link.finalize(clean["y_clean"], noise_var)

        cov = self.covariance_estimator(link["y"])
        ce_ls = self.channel_estimator_ls(link["y"])
        ce_lmmse = self.channel_estimator_lmmse(link["y"], link["noise_var"])

        batch = {
            "bits": tx["bits"],
            "x_data": tx["x_data"],
            "x_data_grid": tx["x_grid"],
            "x_dmrs": dmrs["x_dmrs"],
            "x_grid": x_grid,
            "h_propagation": h_propagation,
            "h_true": h_true,
            "h_hat_ls": ce_ls["h_hat"],
            "h_hat_lmmse": ce_lmmse["h_hat"],
            "h_hat_lmmse_err_var": ce_lmmse["err_var"],
            "tx_power": pc["tx_power"],
            "tx_power_db": pc["tx_power_db"],
            "y": link["y"],
            "y_clean": link["y_clean"],
            "noise_var": link["noise_var"],
            "ruu_true": link["ruu_true"],
            "ruu_hat": cov["ruu_hat"],
            "ruu_scm": cov["ruu_scm"],
            "ruu_shrinkage": cov["ruu_shrinkage"],
            "topology": topology,
            "metadata": {
                "channel_model": "uma",
                "num_ues": self.channel.num_ues,
                "num_streams": self.data_generator.num_streams,
                "num_bs_ant": self.channel.bs_array.num_ant,
                "num_ut_ant": self.channel.ut_array.num_ant,
                "bits_per_symbol": self.data_generator.num_bits_per_symbol,
                "stream_mapping": self.stream_mapper.mode,
                "power_control": self.power_controller.mode,
                "power_control_alpha": self.power_controller.alpha,
                "rx_snr_db": self.link.rx_snr_db,
                "covariance_mode": self.covariance_estimator.mode,
                "covariance_snapshots": cov["num_snapshots"],
                "channel_estimation": "ls_linear+sionna_lmmse_f_t",
            },
        }

        if int_out is not None:
            batch.update({
                "interference": int_out["interference"],
                "ruu_interference": int_out["ruu_interference"],
                "h_interference": int_out["h_interference"],
                "interference_tx_power": int_out["tx_power"],
                "target_inr": int_out["target_inr"],
            })

        return batch