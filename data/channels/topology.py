import math
import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import gen_single_sector_topology


def _sample_velocities(in_state, outdoor_max_kmh, indoor_max_kmh, precision, device):
    dtype = torch.float32 if precision == "single" else torch.float64
    rng = sionna_config.torch_rng(device)
    outdoor_max_ms, indoor_max_ms = outdoor_max_kmh / 3.6, indoor_max_kmh / 3.6
    max_speed = outdoor_max_ms + in_state.to(dtype) * (indoor_max_ms - outdoor_max_ms)
    speed = torch.rand(in_state.shape, dtype=dtype, device=device, generator=rng) * max_speed
    angle = 2.0 * math.pi * torch.rand(in_state.shape, dtype=dtype, device=device, generator=rng)
    vx, vy = speed * torch.cos(angle), speed * torch.sin(angle)
    return torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1)


def generate_uma_topology(batch_size, num_ues, cfg, device):
    tcfg, precision = cfg["topology"], cfg["general"]["precision"]
    kwargs = dict(batch_size=batch_size, num_ut=num_ues, scenario="uma",
                  indoor_probability=float(tcfg["indoor_probability"]),
                  min_ut_velocity=0.0, max_ut_velocity=0.0,
                  precision=precision, device=device)
    ut_loc, bs_loc, ut_orientations, bs_orientations, _, in_state = gen_single_sector_topology(**kwargs)
    ut_velocities = _sample_velocities(in_state, float(tcfg["outdoor_max_speed_kmh"]),
                                      float(tcfg["indoor_max_speed_kmh"]), precision, device)
    return {
        "ut_loc": ut_loc,
        "bs_loc": bs_loc,
        "ut_orientations": ut_orientations,
        "bs_orientations": bs_orientations,
        "ut_velocities": ut_velocities,
        "in_state": in_state,
    }