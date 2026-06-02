import torch
from isaaclab.utils import configclass
import math
from dataclasses import MISSING

@configclass
class UniformVelHeightCommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.57, 1.57)
    heading: tuple = (-math.pi, math.pi)
    base_height: tuple = (0.65, 0.95)  # meters (required, not optional)

@configclass
class UniformVelHeightCommandCfg:
    asset_name: str = "robot"
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = False

    ranges: UniformVelHeightCommandRangesCfg = UniformVelHeightCommandRangesCfg()

class UniformVelHeightCommand:
    """Command: [vx, vy, yaw_rate, base_height_m]. Uses your existing CommandsCfg structure."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self.command = torch.zeros(self.num_envs, 4, device=self.device)
        self._time_left = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self._resample(env_ids)
        self._time_left[env_ids] = self._sample_resample_time(env_ids.numel())

    def compute(self, dt: float):
        self._time_left -= dt
        env_ids = torch.nonzero(self._time_left <= 0.0, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self._resample(env_ids)
            self._time_left[env_ids] = self._sample_resample_time(env_ids.numel())

    def _sample_resample_time(self, n: int):
        lo, hi = self.cfg.resampling_time_range
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def _uniform(self, n: int, lo: float, hi: float):
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def _resample(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        r = self.cfg.ranges

        if r.base_height is None:
            raise RuntimeError("UniformVelHeightCommand requires cfg.ranges.base_height=(min_h,max_h).")

        self.command[env_ids, 0] = self._uniform(n, *r.lin_vel_x)
        self.command[env_ids, 1] = self._uniform(n, *r.lin_vel_y)
        self.command[env_ids, 2] = self._uniform(n, *r.ang_vel_z)
        self.command[env_ids, 3] = self._uniform(n, *r.base_height)

        # standing envs: zero motion, keep height sampled
        if self.cfg.rel_standing_envs > 0.0:
            mask = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
            self.command[env_ids[mask], 0:3] = 0.0

