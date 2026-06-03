import torch
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
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


class LiftCommandCfg:
    asset_name: str = "robot"
    hand_body_names = [".*left_rubber_hand.*", ".*right_rubber_hand.*"]
 
    low_h: float = 0.35        # crouch pelvis height
    high_h: float = 0.78       # standing pelvis height
    grasp_lo: float = 0.30     # below this grasp -> stay fully crouched
    grasp_hi: float = 0.60     # above this grasp -> fully stand
 
    radius: float = 0.125
    surface_std: float = 0.10
    midpoint_std: float = 0.08
    target_sep: float = 0.25
    sep_std: float = 0.08
 
    debug_vis: bool = False
 
 
class LiftCommand:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
 
        hand_cfg = SceneEntityCfg(cfg.asset_name, body_names=cfg.hand_body_names)
        hand_cfg.resolve(env.scene)
        self.hand_ids = hand_cfg.body_ids
        self._target_h = torch.full((self.num_envs,), cfg.low_h, device=self.device)
 
    def _grasp_quality(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        ball = self.env.scene["ball"]
        c = self.cfg
 
        hands = robot.data.body_pos_w[:, self.hand_ids, :]      # [N, 2, 3]
        ball_pos = ball.data.root_pos_w                         # [N, 3]
        ball_c = ball_pos.unsqueeze(1)
 
        surf_dist = torch.clamp(torch.norm(hands - ball_c, dim=-1) - c.radius, min=0.0)
        surface = torch.exp(-(surf_dist.mean(dim=1) ** 2) / (c.surface_std ** 2))
 
        midpoint = hands.mean(dim=1)
        mid_err = torch.norm(midpoint - ball_pos, dim=-1)
        centered = torch.exp(-(mid_err ** 2) / (c.midpoint_std ** 2))
 
        sep = torch.norm(hands[:, 0, :] - hands[:, 1, :], dim=-1)
        squeeze = torch.exp(-((sep - c.target_sep) ** 2) / (c.sep_std ** 2))
 
        return surface * centered * squeeze
 
    def reset(self, env_ids: torch.Tensor):
        if env_ids is not None and len(env_ids) > 0:
            self._target_h[env_ids] = self.cfg.low_h
 
    def compute(self, dt: float):
        c = self.cfg
        g = self._grasp_quality()
        g_eff = torch.clamp((g - c.grasp_lo) / (c.grasp_hi - c.grasp_lo), 0.0, 1.0)
        g_eff = g_eff * g_eff * (3.0 - 2.0 * g_eff)
        self._target_h = c.low_h + (c.high_h - c.low_h) * g_eff
 
    @property
    def command(self) -> torch.Tensor:
        """Returns [dx, dy, dz, target_base_height]."""
        ball = self.env.scene["ball"]
        robot = self.env.scene["robot"]
 
        rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w
        rel_pos_local = math_utils.quat_rotate_inverse(robot.data.root_quat_w, rel_pos_w)
 
        return torch.cat([rel_pos_local, self._target_h.unsqueeze(1)], dim=-1)
