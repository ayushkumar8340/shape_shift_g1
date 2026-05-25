import torch
from isaaclab.utils import configclass
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

class TargetPositionCommandCfg:
    class Ranges:
        pos_x = (1.5, 3.0)  
        pos_y = (-1.0, 1.0) 
    ranges = Ranges()
    resampling_time_range = (10.0, 15.0)

class TargetPositionCommand:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

    def resample(self, env_ids):
        current_pos = self.env.robot.data.root_pos_w[env_ids]
        
        rand_x = torch.rand(len(env_ids), device=self.device) * (self.cfg.ranges.pos_x[1] - self.cfg.ranges.pos_x[0]) + self.cfg.ranges.pos_x[0]
        rand_y = torch.rand(len(env_ids), device=self.device) * (self.cfg.ranges.pos_y[1] - self.cfg.ranges.pos_y[0]) + self.cfg.ranges.pos_y[0]
        
        self.target_pos_w[env_ids, 0] = current_pos[:, 0] + rand_x
        self.target_pos_w[env_ids, 1] = current_pos[:, 1] + rand_y
        self.target_pos_w[env_ids, 2] = 0.0

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.resample(env_ids)

    def compute(self, dt: float):
        """Called during env.step() - we don't need time-based updates for a static target."""
        pass

    @property
    def command(self):
        """Returns [dx_local, dy_local]. This goes straight into your actor observations automatically!"""
        root_pos = self.env.robot.data.root_pos_w
        root_quat = self.env.robot.data.root_quat_w
        
        rel_pos_w = self.target_pos_w - root_pos
        rel_pos_local = math_utils.quat_rotate_inverse(root_quat, rel_pos_w)
        
        return rel_pos_local[:, :2]

    def get_heading_error(self):
        """Used strictly for the face_target reward calculation."""
        rel_pos_local = self.command
        heading_err = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0])
        return heading_err.unsqueeze(-1)


class DribbleCommandCfg:
    pass

class DribbleCommand:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

    def reset(self, env_ids=None):
        pass

    def compute(self, dt: float):
        pass

    @property
    def command(self):
        """Returns [dx, dy, dz, vx, vy, vz] of the ball in the robot's local frame."""
        ball = self.env.scene["ball"]
        robot = self.env.scene["robot"]

        # 1. Relative Position
        rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w
        rel_pos_local = math_utils.quat_rotate_inverse(robot.data.root_quat_w, rel_pos_w)

        # 2. Relative Velocity
        rel_vel_w = ball.data.root_lin_vel_w
        rel_vel_local = math_utils.quat_rotate_inverse(robot.data.root_quat_w, rel_vel_w)

        # 3. Combine them into a 6D command vector
        return torch.cat([rel_pos_local, rel_vel_local], dim=-1)
