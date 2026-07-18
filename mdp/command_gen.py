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


@configclass
class WalkDribbleCommandCfg:
    lin_vel_x: tuple = (0.0, 0.8)
    lin_vel_y: tuple = (-0.2, 0.2)
    ang_vel_z: tuple = (-0.5, 0.5)
    min_moving_speed: float = 0.3
    resampling_time_range: tuple = (5.0, 8.0)

    # Curriculum with WARMUP HOLD: level stays 0 (=> all envs standing, zero commands)
    # for warmup_steps policy steps so the dribble locks in first, then ramps linearly
    # to 1 over ramp_steps. 60000 steps ~= 2500 iters @ 24 steps/iter.
    standing_frac_start: float = 1.0
    standing_frac_end: float = 0.25    # <-- 1.0 = pure static dribble (golden state)
    warmup_steps: int = 60000
    ramp_steps: int = 96000
    # Fresh training: -1.0 = warmup curriculum active (dribble first, then walking ramp. Resume: 1.0.
    fixed_level: float = -1.0
    standing_speed_eps: float = 0.15   # |cmd| below this counts as "standing" for reward gating
    gait_freq: float = 1.2   # full gait cycles per second
    gait_duty: float = 0.55  # stance fraction per foot


class WalkDribbleCommand:
    """Velocity command + ball state + gait clock for the dribble->walk task.

    Observation command (11D, robot frame):
        [0:3] ball dx,dy,dz   [3:6] ball vx,vy,vz   [6:9] commanded vx,vy,yaw_rate
        [9:11] gait clock sin(2*pi*phi), cos(2*pi*phi)

    Each env is assigned a regime ONCE per episode via the standing fraction
    (warmup-hold curriculum). Standing envs get a zero command (pure dribble in
    place, the proven golden behavior); movers get a floored forward command
    (>= min_moving_speed). `is_standing` and `expected_stance` are exposed for
    the reward functions.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._forced_standing = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.gait_phase = torch.rand(self.num_envs, device=self.device)
        self._time_left = torch.zeros(self.num_envs, device=self.device)
        self._steps = 0

    @property
    def level(self) -> float:
        if self.cfg.fixed_level >= 0.0:
            return float(self.cfg.fixed_level)
        return min(1.0, max(0.0, (self._steps - self.cfg.warmup_steps) / max(1, self.cfg.ramp_steps)))

    def _sample_resample_time(self, n: int):
        lo, hi = self.cfg.resampling_time_range
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def _uniform(self, n: int, lo: float, hi: float):
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def resample(self, env_ids: torch.Tensor, redraw_regime: bool = True):
        n = env_ids.numel()
        L = self.level
        r = self.cfg

        if redraw_regime:
            standing_frac = r.standing_frac_start + (r.standing_frac_end - r.standing_frac_start) * L
            self._forced_standing[env_ids] = torch.rand(n, device=self.device) < standing_frac

        forced_standing = self._forced_standing[env_ids]

        vx = self._uniform(n, r.min_moving_speed, r.min_moving_speed + (r.lin_vel_x[1] - r.min_moving_speed) * L)
        vy = self._uniform(n, r.lin_vel_y[0] * L, r.lin_vel_y[1] * L)
        yaw = self._uniform(n, r.ang_vel_z[0] * L, r.ang_vel_z[1] * L)

        vx[forced_standing] = 0.0
        vy[forced_standing] = 0.0
        yaw[forced_standing] = 0.0

        self.vel_command[env_ids, 0] = vx
        self.vel_command[env_ids, 1] = vy
        self.vel_command[env_ids, 2] = yaw

        cmd_mag = torch.norm(self.vel_command[env_ids], dim=-1)
        self.is_standing[env_ids] = forced_standing | (cmd_mag < r.standing_speed_eps)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        self.resample(env_ids, redraw_regime=True)
        self._time_left[env_ids] = self._sample_resample_time(env_ids.numel())
        self.gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)

    def compute(self, dt: float):
        self._steps += 1
        self.gait_phase = (self.gait_phase + dt * self.cfg.gait_freq) % 1.0
        self._time_left -= dt
        env_ids = torch.nonzero(self._time_left <= 0.0, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self.resample(env_ids, redraw_regime=False)
            self._time_left[env_ids] = self._sample_resample_time(env_ids.numel())

    @property
    def expected_stance(self) -> torch.Tensor:
        duty = self.cfg.gait_duty
        left = torch.frac(self.gait_phase) < duty
        right = torch.frac(self.gait_phase + 0.5) < duty
        return torch.stack([left, right], dim=-1)

    @property
    def command(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        ball = self.env.scene["ball"]

        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        rel_ball_pos = math_utils.quat_rotate_inverse(root_quat, ball.data.root_pos_w - root_pos)
        rel_ball_vel = math_utils.quat_rotate_inverse(root_quat, ball.data.root_lin_vel_w)
        two_pi_phi = 2.0 * math.pi * self.gait_phase
        return torch.cat([rel_ball_pos, rel_ball_vel, self.vel_command, torch.sin(two_pi_phi).unsqueeze(-1), torch.cos(two_pi_phi).unsqueeze(-1)], dim=-1)
