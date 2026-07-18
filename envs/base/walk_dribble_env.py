import torch

from envs.base.base_env import BaseEnv
from mdp.command_gen import WalkDribbleCommand, WalkDribbleCommandCfg


class G1WalkDribbleEnv(BaseEnv):
    """Dribble -> Walk task.

    Built on the proven dribble env (which registers the ball, resets it under the
    hand, and is the basis for stand-and-dribble). Adds a velocity command with a
    standing-fraction curriculum, and detects falls via base orientation + height
    instead of body contact forces (so the basketball touching the torso/pelvis can
    never spuriously terminate an upright robot).

    With the cfg's standing_frac_end = 1.0 this is the PROVEN static-dribble state
    (every env stands & dribbles in place); set standing_frac_end < 1.0 to ramp walking.
    """

    def _create_command_generator(self):
        self.command_cfg = getattr(self.cfg, "walk_dribble_command", WalkDribbleCommandCfg())
        self.command_generator = WalkDribbleCommand(self.command_cfg, self)

    def step(self, actions):
        obs, rew, reset_buf, extras = super().step(actions)
        # Debug visualization in the viewer (play): GREEN = commanded velocity,
        # BLUE = actual base velocity. draw_interface only exists when not headless
        # and cfg.commands.debug_vis is on (BaseEnv.__init__).
        if not self.headless and hasattr(self, "draw_interface"):
            self._draw_debug_vis()
        return obs, rew, reset_buf, extras

    def _draw_debug_vis(self):
        """GREEN arrow = commanded velocity, BLUE arrow = actual base velocity —
        proper 3D arrow markers (like the official Isaac Lab velocity tasks),
        floating above the robot, length scaled by speed. Falls back to plain
        debug lines if the marker system is unavailable."""
        robot = self.scene["robot"]
        q = robot.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cmd = self.command_generator.vel_command
        # command is in the robot yaw frame -> world heading & magnitude
        cmd_heading = yaw + torch.atan2(cmd[:, 1], cmd[:, 0])
        cmd_speed = torch.norm(cmd[:, :2], dim=-1)
        act_vel = robot.data.root_lin_vel_w[:, :2]
        act_heading = torch.atan2(act_vel[:, 1], act_vel[:, 0])
        act_speed = torch.norm(act_vel, dim=-1)

        pos = robot.data.root_pos_w.clone()
        pos[:, 2] += 0.6

        try:
            if not hasattr(self, "_vel_markers_ready"):
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
                cmd_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/cmd_velocity")
                act_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/actual_velocity")
                self._cmd_marker = VisualizationMarkers(cmd_cfg)
                self._act_marker = VisualizationMarkers(act_cfg)
                self._arrow_base_scale = torch.tensor(
                    next(iter(cmd_cfg.markers.values())).scale,
                    device=self.device, dtype=torch.float)
                self._vel_markers_ready = True

            def quat_scale(heading, speed):
                zeros = torch.zeros_like(heading)
                quat = torch.stack(
                    [torch.cos(heading / 2), zeros, zeros, torch.sin(heading / 2)], dim=-1)
                scale = self._arrow_base_scale.repeat(speed.shape[0], 1).clone()
                scale[:, 0] = scale[:, 0] * speed * 3.0
                scale = scale * (speed > 0.05).float().unsqueeze(-1)
                return quat, scale

            cq, cs = quat_scale(cmd_heading, cmd_speed)
            aq, asc = quat_scale(act_heading, act_speed)
            act_pos = pos.clone()
            act_pos[:, 2] += 0.15  # stack the blue arrow slightly above the green
            self._cmd_marker.visualize(translations=pos, orientations=cq, scales=cs)
            self._act_marker.visualize(translations=act_pos, orientations=aq, scales=asc)
        except Exception:
            # fallback: plain lines (old behavior)
            self.draw_interface.clear_lines()
            cmd_end = pos.clone()
            cmd_end[:, 0] += cmd_speed * torch.cos(cmd_heading)
            cmd_end[:, 1] += cmd_speed * torch.sin(cmd_heading)
            vel_end = pos.clone()
            vel_end[:, :2] += act_vel
            n = self.num_envs
            starts = pos.tolist()
            self.draw_interface.draw_lines(starts, cmd_end.tolist(), [(0.0, 1.0, 0.0, 1.0)] * n, [3.0] * n)
            self.draw_interface.draw_lines(starts, vel_end.tolist(), [(0.2, 0.4, 1.0, 1.0)] * n, [3.0] * n)

    def check_reset(self):
        robot = self.scene["robot"]

        time_out_buf = self.episode_length_buf >= self.max_episode_length

        # Fall detection via base orientation + height (NOT contact forces, so the
        # ball touching the body can't spuriously terminate). tilt = sin(angle from
        # vertical): 0.5 ~= 30 deg.
        tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
        fell_over = tilt > 0.5
        base_height = robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fell_low = base_height < 0.40
        reset_buf = fell_over | fell_low | time_out_buf

        # Lost ball: rolled / bounced too far from the robot. Movers get extra
        # headroom because the pocket now LEADS the robot by up to 15cm.
        ball_pos = self.scene["ball"].data.root_pos_w
        dist_xy = torch.norm(ball_pos[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
        lost_thresh = torch.where(
            self.command_generator.is_standing,
            torch.tensor(1.0, device=self.device),
            torch.tensor(1.2, device=self.device),
        )
        reset_buf |= dist_xy > lost_thresh

        return reset_buf, time_out_buf
