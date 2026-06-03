import torch
from envs.base.base_env import BaseEnv
from mdp.command_gen import WalkDribbleCommand, WalkDribbleCommandCfg


class G1WalkDribbleEnv(BaseEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)

    def _create_command_generator(self):
        self.command_cfg = WalkDribbleCommandCfg()
        self.command_generator = WalkDribbleCommand(self.command_cfg, self)

    def check_reset(self):
        reset_buf, time_out_buf = super().check_reset()

        ball_pos = self.scene["ball"].data.root_pos_w
        robot_pos = self.scene["robot"].data.root_pos_w
        dist_xy = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=-1)

        lost_ball = dist_xy > 1.2

        reset_buf |= lost_ball
        return reset_buf, time_out_buf
