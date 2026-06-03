import torch
from envs.base.base_env import BaseEnv
from mdp.command_gen import LiftCommand, LiftCommandCfg

from utils.env_utils.scene import SceneCfg
from envs.g1.g1_lift_config import BASKETBALL_CFG

SceneCfg.ball = BASKETBALL_CFG


class G1LiftEnv(BaseEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)

    def _create_command_generator(self):
        self.command_cfg = LiftCommandCfg()
        self.command_generator = LiftCommand(self.command_cfg, self)

    def check_reset(self):
        reset_buf, time_out_buf = super().check_reset()

        # Early termination if the ball rolls out of reach
        ball_pos = self.scene["ball"].data.root_pos_w
        robot_pos = self.scene["robot"].data.root_pos_w
        dist = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=-1)
        lost_ball = dist > 0.8
        reset_buf |= lost_ball

        return reset_buf, time_out_buf
