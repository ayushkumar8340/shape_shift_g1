import torch
from envs.base.base_env import BaseEnv
from mdp.command_gen import DribbleCommand, DribbleCommandCfg
from utils.env_utils.scene import SceneCfg
from envs.g1.g1_dribble_config import BASKETBALL_CFG


class G1DribbleEnv(BaseEnv):
    def __init__(self, cfg, headless):
        SceneCfg.ball = BASKETBALL_CFG
        super().__init__(cfg, headless)

    def _create_command_generator(self):
        self.command_cfg = DribbleCommandCfg()
        self.command_generator = DribbleCommand(self.command_cfg, self)

    def check_reset(self):
        reset_buf, time_out_buf = super().check_reset()

        ball_pos = self.scene["ball"].data.root_pos_w
        robot_pos = self.scene["robot"].data.root_pos_w

        # If the ball rolls more than 0.8 meters away, kill the episode
        dist_xy = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=-1)
        lost_ball = dist_xy > 0.8

        reset_buf |= lost_ball
        return reset_buf, time_out_buf
