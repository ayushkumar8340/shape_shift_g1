from envs.base.base_env import BaseEnv
from mdp.command_gen import UniformVelHeightCommand, UniformVelHeightCommandCfg


class G1CrawlEnv(BaseEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)

    def _create_command_generator(self):
        if self.cfg.commands.ranges.base_height is None:
            raise RuntimeError("Crawl env requires cfg.commands.ranges.base_height=(min_h,max_h).")

        self.command_cfg = UniformVelHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )

        self.command_generator = UniformVelHeightCommand(self.command_cfg, self)