# mdp/rewards_height.py
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

def track_base_height_exp(env, std: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward exp(- (z - z_cmd)^2 / std^2 ). Command is env.command_generator.command[:, 3]."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    z_cmd = env.command_generator.command[:, 3]
    err2 = (z - z_cmd) ** 2
    return torch.exp(-err2 / (std**2))

def base_height_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    z_cmd = env.command_generator.command[:, 3]
    return (z - z_cmd) ** 2