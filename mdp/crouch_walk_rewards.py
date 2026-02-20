import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

def track_base_height_exp(env, std: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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

def feet_flat_orientation(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity_vec = asset.data.GRAVITY_VEC_W
    num_feet = len(asset_cfg.body_ids)
    gravity_expanded = gravity_vec.unsqueeze(1).repeat(1, num_feet, 1)
    projected_gravity_foot = math_utils.quat_apply_inverse(foot_quat, gravity_expanded)
    
    return torch.sum(torch.square(projected_gravity_foot[:, :, :2]), dim=(1, 2))