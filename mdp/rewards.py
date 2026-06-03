from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from envs.base.base_env import BaseEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def desired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    return torch.sum(is_contact.float(), dim=1)

def hands_to_ball_surface_tanh(env: BaseEnv, radius: float = 0.125, std: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Rewards reaching the surface of the ball, not the center."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    
    wrist_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    ball_pos = ball.data.root_pos_w.unsqueeze(1)
    
    # Distance to center
    dist_to_center = torch.norm(wrist_pos - ball_pos, dim=-1)
    
    # Distance to surface (clamps at 0 so it doesn't get bonus points for penetrating the ball)
    dist_to_surface = torch.clamp(dist_to_center - radius, min=0.0)
    
    # Average the distance of both hands to the surface
    mean_dist = dist_to_surface.mean(dim=1)
    
    return 1.0 - torch.tanh(mean_dist / std)

def ball_rolled_away(env: BaseEnv, max_distance: float = 0.8) -> torch.Tensor:
    """Terminates the episode if the ball rolls out of the robot's reachable workspace."""
    ball_pos = env.scene["ball"].data.root_pos_w
    robot_pos = env.scene["robot"].data.root_pos_w
    
    # 2D distance between robot base and the ball
    dist = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=-1)
    
    # Returns True if the ball is knocked further than max_distance
    return dist > max_distance

def penalize_walking(env: BaseEnv) -> torch.Tensor:
    """Punishes the robot for moving its base horizontally. Forces it to plant its feet."""
    lin_vel = env.scene["robot"].data.root_lin_vel_w[:, :2]
    return torch.sum(torch.square(lin_vel), dim=-1)

def penalize_hand_kickstand(env: BaseEnv, min_height: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes using hands to hold body weight on the floor. Hands should grasp the ball (~0.12m high), not the ground."""
    robot = env.scene["robot"]
    hand_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2] 
    # The ball is 0.125m radius. If hands hit 0.06m or lower, they missed the ball and hit the floor.
    penalty = torch.where(hand_z < min_height, min_height - hand_z, 0.0)
    return torch.sum(penalty, dim=1)

def penalize_lifted_feet(env: BaseEnv, max_height: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Punishes the robot if either foot leaves the floor to prevent the one-legged 'golfer's lift'."""
    robot = env.scene["robot"]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    penalty = torch.where(foot_z > max_height, foot_z - max_height, 0.0)
    return torch.sum(penalty, dim=1)

def _ball_grasp_quality(
    env: "BaseEnv",
    body_ids,
    radius: float = 0.125,
    surface_std: float = 0.10,
    midpoint_std: float = 0.08,
    target_sep: float = 0.23,
    sep_std: float = 0.08,
) -> torch.Tensor:
    """Continuous [0,1] quality of a TRUE bimanual sandwich grasp.
 
    Product of three smooth Gaussians:
      surface  -> both hands on the ball surface
      centered -> midpoint of the hands == ball center (forces OPPOSITE sides;
                  this is the anti-fake-grip term that rules out both hands /
                  backs-of-hands on the near side)
      squeeze  -> hands ~one diameter apart
    """
    robot = env.scene["robot"]
    ball = env.scene["ball"]
 
    hands = robot.data.body_pos_w[:, body_ids, :]           # [N, 2, 3]
    ball_pos = ball.data.root_pos_w                          # [N, 3]
    ball_c = ball_pos.unsqueeze(1)
 
    surf_dist = torch.clamp(torch.norm(hands - ball_c, dim=-1) - radius, min=0.0)
    surface = torch.exp(-(surf_dist.mean(dim=1) ** 2) / (surface_std ** 2))
 
    midpoint = hands.mean(dim=1)
    mid_err = torch.norm(midpoint - ball_pos, dim=-1)
    centered = torch.exp(-(mid_err ** 2) / (midpoint_std ** 2))
 
    sep = torch.norm(hands[:, 0, :] - hands[:, 1, :], dim=-1)
    squeeze = torch.exp(-((sep - target_sep) ** 2) / (sep_std ** 2))
 
    return surface * centered * squeeze
 
 
def grasp_quality_reward(
    env: "BaseEnv", asset_cfg: SceneEntityCfg, radius: float = 0.125
) -> torch.Tensor:
    """Smooth bootstrap reward: learn a real sandwich grasp."""
    return _ball_grasp_quality(env, asset_cfg.body_ids, radius)
 
 
# def ball_height_reward(
#     env: "BaseEnv",
#     asset_cfg: SceneEntityCfg,
#     radius: float = 0.125,
#     target_h: float = 0.575,
# ) -> torch.Tensor:
#     """Direct reward for lifting the BALL, gated by a real grasp.
#     """
#     ball = env.scene["ball"]
#     grip = _ball_grasp_quality(env, asset_cfg.body_ids, radius)
#     ball_h = torch.clamp(ball.data.root_pos_w[:, 2] - radius, min=0.0)
#     prog = torch.clamp(ball_h / target_h, 0.0, 1.0)
#     return grip * prog

def ball_height_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg,
    radius: float = 0.125,
    target_h: float = 0.575,
    body_low: float = 0.35,
    body_high: float = 0.78,
) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    grip = _ball_grasp_quality(env, asset_cfg.body_ids, radius)
    ball_h = torch.clamp(ball.data.root_pos_w[:, 2] - radius, min=0.0)
    ball_prog = torch.clamp(ball_h / target_h, 0.0, 1.0)
    root_h = robot.data.root_pos_w[:, 2]
    body_prog = torch.clamp((root_h - body_low) / (body_high - body_low), 0.0, 1.0)
    return grip * ball_prog * body_prog

def bimanual_sphere_grasp(env: BaseEnv, radius: float = 0.125, std: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Rewards a perfect bimanual sandwich grasp: hands diametrically opposed with the ball in the middle."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    
    # Extract left and right hands
    left_hand = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    right_hand = robot.data.body_pos_w[:, asset_cfg.body_ids[1], :]
    ball_pos = ball.data.root_pos_w
    
    # The point exactly halfway between the hands should be inside the center of the ball
    midpoint = (left_hand + right_hand) / 2.0
    midpoint_error = torch.norm(midpoint - ball_pos, dim=-1)
    
    # The hands should be separated by exactly the diameter, minus 2cm (0.02m) to ensure firm contact
    diameter = radius * 2.0
    target_separation = diameter - 0.02
    current_separation = torch.norm(left_hand - right_hand, dim=-1)
    
    # How far off are the hands from the perfect grip width?
    squeeze_error = torch.abs(current_separation - target_separation)
    
    # Combine the errors. Both must be 0.0 for a perfect score.
    total_error = midpoint_error + squeeze_error
    
    # Use a tighter std (0.1) so it only rewards high-precision alignments
    return 1.0 - torch.tanh(total_error / std)

def lift_ball_reward(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards lifting the ball ONLY if the hands are actively clamping it."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    
    wrist_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    ball_pos = ball.data.root_pos_w.unsqueeze(1)
    
    # Surface Distance Check: Are wrists close to the outside of the ball?
    dist_to_center = torch.norm(wrist_pos - ball_pos, dim=-1)
    dist_to_surface = torch.clamp(dist_to_center - 0.125, min=0.0)
    is_close = dist_to_surface.mean(dim=1) < 0.03
    
    # Bimanual Squeeze Check: Are the hands separated by roughly the ball's diameter (0.25m)?
    left_hand = wrist_pos[:, 0, :]
    right_hand = wrist_pos[:, 1, :]
    hand_separation = torch.norm(left_hand - right_hand, dim=-1)
    is_squeezing = (hand_separation > 0.22) & (hand_separation < 0.28)
    
    # Secure Grasp Flag
    is_held = (is_close & is_squeezing).float()
    
    lift_height = ball.data.root_pos_w[:, 2] - 0.125    # Subtract ball radius
    lift_height = torch.clamp(lift_height, min=0.0, max=0.575)
    lift_reward = torch.where(lift_height > 0.0, torch.exp(lift_height)-1.0, 0.0)

    return torch.where(is_held.bool(), lift_reward, lift_reward*1e-4)


def stand_straight_with_ball(env: BaseEnv, asset_cfg: SceneEntityCfg, stance_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards the robot for raising its Center of Mass, ONLY when the ball is secured."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    wrist_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    ball_pos = ball.data.root_pos_w.unsqueeze(1)

    # Exact same secure grasp logic as above
    dist_to_center = torch.norm(wrist_pos - ball_pos, dim=-1)
    dist_to_surface = torch.clamp(dist_to_center - 0.125, min=0.0)
    is_close = dist_to_surface.mean(dim=1) < 0.03

    left_hand = wrist_pos[:, 0, :]
    right_hand = wrist_pos[:, 1, :]
    hand_separation = torch.norm(left_hand - right_hand, dim=-1)
    is_squeezing = (hand_separation > 0.22) & (hand_separation < 0.28)

    is_held = (is_close & is_squeezing).float()

    # Posture Check
    root_height = torch.clamp(robot.data.root_pos_w[:, 2], max=0.8)
    root_height_reward = torch.where(root_height > 0.60, torch.exp(root_height)-1.0, 0.0)

    # Joint Stance Check (Your idea implemented as a reward!)
    current_pos = robot.data.joint_pos[:, stance_cfg.joint_ids]
    default_pos = robot.data.default_joint_pos[:, stance_cfg.joint_ids]

    # Calculate error from default standing pose. Tanh provides a smooth, continuous pull upwards.
    stance_error = torch.norm(current_pos - default_pos, dim=-1)
    stance_reward = 1.0 - torch.tanh(stance_error / 2.0)

    return torch.where(is_held.bool(), (stance_reward + root_height_reward), (stance_reward + root_height_reward)*1e-4)
