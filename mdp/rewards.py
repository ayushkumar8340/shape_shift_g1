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
    # Reward stepping only if we are far from the target
    dist = torch.norm(env.command_generator.command, dim=-1)
    reward *= (dist > 0.5)
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

def track_position_exp(env: BaseEnv, sigma: float = 1.0) -> torch.Tensor:
    rel_pos = env.command_generator.command
    dist = torch.norm(rel_pos, dim=-1)
    dist_outside = torch.clamp(dist - 0.5, min=0.0)
    return torch.exp(-torch.square(dist_outside) / sigma)

def stop_at_target_exp(env: BaseEnv, dist_threshold: float = 0.5, sigma: float = 0.2) -> torch.Tensor:
    rel_pos = env.command_generator.command
    dist = torch.norm(rel_pos, dim=-1)
    lin_vel = env.robot.data.root_lin_vel_w
    vel_sq = torch.sum(lin_vel[:, :2] ** 2, dim=-1)
    return (dist < dist_threshold) * torch.exp(-vel_sq / sigma)

def progress_towards_target(env: BaseEnv) -> torch.Tensor:
    rel_pos_w = env.command_generator.target_pos_w - env.robot.data.root_pos_w
    dist = torch.norm(rel_pos_w[:, :2], dim=-1)
    target_dir_w = rel_pos_w[:, :2] / (dist.unsqueeze(-1) + 1e-5)
    lin_vel_w = env.robot.data.root_lin_vel_w[:, :2]
    vel_towards_target = torch.sum(lin_vel_w * target_dir_w, dim=-1)
    vel_towards_target = torch.clamp(vel_towards_target, min=-2.0, max=1.0)
    return vel_towards_target * (dist > 0.5).float()

def face_target_exp(env: BaseEnv, sigma: float = 0.5) -> torch.Tensor:
    dist = torch.norm(env.command_generator.command, dim=-1)
    heading_error = env.command_generator.get_heading_error().squeeze(-1)
    is_far = (dist > 0.5).float()
    return torch.exp(-torch.square(heading_error) / sigma) * is_far

def penalize_wobble_at_target(env: BaseEnv, dist_threshold: float = 0.5) -> torch.Tensor:
    dist = torch.norm(env.command_generator.command, dim=-1)
    joint_vel_sq = torch.sum(torch.square(env.robot.data.joint_vel), dim=-1)
    return (dist < dist_threshold).float() * joint_vel_sq

def ball_under_hand_xy_tanh(env: BaseEnv, std: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Rewards keeping the right wrist directly above the ball in the X/Y plane."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    # Extract only X and Y coordinates
    wrist_pos_xy = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    ball_pos_xy = ball.data.root_pos_w[:, :2]

    dist_xy = torch.norm(wrist_pos_xy - ball_pos_xy, dim=-1)
    return 1.0 - torch.tanh(dist_xy / std)

def ball_bounce_activity(env: BaseEnv, max_reward_vel: float = 3.0) -> torch.Tensor:
    """Rewards the ball for having vertical kinetic energy (moving up or down), capped at a natural speed."""
    ball = env.scene["ball"]
    vel_z = torch.abs(ball.data.root_lin_vel_w[:, 2])
    # Cap the reward so hitting it harder doesn't yield infinite points
    return torch.clamp(vel_z, max=max_reward_vel)

def dribble_strike(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Rewards the hand for moving downwards while physically touching the ball."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    # Track the right hand specifically
    hand_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    hand_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    ball_pos = ball.data.root_pos_w

    dist = torch.norm(hand_pos - ball_pos, dim=-1) - 0.125   # Subtract ball radius since the ball_pos is with respect to the CoM of the ball
    
    # Ball radius is 0.125m. Contact occurs when distance is ~0.13m to 0.15m.
    is_touching = (dist < 0.10).float()     # 0.12

    # Get downward velocity of the hand (caps at 2.0 m/s so it doesn't over-learn)
    push_down_vel = torch.clamp(-hand_vel[:, 2], min=0.0, max=2.0)

    # Only rewards if touching AND pushing down. Prevents "pinning" the ball to the floor.
    return is_touching * push_down_vel

def penalize_pinning(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Punishes the robot for resting its hand on the ball without letting it bounce."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    
    hand_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    dist = torch.norm(hand_pos - ball.data.root_pos_w, dim=-1) - 0.125
    is_touching = (dist < 0.10).float()
    
    # If the hand is touching the ball, but the ball has almost ZERO vertical velocity, it is pinning it!
    ball_vel_z = torch.abs(ball.data.root_lin_vel_w[:, 2])
    is_pinned = (ball_vel_z < 0.5).float()
    
    return is_touching * is_pinned

def wild_dribbling(env: BaseEnv, speed_limit: float = 4.0) -> torch.Tensor:
    """Punishes the robot if it spikes the ball at unnatural speeds."""
    ball = env.scene["ball"]
    vel_z = torch.abs(ball.data.root_lin_vel_w[:, 2])
    # Returns 0 if under limit, scales up linearly if exceeded
    return torch.where(vel_z > speed_limit, vel_z - speed_limit, 0.0)

def ball_xy_drift(env: BaseEnv) -> torch.Tensor:
    """Punishes the ball for moving horizontally (rolling or bouncing away)."""
    ball = env.scene["ball"]
    vel_xy = ball.data.root_lin_vel_w[:, :2]
    return torch.sum(torch.square(vel_xy), dim=-1)

def penalize_lateral_lean(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mathematically forces the spine to remain perfectly vertical by keeping the head over the pelvis."""
    robot = env.scene["robot"]
    
    # Requires asset_cfg to pass head (index 0) and pelvis (index 1)
    head_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    pelvis_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[1], :]
    
    # Calculate the 2D drift in the X/Y plane. If the head drifts sideways from the pelvis, it is leaning!
    xy_drift = torch.norm(head_pos[:, :2] - pelvis_pos[:, :2], dim=-1)
    return xy_drift

def penalize_lifted_feet(env: BaseEnv, max_height: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene["robot"]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    penalty = torch.where(foot_z > max_height, foot_z - max_height, 0.0)
    return torch.sum(penalty, dim=1)
