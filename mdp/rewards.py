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
    vel_yaw = math_utils.quat_apply_inverse(
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


def track_crawl_lin_vel_xy_exp(
    env,
    std: float,
    forward_axis: int = 2,
    forward_sign: float = 1.0,
    lateral_axis: int = 1,
    lateral_sign: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*torso.*"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat = asset.data.body_quat_w[:, body_cfg.body_ids[0], :]

    # Local forward axis in crawl body frame.
    local_forward = torch.zeros((env.num_envs, 3), device=env.device)
    local_forward[:, forward_axis] = forward_sign

    # Local lateral axis in crawl body frame.
    local_lateral = torch.zeros((env.num_envs, 3), device=env.device)
    local_lateral[:, lateral_axis] = lateral_sign

    # Convert crawl frame axes to world.
    forward_w = math_utils.quat_apply(body_quat, local_forward)
    lateral_w = math_utils.quat_apply(body_quat, local_lateral)

    # Project both onto horizontal ground plane.
    forward_w[:, 2] = 0.0
    lateral_w[:, 2] = 0.0

    forward_w = forward_w / torch.norm(forward_w, dim=1, keepdim=True).clamp(min=1e-6)

    # Make lateral exactly perpendicular to forward on the ground.
    # This avoids weird non-orthogonal body axes after pitch/roll.
    lateral_w = lateral_w - torch.sum(lateral_w * forward_w, dim=1, keepdim=True) * forward_w
    lateral_w = lateral_w / torch.norm(lateral_w, dim=1, keepdim=True).clamp(min=1e-6)

    root_vel_w = asset.data.root_lin_vel_w[:, :3]

    actual_vx = torch.sum(root_vel_w * forward_w, dim=1)
    actual_vy = torch.sum(root_vel_w * lateral_w, dim=1)

    target_vx = env.command_generator.command[:, 0]
    target_vy = env.command_generator.command[:, 1]

    lin_vel_error = torch.square(target_vx - actual_vx) + torch.square(target_vy - actual_vy)

    return torch.exp(-lin_vel_error / std**2)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
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


def forward_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    disp_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    cmd_xy = env.command_generator.command[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True).clamp(min=1e-6)
    cmd_dir = cmd_xy / cmd_norm

    progress = torch.sum(disp_xy * cmd_dir, dim=1)

    progress *= torch.norm(cmd_xy, dim=1) > 0.1
    return progress


def alternating_knee_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    left_contact = contact[:, 0]
    right_contact = contact[:, 1]

    # reward exactly one knee in contact
    return torch.logical_xor(left_contact, right_contact).float()

def hand_force_support(
    env,
    sensor_cfg: SceneEntityCfg,
    min_force: float = 20.0,
    max_force: float = 120.0,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    fz = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])

    # reward hand vertical force in a useful range, but do not reward huge impacts
    clipped = torch.clamp(fz, min=0.0, max=max_force)

    # only count force above min_force
    useful = torch.clamp(clipped - min_force, min=0.0)

    return torch.sum(useful, dim=1)

def knee_x_separation(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    knee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    left_x = knee_pos[:, 0, 0]
    right_x = knee_pos[:, 1, 0]

    return torch.abs(left_x - right_x)

def alternating_knee_gait_phase(
    env,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.8,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    left_contact = contact[:, 0]
    right_contact = contact[:, 1]

    t = env.episode_length_buf.float() * env.step_dt
    phase = torch.remainder(t, period) / period

    # first half: left knee should be contact, right knee swing
    left_support_phase = phase < 0.5

    reward_left_support = left_contact.float() + (~right_contact).float()
    reward_right_support = right_contact.float() + (~left_contact).float()

    reward = torch.where(
        left_support_phase,
        reward_left_support,
        reward_right_support,
    )

    # normalize to [0, 1]
    return reward * 0.5

def desired_contacts_binary(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward if all selected bodies are in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    return torch.all(contact, dim=1).float()


def crawl_diagonal_gait_strict(
    env,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.8,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    # body order:
    # [left_hand, right_hand, left_knee, right_knee]
    left_hand = contact[:, 0]
    right_hand = contact[:, 1]
    left_knee = contact[:, 2]
    right_knee = contact[:, 3]

    t = env.episode_length_buf.float() * env.step_dt
    phase = torch.remainder(t, period) / period

    phase_a = phase < 0.5

    # Phase A: LH + RK should support, RH + LK should swing
    phase_a_correct = (
        left_hand
        & right_knee
        & (~right_hand)
        & (~left_knee)
    )

    # Phase B: RH + LK should support, LH + RK should swing
    phase_b_correct = (
        right_hand
        & left_knee
        & (~left_hand)
        & (~right_knee)
    )

    return torch.where(phase_a, phase_a_correct, phase_b_correct).float()


def crawl_wrong_contact_count(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    target_contacts: float = 2.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    num_contacts = torch.sum(contact.float(), dim=1)

    return torch.square(num_contacts - target_contacts)

def crawl_fourbeat_swing_drag_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    period: float = 1.0,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    # order: [LH, RH, LK, RK]
    lh = contact[:, 0]
    rh = contact[:, 1]
    lk = contact[:, 2]
    rk = contact[:, 3]

    t = env.episode_length_buf.float() * env.step_dt
    phase = torch.remainder(t, period) / period

    q1 = phase < 0.25
    q2 = (phase >= 0.25) & (phase < 0.50)
    q3 = (phase >= 0.50) & (phase < 0.75)
    q4 = phase >= 0.75

    # phase 1 swing: LH
    # phase 2 swing: RK
    # phase 3 swing: RH
    # phase 4 swing: LK
    penalty = torch.zeros_like(phase)
    penalty = torch.where(q1, lh.float(), penalty)
    penalty = torch.where(q2, rk.float(), penalty)
    penalty = torch.where(q3, rh.float(), penalty)
    penalty = torch.where(q4, lk.float(), penalty)

    return penalty



def crawl_fourbeat_gait_phase(
    env,
    sensor_cfg: SceneEntityCfg,
    period: float = 1.0,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    # order: [LH, RH, LK, RK]
    lh = contact[:, 0]
    rh = contact[:, 1]
    lk = contact[:, 2]
    rk = contact[:, 3]

    t = env.episode_length_buf.float() * env.step_dt
    phase = torch.remainder(t, period) / period

    q1 = phase < 0.25
    q2 = (phase >= 0.25) & (phase < 0.50)
    q3 = (phase >= 0.50) & (phase < 0.75)
    q4 = phase >= 0.75

    # score each phase: 3 support limbs should contact, 1 swing limb should not
    s1 = (rh.float() + lk.float() + rk.float() + (~lh).float()) / 4.0
    s2 = (lh.float() + rh.float() + lk.float() + (~rk).float()) / 4.0
    s3 = (lh.float() + lk.float() + rk.float() + (~rh).float()) / 4.0
    s4 = (lh.float() + rh.float() + rk.float() + (~lk).float()) / 4.0

    reward = torch.zeros_like(s1)
    reward = torch.where(q1, s1, reward)
    reward = torch.where(q2, s2, reward)
    reward = torch.where(q3, s3, reward)
    reward = torch.where(q4, s4, reward)

    return reward

def crawl_contact_count_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    target_contacts: float = 3.0,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    num_contacts = torch.sum(contact.float(), dim=1)
    return torch.square(num_contacts - target_contacts)

def hop_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    num_contacts = torch.sum(contact.float(), dim=1)

    # penalize vertical velocity + too few contacts
    return torch.square(asset.data.root_lin_vel_w[:, 2]) + 2.0 * (num_contacts < 3).float()

def hand_contact_reward(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    return torch.sum(contact.float(), dim=1)


def hand_force_reward(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    return torch.sum(forces_z, dim=1)

def crawl_limb_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Penalize sliding of hands/knees while they are in contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history

    contact = torch.max(
        torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold

    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    return torch.sum(torch.norm(body_vel_xy, dim=-1) * contact.float(), dim=1)


def hand_knee_spacing(
    env,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.15,
    max_dist: float = 0.75,
) -> torch.Tensor:
    """
    Penalize crawl limbs being too collapsed or too stretched.
    body order:
      [left_hand, right_hand, left_knee, right_knee]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    left_hand = pos[:, 0]
    right_hand = pos[:, 1]
    left_knee = pos[:, 2]
    right_knee = pos[:, 3]

    left_dist = torch.norm(left_hand[:, :2] - left_knee[:, :2], dim=1)
    right_dist = torch.norm(right_hand[:, :2] - right_knee[:, :2], dim=1)

    too_close = (min_dist - left_dist).clamp(min=0.0) + (min_dist - right_dist).clamp(min=0.0)
    too_far = (left_dist - max_dist).clamp(min=0.0) + (right_dist - max_dist).clamp(min=0.0)

    return too_close + too_far


def body_orientation_target_l2(
    env,
    target_gravity_b: tuple,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]

    gravity_b = math_utils.quat_apply_inverse(
        body_quat,
        asset.data.GRAVITY_VEC_W,
    )

    target = torch.tensor(
        target_gravity_b,
        device=gravity_b.device,
        dtype=gravity_b.dtype,
    ).unsqueeze(0)

    return torch.sum(torch.square(gravity_b - target), dim=1)