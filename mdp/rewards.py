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

def ball_drift_relative_to_robot(env: BaseEnv) -> torch.Tensor:
    """Penalize ball horizontal motion *relative to the robot* — so dribbling
    while walking forward at 1 m/s is NOT punished (ball moves with you),
    but the ball squirting sideways out of reach IS punished."""
    ball_vel_xy = env.scene["ball"].data.root_lin_vel_w[:, :2]
    robot_vel_xy = env.scene["robot"].data.root_lin_vel_w[:, :2]
    rel_vel = ball_vel_xy - robot_vel_xy
    return torch.sum(torch.square(rel_vel), dim=-1)

def ball_under_hand_active_tanh(env: BaseEnv, std: float = 0.15, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """track_hand, but only paid when the ball is actually in a dribble cycle
    (above the floor or moving vertically). Stops the policy from harvesting
    this reward by freezing with the hand in init pose."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    wrist_xy = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    ball_xy = ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(wrist_xy - ball_xy, dim=-1)

    ball_z = ball.data.root_pos_w[:, 2]
    ball_vz = torch.abs(ball.data.root_lin_vel_w[:, 2])
    is_dribbling = ((ball_z > 0.2) | (ball_vz > 0.3)).float()

    return is_dribbling * (1.0 - torch.tanh(dist_xy / std))

def ball_bounce_activity_gated(env: BaseEnv, max_reward_vel: float = 3.0, xy_proximity: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """bounce_activity, but ONLY paid when the dribbling hand is above the ball.
    Kills the passive 'let the ball bounce itself out' exploit — the policy
    now has to keep the ball alive under its hand to collect this reward."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    hand_xy = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    ball_xy = ball.data.root_pos_w[:, :2]
    near = (torch.norm(hand_xy - ball_xy, dim=-1) < xy_proximity).float()
    vel_z = torch.abs(ball.data.root_lin_vel_w[:, 2])
    return near * torch.clamp(vel_z, max=max_reward_vel)

def feet_too_far(env: BaseEnv, max_width: float = 0.40, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize an abnormally wide stance — kills the 'straddle the ball' gait.
    Pairs with feet_too_near to keep stance width in a normal band."""
    feet_xy = env.robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    width = torch.norm(feet_xy[:, 0] - feet_xy[:, 1], dim=-1)
    return torch.clamp(width - max_width, min=0.0)

def _ball_alive_factor(env: BaseEnv) -> torch.Tensor:
    """Smooth [0,1]: ~1 when the ball is up / moving vertically (being dribbled),
    ~0 when it is dead on the floor. Used to gate the walking reward so the robot
    can never earn walk reward while letting the ball die."""
    ball = env.scene["ball"]
    z = ball.data.root_pos_w[:, 2]
    vz = torch.abs(ball.data.root_lin_vel_w[:, 2])
    by_height = torch.clamp((z - 0.15) / 0.35, 0.0, 1.0)   # 0 at rest (~0.125m), 1 at >=0.5m
    by_vel = torch.clamp(vz / 1.5, 0.0, 1.0)
    return torch.maximum(by_height, by_vel)

def _g1_standing(env: BaseEnv) -> torch.Tensor:
    return env.command_generator.is_standing.float()

def _g1_moving(env: BaseEnv) -> torch.Tensor:
    return (~env.command_generator.is_standing).float()

def track_lin_vel(env: BaseEnv, std: float = 0.4) -> torch.Tensor:
    """Track commanded linear velocity (robot/yaw frame). Moving envs only, and
    only while the ball is alive."""
    robot = env.scene["robot"]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3]
    )
    cmd = env.command_generator.vel_command
    err = torch.sum(torch.square(cmd[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-err / std**2) * _g1_moving(env) * _ball_alive_factor(env)

def track_ang_vel(env: BaseEnv, std: float = 0.5) -> torch.Tensor:
    """Track commanded yaw rate. Moving envs only, gated by ball alive."""
    robot = env.scene["robot"]
    cmd = env.command_generator.vel_command
    err = torch.square(cmd[:, 2] - robot.data.root_ang_vel_w[:, 2])
    return torch.exp(-err / std**2) * _g1_moving(env) * _ball_alive_factor(env)

def feet_air_time(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward single-support stepping. Moving envs only — encourages an actual
    gait instead of foot-dragging to satisfy the velocity command."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = cs.data.current_air_time[:, sensor_cfg.body_ids]
    contact = cs.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact > 0.0
    in_mode = torch.where(in_contact, contact, air)
    single = torch.sum(in_contact.int(), dim=1) == 1
    rew = torch.min(torch.where(single.unsqueeze(-1), in_mode, 0.0), dim=1)[0]
    rew = torch.clamp(rew, max=threshold)
    return rew * _g1_moving(env)

def feet_planted(env: BaseEnv, max_height: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Keep both feet on the ground — STANDING envs only."""
    robot = env.scene["robot"]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    pen = torch.where(foot_z > max_height, foot_z - max_height, 0.0)
    return torch.sum(pen, dim=1) * _g1_standing(env)

def stand_still(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Keep the listed joints near their default — STANDING envs only."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * _g1_standing(env)

def ball_xy_drift(env: BaseEnv) -> torch.Tensor:
    """Penalize the ball drifting horizontally — STANDING envs only (a walking
    env legitimately carries the ball with it)."""
    ball = env.scene["ball"]
    vel_xy = ball.data.root_lin_vel_w[:, :2]
    return torch.sum(torch.square(vel_xy), dim=-1) * _g1_standing(env)

def gait_phase(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, force_threshold: float = 1.0, swing_clearance: float = 0.05) -> torch.Tensor:
    """Gait-clock contact-schedule enforcement — MOVING envs only.

    The command generator dictates, per foot, stance/swing windows on a periodic
    clock (see WalkDribbleCommand.expected_stance). Violations, per foot per step:
      - expected STANCE but no contact force            -> violation
      - expected SWING  but foot LOW (z < clearance)    -> violation
    The swing test is POSITION-based on purpose: merely unweighting the foot below
    the force threshold while skimming the ground (the 'phantom-step shuffle')
    still counts as a violation — the foot must actually clear the ground.
    A both-feet-down slider violates every swing window (~0.9/foot-pair/step);
    a correct stepper pays ~0. Standing envs pay 0 (gated).

    sensor_cfg and asset_cfg must BOTH resolve to [left, right] ankle_roll (use
    preserve_order=True with explicit left/right patterns) so violations align with
    the schedule's [left, right] columns.
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_contact = forces > force_threshold                                   # (N, 2)
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    lifted = foot_z > swing_clearance                                       # (N, 2)

    expected_stance = env.command_generator.expected_stance                 # (N, 2)
    stance_violation = expected_stance & ~in_contact
    # GRACE BAND: only demand clearance in the MIDDLE of the swing (s in [0.2, 0.8]).
    # Without it, the mandatory liftoff/touchdown moments (foot at floor level, exactly
    # as the swing_traj bell prescribes) were fined -2.2/step even for a PERFECT
    # stride — taxing the precise gradient direction the swing rewards try to open.
    cg = env.command_generator
    duty = cg.cfg.gait_duty
    foot_phase = torch.stack([torch.frac(cg.gait_phase), torch.frac(cg.gait_phase + 0.5)], dim=-1)
    s = torch.clamp((foot_phase - duty) / (1.0 - duty), 0.0, 1.0)
    mid_swing = (s > 0.2) & (s < 0.8)
    swing_violation = (~expected_stance) & ~lifted & mid_swing
    violations = (stance_violation | swing_violation).float().sum(dim=1)
    return violations * _g1_moving(env)


def move_or_die(env: BaseEnv, min_speed_ratio: float = 0.5) -> torch.Tensor:
    """Penalize ignoring a nonzero velocity command — MOVING envs only.

    Kills the march-in-place optimum: satisfying the gait clock without translating
    (root speed below min_speed_ratio * commanded speed) now costs reward instead of
    merely forgoing part of the tracking kernel."""
    cmd_speed = torch.norm(env.command_generator.vel_command[:, :2], dim=-1)
    root_speed = torch.norm(env.robot.data.root_lin_vel_w[:, :2], dim=-1)
    too_slow = (root_speed < min_speed_ratio * cmd_speed).float()
    return too_slow * _g1_moving(env)


def ball_drift_rel(env: BaseEnv) -> torch.Tensor:
    """Ball containment while walking — MOVING envs only. Penalizes ball velocity
    RELATIVE to the robot: carrying the ball forward at walking speed is free, the
    ball squirting away from the robot is penalized."""
    return ball_drift_relative_to_robot(env) * _g1_moving(env)


def swing_height(env: BaseEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.12, ground_offset: float = 0.04) -> torch.Tensor:
    """Smooth swing-foot LIFT reward — MOVING envs only.

    IMPORTANT: the ankle_roll link origin sits ~4cm above the ground when the foot
    is planted flat, so raw z must be re-zeroed by ground_offset — otherwise a
    planted foot already collects a third of this reward for free (this is what
    stalled the gait learning).
    asset_cfg must resolve [left, right] ankle_roll (preserve_order=True)."""
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]   # (N, 2)
    lift = torch.clamp((foot_z - ground_offset) / target_height, 0.0, 1.0)
    in_swing = (~env.command_generator.expected_stance).float()                    # (N, 2)
    return (lift * in_swing).sum(dim=1) * _g1_moving(env)


def swing_traj(env: BaseEnv, asset_cfg: SceneEntityCfg, peak_height: float = 0.08, ground_offset: float = 0.04, std: float = 0.05) -> torch.Tensor:
    """Primary gait shaper: swing-foot HEIGHT TRAJECTORY tracking — MOVING envs only.

    Each foot's scheduled swing window gets a bell-shaped reference height:
    lift off the ground, peak at ~8cm of REAL clearance mid-swing, and be back
    down by touchdown. Reward = exp kernel on the tracking error, gated by the
    CLOCK (the scheduled swing), never by actual contact — so a planted foot at
    mid-swing scores near zero and cannot opt out (the loophole that defeats
    contact-gated versions of this reward). Bell shape also removes the
    "keep the foot high into touchdown" exploit of a plain lift carrot."""
    cg = env.command_generator
    duty = cg.cfg.gait_duty
    foot_phase = torch.stack(
        [torch.frac(cg.gait_phase), torch.frac(cg.gait_phase + 0.5)], dim=-1)      # (N, 2) [L, R]
    s = torch.clamp((foot_phase - duty) / (1.0 - duty), 0.0, 1.0)                  # swing progress 0->1
    z_ref = ground_offset + peak_height * torch.sin(torch.pi * s)                  # bell: z0 -> z0+peak -> z0
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]   # (N, 2)
    in_swing = (~cg.expected_stance).float()
    track = torch.exp(-torch.square(foot_z - z_ref) / std**2)
    return (track * in_swing).sum(dim=1) * _g1_moving(env)


def swing_fwd(env: BaseEnv, asset_cfg: SceneEntityCfg, speed_ratio: float = 2.0, std: float = 0.5) -> torch.Tensor:
    """Swing-foot FORWARD-TRAVEL reward — MOVING envs only.

    The vertical bell (swing_traj) says "lift and lower" but not "land ahead",
    so the policy learned to STOMP in place and glide. In a real stride the swing
    foot moves forward at ~2x body speed (the planted foot moves at 0, the body
    averages the two). Reward the swing foot's forward velocity (robot yaw frame)
    matching speed_ratio * commanded forward speed."""
    cg = env.command_generator
    robot = env.scene[asset_cfg.name]
    foot_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]               # (N, 2, 3)
    q = robot.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    fwd_x = torch.cos(yaw).unsqueeze(1)
    fwd_y = torch.sin(yaw).unsqueeze(1)
    foot_vx = foot_vel_w[..., 0] * fwd_x + foot_vel_w[..., 1] * fwd_y             # (N, 2)
    target = (speed_ratio * cg.vel_command[:, 0]).unsqueeze(1)                     # (N, 1)
    track = torch.exp(-torch.square(foot_vx - target) / std**2)
    in_swing = (~cg.expected_stance).float()
    return (track * in_swing).sum(dim=1) * _g1_moving(env)


def swing_x_traj(env: BaseEnv, asset_cfg: SceneEntityCfg, stride_scale: float = 1.0, std: float = 0.08) -> torch.Tensor:
    """Swing-foot FORE-AFT TRAJECTORY target — MOVING envs only.

    The missing half of swing_traj: the height bell says lift/lower but nothing
    says WHERE the foot must be along the direction of travel, so one leg can take
    full strides while the other barely advances (the right-leg lag). Both feet get
    the SAME clock-referenced fore-aft reference (robot yaw frame, relative to base):
        x_ref(s) = (s - 0.5) * duty * T * v_cmd * stride_scale
    — start the swing half the stance travel BEHIND the base and land the mirror
    image ahead, which is the excursion a constant-speed gait geometrically
    requires (the planted foot travels backward at v for duty*T seconds). One
    shared target = equal strides by construction; the amplitude scales with the
    COMMAND, so a zero-stride 'symmetric' gait scores near zero and cannot game
    it. Clock-gated like swing_traj (a planted foot at mid-swing pays the
    tracking error; it cannot opt out). asset_cfg must resolve
    [left_ankle_roll, right_ankle_roll] (preserve_order=True)."""
    cg = env.command_generator
    duty = cg.cfg.gait_duty
    period = 1.0 / cg.cfg.gait_freq
    foot_phase = torch.stack(
        [torch.frac(cg.gait_phase), torch.frac(cg.gait_phase + 0.5)], dim=-1)      # (N, 2) [L, R]
    s = torch.clamp((foot_phase - duty) / (1.0 - duty), 0.0, 1.0)                  # swing progress 0->1
    robot = env.scene[asset_cfg.name]
    rel_xy = (robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
              - robot.data.root_pos_w[:, :2].unsqueeze(1))                          # (N, 2, 2)
    q = robot.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    fwd_x = torch.cos(yaw).unsqueeze(1)
    fwd_y = torch.sin(yaw).unsqueeze(1)
    foot_x = rel_xy[..., 0] * fwd_x + rel_xy[..., 1] * fwd_y                        # (N, 2) fore-aft
    stance_travel = duty * period * cg.vel_command[:, 0].unsqueeze(1)               # (N, 1)
    x_ref = (s - 0.5) * stance_travel * stride_scale
    track = torch.exp(-torch.square(foot_x - x_ref) / std**2)
    in_swing = (~cg.expected_stance).float()
    return (track * in_swing).sum(dim=1) * _g1_moving(env)


def knee_swing(env: BaseEnv, asset_cfg: SceneEntityCfg, amplitude: float = 0.6, std: float = 0.3, foot: int = -1) -> torch.Tensor:
    """Phase-referenced KNEE FLEXION target during scheduled swing — MOVING envs only.

    humanoid-gym-style joint-reference shaping, limited to the KNEES (arms untouched):
    during each foot's scheduled swing window (same clock/duty math as swing_traj)
    the SAME-SIDE knee tracks
        knee_ref = knee_default + amplitude * sin(pi * swing_progress)
    i.e. bend to default+amplitude mid-swing, back to default by touchdown. G1 knee
    sign: POSITIVE = flexion (init 0.42, kneel pose 1.507), so +amplitude bends the
    knee; peak 1.02 rad is far inside the soft limit. Gated by the CLOCK, never
    by contact — a planted straight leg at mid-swing scores ~0.02 and cannot opt out.

    foot=-1 (both): asset_cfg must resolve [left_knee, right_knee]
    (preserve_order=True) so columns align with expected_stance's [left, right].
    foot=0 (left) / foot=1 (right): asset_cfg resolves THAT single knee only —
    used to split the reward into per-leg terms so each leg's earning is a
    separate logged metric. sum(left term + right term) == the both-feet term."""
    cg = env.command_generator
    duty = cg.cfg.gait_duty
    foot_phase = torch.stack(
        [torch.frac(cg.gait_phase), torch.frac(cg.gait_phase + 0.5)], dim=-1)      # (N, 2) [L, R]
    s = torch.clamp((foot_phase - duty) / (1.0 - duty), 0.0, 1.0)                  # swing progress 0->1
    in_swing = (~cg.expected_stance).float()                                       # (N, 2)
    if foot >= 0:
        s = s[:, foot:foot + 1]                                                    # (N, 1)
        in_swing = in_swing[:, foot:foot + 1]                                      # (N, 1)
    asset: Articulation = env.scene[asset_cfg.name]
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]                            # (N, 2) or (N, 1)
    knee_ref = (asset.data.default_joint_pos[:, asset_cfg.joint_ids]
                + amplitude * torch.sin(torch.pi * s))                             # bend & return
    track = torch.exp(-torch.square(knee - knee_ref) / std**2)
    return (track * in_swing).sum(dim=1) * _g1_moving(env)


def feet_lateral_width(env: BaseEnv, asset_cfg: SceneEntityCfg, max_width: float = 0.40) -> torch.Tensor:
    """Penalize an over-wide stance using LATERAL separation only (robot yaw frame).

    Replaces the XY-norm feet_too_far, which counted fore-aft stride separation as
    'width' and silently clipped stride length at walking speed. A long stride is
    free here; only the sideways straddle pays."""
    robot = env.scene[asset_cfg.name]
    feet_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]                     # (N, 2, 2)
    diff = feet_xy[:, 0, :] - feet_xy[:, 1, :]                                     # (N, 2)
    q = robot.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # lateral component = projection onto the robot's left axis (-sin, cos)
    lateral = torch.abs(-torch.sin(yaw) * diff[:, 0] + torch.cos(yaw) * diff[:, 1])
    return torch.clamp(lateral - max_width, min=0.0)

def strike_forward(env: BaseEnv, asset_cfg: SceneEntityCfg, speed_ratio: float = 1.0, std: float = 0.4, touch_dist: float = 0.10) -> torch.Tensor:
    """Strike-DIRECTION shaping — MOVING envs only.

    Co-fires with dribble_strike (same touching + downward-push core, so it can only
    be earned during a genuine firm strike) and additionally rewards the ball leaving
    the strike with FORWARD velocity ~= commanded speed, so the ball CO-MOVES with
    the walking robot. Physics of the fix: a straight-down strike leaves the ball
    behind by cmd_speed * bounce_period ~= 0.3-0.5m per bounce — a sawtooth that
    periodically drops the ball beside the right foot and shortens the right stride.
    Matching ball forward speed to the command zeroes that drift. Target is EXACTLY
    the command (no surge), so the ball never systematically outruns the robot.
    Standing envs pay 0 — the golden static dribble is untouched."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    hand_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    hand_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    ball_pos = ball.data.root_pos_w
    ball_vel = ball.data.root_lin_vel_w

    dist = torch.norm(hand_pos - ball_pos, dim=-1) - 0.125     # ball radius
    is_touching = (dist < touch_dist).float()
    push_down_vel = torch.clamp(-hand_vel[:, 2], min=0.0, max=2.0)

    q = robot.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    ball_vx = ball_vel[:, 0] * torch.cos(yaw) + ball_vel[:, 1] * torch.sin(yaw)

    target = speed_ratio * env.command_generator.vel_command[:, 0]
    match = torch.exp(-torch.square(ball_vx - target) / std**2)
    return is_touching * push_down_vel * match * _g1_moving(env) * _ball_alive_factor(env)


def ball_pocket_asym(env: BaseEnv, base_x: float = 0.35, max_lead: float = 0.20,
                        std_x_near: float = 0.12, std_x_far: float = 0.22,
                        plateau_x: float = 0.06, pocket_y: float = -0.27,
                        std_y: float = 0.12) -> torch.Tensor:
    """Asymmetric ball pocket (replaces the isotropic dw_ball_pocket in the cfg).

    Fore-aft: STEEP behind the target (a lagging ball gets pulled forward before it
    re-enters the right foot's reach band) and LOOSE ahead (the natural
    strike -> lead -> catch-up oscillation of a moving dribble is not penalized),
    with a free +/-plateau_x zone. Lateral: tighter kernel at pocket_y=-0.27 so the
    ball's inboard edge clears the right foot lane (at -0.25 the edge was tangent
    to the lane center). Gated by ball_alive as before."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    rel = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(robot.data.root_quat_w), ball.data.root_pos_w - robot.data.root_pos_w)
    cmd_vx = env.command_generator.vel_command[:, 0]
    px = base_x + max_lead * torch.clamp(cmd_vx.abs(), max=0.8) / 0.8
    dx = rel[:, 0] - px
    dx = torch.sign(dx) * torch.clamp(dx.abs() - plateau_x, min=0.0)   # free plateau
    std_x = torch.where(dx < 0, torch.full_like(dx, std_x_near), torch.full_like(dx, std_x_far))
    score_x = torch.exp(-torch.square(dx / std_x))
    score_y = torch.exp(-torch.square((rel[:, 1] - pocket_y) / std_y))
    return score_x * score_y * _ball_alive_factor(env)


def ball_high_midswing(env: BaseEnv, z_gate: float = 0.30, target_z: float = 0.40) -> torch.Tensor:
    """CONTINGENCY LEVER — defined but NOT wired into the cfg by default.

    Pays for the ball being HIGH during either foot's mid-swing window: favors the
    bounce rhythm whose apexes land inside swing windows, so the ball is never low
    (kickable) exactly when a foot is airborne. Wire at ~+2.0 only if stride
    asymmetry persists after the strike-forward + asymmetric-pocket fix."""
    cg = env.command_generator
    duty = cg.cfg.gait_duty
    fp = torch.stack([torch.frac(cg.gait_phase), torch.frac(cg.gait_phase + 0.5)], dim=-1)
    s = torch.clamp((fp - duty) / (1.0 - duty), 0.0, 1.0)
    mid = ((s > 0.2) & (s < 0.8)).any(dim=1).float()
    z = env.scene["ball"].data.root_pos_w[:, 2]
    high = torch.clamp((z - z_gate) / (target_z - z_gate), 0.0, 1.0)
    return high * mid * _g1_moving(env) * _ball_alive_factor(env)


def foot_ball_kick(env: BaseEnv, asset_cfg: SceneEntityCfg, margin: float = 0.20, toe_offset: float = 0.12) -> torch.Tensor:
    """Graded "don't step into the ball" penalty — ALL envs.

    Two probe points per foot (ankle origin + a toe probe 12cm along the foot's
    yaw-forward, since the ankle origin is rear-biased). Hinge penalty grows as a
    probe's XY distance to the ball center drops below margin. Only counts when
    the ball is LOW (z < 0.30) — a ball at bounce apex cannot be kicked.
    Deliberately NOT gated by ball_alive: kicking a dead ball away is a real error."""
    robot = env.scene[asset_cfg.name]
    ball = env.scene["ball"].data.root_pos_w
    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]                     # (N, 2, 3)
    q = robot.data.body_quat_w[:, asset_cfg.body_ids, :]                           # (N, 2, 4)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    toe = foot_pos[..., :2].clone()
    toe[..., 0] += toe_offset * torch.cos(yaw)
    toe[..., 1] += toe_offset * torch.sin(yaw)
    probes = torch.cat([foot_pos[..., :2], toe], dim=1)                            # (N, 4, 2)
    d = torch.norm(probes - ball[:, None, :2], dim=-1).min(dim=1)[0]               # (N,)
    pen = torch.clamp(margin - d, min=0.0) / margin
    ball_low = (ball[:, 2] < 0.30).float()
    return pen * ball_low


def wrong_limb_touch(env: BaseEnv, wrong_cfg: SceneEntityCfg, hand_cfg: SceneEntityCfg, touch_radius: float = 0.165, hand_margin: float = 0.02) -> torch.Tensor:
    """Penalize dribbling with the WRONG limb (forearm/elbow/off-hand carry) — ALL envs.

    Fires only when the ball is strictly CLOSER to a wrong limb than to the rubber
    hand (minus a margin) — that is the carry signature. A plain proximity test
    would fire on every legitimate deep strike, because the wrist links sit only
    4-9cm from the hand; the relative test exempts all real strikes."""
    robot = env.scene[wrong_cfg.name]
    ball = env.scene["ball"].data.root_pos_w
    d_wrong = torch.norm(
        robot.data.body_pos_w[:, wrong_cfg.body_ids, :] - ball[:, None, :], dim=-1).min(dim=1)[0]
    d_hand = torch.norm(
        robot.data.body_pos_w[:, hand_cfg.body_ids[0], :] - ball, dim=-1)
    return ((d_wrong < touch_radius) & (d_wrong < d_hand - hand_margin)).float()
