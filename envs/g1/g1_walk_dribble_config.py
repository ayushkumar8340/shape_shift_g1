from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import mdp as mdp
from assets.unitree import G1_CFG
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg
from envs.g1.g1_dribble_config import (
    DribbleSceneCfg,
    reset_ball_under_hand,
)


@configclass
class G1WalkDribbleRewardCfg(RewardCfg):
    # ===== Walk-to-target =====
    track_pos = RewTerm(func=mdp.track_position_exp_wd, weight=3.0, params={"sigma": 1.0})
    stop_at_target = RewTerm(func=mdp.stop_at_target_exp_wd, weight=5.0, params={"dist_threshold": 0.35, "sigma": 0.2})
    move_to_target = RewTerm(func=mdp.progress_towards_target, weight=15.0)
    face_target = RewTerm(func=mdp.face_target_exp_wd, weight=1.5, params={"sigma": 0.5})
    stand_still_at_target = RewTerm(func=mdp.penalize_wobble_at_target_wd, weight=-0.5, params={"dist_threshold": 0.35})

    # ===== Locomotion regularizers =====
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-5e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # ===== Contact safety =====
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor",
                body_names="(?!.*ankle.*|.*right_rubber_hand.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "threshold": 1.0},
    )

    # ===== Posture =====
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ===== Walking quality =====
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped_wd,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")},
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "threshold": 500, "max_reward": 400},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
                "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    # ===== Joint deviation =====
    # Hip yaw/roll only — pitch must stay free for stepping.
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot",
                joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    # LEFT arm + waist only — the dribbling arm must be free.
    left_arm_stiff = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                joint_names=[".*left_shoulder.*", ".*left_elbow.*", ".*left_wrist.*"])},
    )

    waist_stiff = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])},
    )

    # Light pose anchor on legs (same as g1_flat) — does NOT prevent walking.
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot",
                joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    )
    no_loitering = RewTerm(
        func=mdp.penalize_stationary_when_far,
        weight=-3.0,
        params={"dist_threshold": 0.5, "speed_threshold": 0.15},
    )
    upright_posture = RewTerm(func=mdp.upright_posture_exp,
        weight=6.0,
        params={"target_height": 0.70, "std": 0.12}
    )

    # ===== Dribbling (scaled down vs stand-and-dribble task) =====
    track_hand = RewTerm(
        func=mdp.ball_under_hand_active_tanh,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"]),
                "std": 0.15},
    )
    bounce_activity = RewTerm(
        func=mdp.ball_bounce_activity_gated, weight=6.0,
        params={"max_reward_vel": 3.0, "xy_proximity": 0.25,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])},
    )
    dribble_strike = RewTerm(
        func=mdp.dribble_strike,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])},
    )

    # ===== Dribbling penalties =====
    pinning_penalty = RewTerm(
        func=mdp.penalize_pinning,
        weight=-30.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])},
    )
    wild_dribbling = RewTerm(
        func=mdp.wild_dribbling,
        weight=-1.0,
        params={"speed_limit": 3.5}
    )

    ball_drift = RewTerm(func=mdp.ball_drift_relative_to_robot, weight=-1.0)
    # Reward the ball moving WITH the robot when robot is walking.
    ball_follow = RewTerm(
        func=mdp.ball_keeps_up_with_robot,
        weight=2.0,
        params={"sigma": 0.4}
    )
    feet_too_far = RewTerm(
        func=mdp.feet_too_far, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
                "max_width": 0.40},
    )


@configclass
class G1WalkDribbleFlatEnvCfg(BaseEnvCfg):
    scene: DribbleSceneCfg = DribbleSceneCfg()
    reward = G1WalkDribbleRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # Arm in dribble-ready pose (same as g1_dribble_flat).
        new_joint_pos = {
            # "left_elbow_joint": 1.5,
            "right_elbow_joint": 1.5,
            "right_shoulder_pitch_joint": 0.5,
            "right_wrist_roll_joint": -1.9,
            "right_wrist_yaw_joint": -0.6,
        }
        self.scene.robot = G1_CFG.replace(
            init_state=G1_CFG.init_state.replace(joint_pos=new_joint_pos)
        )

        self.scene.terrain_type = "plane"
        self.scene.max_episode_length_s = 20.0

        # Lenient terminations -- only true falls. The shoulder/elbow/hand
        # contacts that the stand-and-dribble task terminates on would kill
        # episodes during walking due to natural arm swing.
        self.robot.terminate_contacts_body_names = [
            ".*torso.*", ".*pelvis.*", ".*head.*",
        ]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        # Modest base reset randomization (helps the policy generalize).
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        # Don't perturb joints at reset -- keep the dribble-ready arm pose.
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.domain_rand.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Reset ball under the right hand.
        self.domain_rand.events.reset_ball = EventTerm(
            func=reset_ball_under_hand,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "offset_range_x": (0.30, 0.40),
                "offset_range_y": (-0.05, 0.05),
            },
        )


@configclass
class G1WalkDribbleFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_walk_dribble_flat"
    max_iterations: int = 20000

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 1.0
        self.algorithm.entropy_coef = 0.003