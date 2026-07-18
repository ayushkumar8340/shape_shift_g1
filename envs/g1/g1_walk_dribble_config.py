from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import mdp as mdp
from assets.unitree import G1_DRIBBLE_CFG
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg
from envs.g1.g1_dribble_config import (
    DribbleSceneCfg,
    reset_ball_under_hand
)
from mdp.command_gen import WalkDribbleCommandCfg


@configclass
class G1WalkDribbleRewardCfg(RewardCfg):
    # ===================================================================
    # Dribble core — ALL envs. Matches the proven g1_dribble_flat dribble set.
    # ===================================================================
    track_hand = RewTerm(
        func=mdp.ball_under_hand_active_tanh, weight=5.0,
        params={"std": 0.15, "asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])},
    )
    bounce_activity = RewTerm(
        func=mdp.ball_bounce_activity_gated, weight=10.0,
        params={"max_reward_vel": 3.0, "xy_proximity": 0.25,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])},
    )
    dribble_strike = RewTerm(
        func=mdp.dribble_strike, weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])},
    )
    pinning_penalty = RewTerm(
        func=mdp.penalize_pinning, weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])},
    )
    wild_dribbling = RewTerm(func=mdp.wild_dribbling, weight=-1.0, params={"speed_limit": 3.0})

    # ===================================================================
    # Walking — velocity-tracking + GAIT CLOCK, all GATED to moving envs
    # ===================================================================
    track_lin_vel = RewTerm(func=mdp.track_lin_vel, weight=8.0, params={"std": 0.25})
    track_ang_vel = RewTerm(func=mdp.track_ang_vel, weight=4.0, params={"std": 0.4})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=1.0,  # 0.5 -> 1.0: encourage longer swings
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    swing_traj = RewTerm(
        func=mdp.swing_traj, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
                "peak_height": 0.08, "ground_offset": 0.04, "std": 0.05},
    )
    swing_height = RewTerm(
        func=mdp.swing_height, weight=0.5,  # was 2.0
        params={"asset_cfg": SceneEntityCfg("robot",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
                "target_height": 0.12, "ground_offset": 0.04},
    )
    swing_fwd = RewTerm(
        func=mdp.swing_fwd, weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
                "speed_ratio": 2.0, "std": 0.5},
    )
    swing_x_traj = RewTerm(
        func=mdp.swing_x_traj, weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
                "stride_scale": 1.0, "std": 0.08},
    )
    hip_stiff = RewTerm(
        func=mdp.joint_deviation_l1, weight=-15.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*", ".*hip_yaw.*"])},
    )
    knee_swing_left = RewTerm(
        func=mdp.knee_swing, weight=12.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*left_knee_joint"]),
                "amplitude": 0.6, "std": 0.3, "foot": 0},
    )
    knee_swing_right = RewTerm(
        func=mdp.knee_swing, weight=8.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*right_knee_joint"]),
                "amplitude": 0.6, "std": 0.3, "foot": 1},
    )
    feet_too_far = RewTerm(
        func=mdp.feet_lateral_width, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
                "max_width": 0.40},
    )
    gait_phase = RewTerm(
        func=mdp.gait_phase, weight=-14.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
            "force_threshold": 1.0, "swing_clearance": 0.07,
        },
    )
    move_or_die = RewTerm(func=mdp.move_or_die, weight=-2.0, params={"min_speed_ratio": 0.5})
    ball_drift_rel = RewTerm(func=mdp.ball_drift_rel, weight=-1.0)

    ball_pocket_asym = RewTerm(
        func=mdp.ball_pocket_asym, weight=4.0,
        params={"base_x": 0.45, "max_lead": 0.20, "std_x_near": 0.12, "std_x_far": 0.22,
                "plateau_x": 0.06, "pocket_y": -0.15, "std_y": 0.12},
    )

    strike_forward = RewTerm(
        func=mdp.strike_forward, weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"]),
                "speed_ratio": 1.0, "std": 0.4, "touch_dist": 0.10},
    )
    ball_high_swing = RewTerm(
        func=mdp.ball_high_midswing, weight=1.0,
        params={"z_gate": 0.30, "target_z": 0.40},
    )
    foot_ball_kick = RewTerm(
        func=mdp.foot_ball_kick, weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot",
                body_names=[".*left_ankle_roll.*", ".*right_ankle_roll.*"], preserve_order=True),
                "margin": 0.12, "toe_offset": 0.12},
    )
    wrong_limb_touch = RewTerm(
        func=mdp.wrong_limb_touch, weight=-10.0,
        params={"wrong_cfg": SceneEntityCfg("robot",
                body_names=[".*right_wrist_roll_link.*", ".*right_elbow.*", ".*left_rubber_hand.*"]),
                "hand_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"]),
                "touch_radius": 0.165, "hand_margin": 0.02},
    )

    # ===================================================================
    # Stand-and-dribble stabilizers - STANDING envs only (gated via is_standing).
    # ===================================================================
    feet_on_ground = RewTerm(
        func=mdp.feet_planted, weight=-200.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_.*"]), "max_height": 0.08},
    )
    ball_xy_drift = RewTerm(func=mdp.ball_xy_drift, weight=-2.0)
    stand_still = RewTerm(
        func=mdp.stand_still, weight=-20.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*waist.*"])},
    )

    # ===================================================================
    # Shared regularizers / safety — ALL envs.
    # ===================================================================
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    left_arm_stiff = RewTerm(
        func=mdp.joint_deviation_l1, weight=-20.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*left_shoulder.*", ".*left_elbow.*", ".*left_wrist.*"])},
    )
    waist_stiff = RewTerm(
        func=mdp.joint_deviation_l1, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    fly = RewTerm(
        func=mdp.fly, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class G1WalkDribbleFlatEnvCfg(BaseEnvCfg):
    scene: DribbleSceneCfg = DribbleSceneCfg()
    reward = G1WalkDribbleRewardCfg()
    # Set -1.0 for from-scratch runs.
    walk_dribble_command: WalkDribbleCommandCfg = WalkDribbleCommandCfg(fixed_level=1.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_DRIBBLE_CFG
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.domain_rand.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.scene.terrain_type = "plane"
        self.robot.terminate_contacts_body_names = [
            ".*torso.*", ".*pelvis.*", ".*head.*",
        ]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        # Reset the ball under the right hand each reset.
        self.domain_rand.events.reset_ball = EventTerm(
            func=reset_ball_under_hand,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "offset_range_x": (0.4, 0.5),
                "offset_range_y": (-0.1, -0.2),
                # TODO: Add z range as well so it learns to bend down a little if and when needed!!
            },
        )


@configclass
class G1WalkDribbleFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_walk_dribble_flat"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.005
