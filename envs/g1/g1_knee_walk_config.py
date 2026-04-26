from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import mdp as mdp
from assets.unitree import G1_KNEE_WALK_CFG
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg
from terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class G1KneeWalkingRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"std": 0.5},
    )

    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=2.0,
        params={"std": 0.06},
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)

    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # knee_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.10,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*knee.*"), "threshold": 0.3},
    # )

    knee_contact = RewTerm(
        func=mdp.desired_contacts,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*knee.*"),
            "threshold": 1.0,
        },
    )

    knee_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*knee.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*knee.*"),
        },
    )

    knee_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*knee.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names="(?!.*(knee|ankle).*).*",
            ),
            "threshold": 1.0,
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    joint_deviation_lower_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch.*",
                    ".*_hip_roll.*",
                    ".*_hip_yaw.*",
                    ".*_knee.*",
                    ".*_ankle.*",
                ],
            )
        },
    )

    joint_deviation_upper_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*waist.*",
                    ".*_shoulder.*",
                    ".*_elbow.*",
                    ".*_wrist.*",
                ],
            )
        },
    )


@configclass
class G1KneeWalkFlatEnvCfg(BaseEnvCfg):
    reward = G1KneeWalkingRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        H0 = 0.78 

        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_KNEE_WALK_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG

        # Knees are the locomotion contact bodies.
        self.robot.feet_body_names = [".*knee.*"]

        # Do not terminate on knees. Terminate on upper body / torso / arms / head / feet.
        self.robot.terminate_contacts_body_names = [
            ".*torso.*",
            ".*head.*",
            ".*shoulder.*",
            ".*elbow.*",
            ".*wrist.*",
        ]

        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        # Low height range, but still commanded as obs.
        self.commands.ranges.base_height = (0.22 * H0, 0.45 * H0)

        # Start conservative.
        self.commands.ranges.lin_vel_x = (0.0, 0.4)
        self.commands.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1KneeWalkFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_knee_walk_flat"
    wandb_project: str = "g1_knee_walk_flat"


@configclass
class G1KneeWalkRoughEnvCfg(G1KneeWalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1


@configclass
class G1KneeWalkRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_knee_walk_rough"
    wandb_project: str = "g1_knee_walk_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"