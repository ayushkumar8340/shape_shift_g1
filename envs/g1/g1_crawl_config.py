from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import mdp as mdp
from assets.unitree import G1_CRAWL_CFG
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg
from terrains import FLAT_TERRAINS_CFG, GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class G1CrawlRewardCfg(RewardCfg):
    
    track_crawl_lin_vel_xy_exp = RewTerm(
        func=mdp.track_crawl_lin_vel_xy_exp,
        weight=4.0,
        params={
            "std": 0.3,
            "forward_axis": 2,
            "forward_sign": 1.0,
            "lateral_axis": 1,
            "lateral_sign": 1.0,
            "body_cfg": SceneEntityCfg("robot", body_names=".*torso.*"),
        },
    )


    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.8,
        params={"std": 0.5},
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.15,
    )

    body_orientation_horizontal = RewTerm(
        func=mdp.body_orientation_target_l2,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*torso.*",
            ),
            "target_gravity_b": (1.0, 0.0, 0.0),
        },
    )



    hand_force_support = RewTerm(
        func=mdp.hand_force_support,
        weight=0.001,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                ],
            ),
            "min_force": 20.0,
            "max_force": 120.0,
        },
    )

    hand_knee_spacing = RewTerm(
        func=mdp.hand_knee_spacing,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "min_dist": 0.05,
            "max_dist": 0.90,
        },
    )


    crawl_left_right_contact_balance = RewTerm(
        func=mdp.crawl_left_right_contact_balance,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "threshold": 1.0,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                 body_names="(?!.*(rubber_hand|wrist|knee|ankle).*).*",
            ),
            "threshold": 1.0,
        },
    )


    crawl_limb_force = RewTerm(
        func=mdp.body_force,
        weight=-2e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=".*(knee|rubber_hand).*",
            ),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    energy = RewTerm(
        func=mdp.energy,
        weight=-5e-4,
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.005,
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
    )

    joint_deviation_lower_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.50,
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
        weight=-0.30,
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

    crawl_contact_count_penalty = RewTerm(
        func=mdp.crawl_contact_count_penalty,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "threshold": 1.0,
            "target_contacts": 3.5,
        },
    )

    hop_penalty = RewTerm(
        func=mdp.hop_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "threshold": 1.0,
        },
    )

    # hand_contact_reward = RewTerm(
    #     func=mdp.hand_contact_reward,
    #     weight=0.8,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_sensor",
    #             body_names=[
    #                 "left_rubber_hand",
    #                 "right_rubber_hand",
    #             ],
    #         ),
    #         "threshold": 1.0,
    #     },
    # )

    hand_wrist_contact_pair = RewTerm(
        func=mdp.hand_wrist_contact_pair,
        weight=0.8,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                    "left.*wrist.*",
                    "right.*wrist.*",
                ],
            ),
            "threshold": 1.0,
        },
    )

    knee_ankle_contact_pair = RewTerm(
        func=mdp.knee_ankle_contact_pair,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left.*knee.*",
                    "right.*knee.*",
                    "left.*ankle.*",
                    "right.*ankle.*",
                ],
            ),
            "threshold": 1.0,
        },
    )

    crawl_left_right_force_balance = RewTerm(
        func=mdp.crawl_left_right_force_balance,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
        },
    )


@configclass
class G1CrawlFlatEnvCfg(BaseEnvCfg):
    reward = G1CrawlRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        H0 = 0.38

        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CRAWL_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = FLAT_TERRAINS_CFG

        self.robot.feet_body_names = [
            ".*rubber_hand.*",
            ".*wrist.*",
            ".*knee.*",
            ".*ankle.*",
        ]

        self.robot.terminate_contacts_body_names = [
            ".*torso.*",
            ".*head.*",
            ".*shoulder.*",
            ".*elbow.*",
        ]

        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        self.domain_rand.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.domain_rand.events.reset_base.params["pose_range"]["yaw"] = (0,0)

        self.commands.ranges.base_height = (1.0 * H0, 1.0 * H0)

        self.commands.rel_standing_envs = 0.0
        self.commands.ranges.lin_vel_x = (-0.30, 0.30)
        self.commands.ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1CrawlFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_crawl_flat"
    wandb_project: str = "g1_crawl_flat"


@configclass
class G1CrawlRoughEnvCfg(G1CrawlFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG

        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1


@configclass
class G1CrawlRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_crawl_rough"
    wandb_project: str = "g1_crawl_rough"

    def __post_init__(self):
        super().__post_init__()

        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"