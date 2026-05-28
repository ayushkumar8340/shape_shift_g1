from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import mdp as mdp
from assets.unitree import G1_CRAWL_CFG
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg
from terrains import FLAT_TERRAINS_CFG, GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class G1CrawlRewardCfg(RewardCfg):
    # -------------------------
    # Command tracking
    # -------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.8,
        params={"std": 0.5},
    )

    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=2.0,
        params={"std": 0.05},
    )

    forward_progress = RewTerm(
        func=mdp.forward_progress,
        weight=0.5,
    )

    # -------------------------
    # Body stability
    # -------------------------
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.15,
    )

    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*torso.*",
            )
        },
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
    )

    # -------------------------
    # Crawl contacts
    # Order is important:
    # [left_hand, right_hand, left_knee, right_knee]
    # -------------------------
    crawl_diagonal_gait_phase = RewTerm(
        func=mdp.crawl_diagonal_gait_phase,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left.*wrist.*",
                    "right.*wrist.*",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "period": 0.9,
            "threshold": 1.0,
        },
    )

    crawl_contact_balance = RewTerm(
        func=mdp.crawl_contact_balance,
        weight=0.8,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left.*wrist.*",
                    "right.*wrist.*",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "threshold": 1.0,
        },
    )

    crawl_limb_slide = RewTerm(
        func=mdp.crawl_limb_slide,
        weight=-0.04,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    "left.*wrist.*",
                    "right.*wrist.*",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left.*wrist.*",
                    "right.*wrist.*",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "threshold": 1.0,
        },
    )

    hand_knee_spacing = RewTerm(
        func=mdp.hand_knee_spacing,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left.*wrist.*",
                    "right.*wrist.*",
                    "left.*knee.*",
                    "right.*knee.*",
                ],
            ),
            "min_dist": 0.15,
            "max_dist": 0.75,
        },
    )

    # Penalize contacts except knees, hands/wrists, and ankles if your model touches ankles in crawl.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names="(?!.*(knee|wrist|ankle).*).*",
            ),
            "threshold": 1.0,
        },
    )

    # -------------------------
    # Force / safety
    # -------------------------
    crawl_limb_force = RewTerm(
        func=mdp.body_force,
        weight=-2e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=".*(knee|wrist).*",
            ),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    # -------------------------
    # Regularization
    # -------------------------
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
        weight=-0.25,
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
        weight=-0.20,
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
class G1CrawlFlatEnvCfg(BaseEnvCfg):
    reward = G1CrawlRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        H0 = 0.38

        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CRAWL_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = FLAT_TERRAINS_CFG

        # These are the contact bodies used in critic contact obs.
        # BaseEnv still calls this "feet_body_names", but for crawl we use knees + hands.
        self.robot.feet_body_names = [
            ".*knee.*",
            ".*wrist.*",
        ]

        # Terminate if torso/head/upper arms touch.
        # Do NOT terminate on knees/wrists/hands.
        self.robot.terminate_contacts_body_names = [
            ".*torso.*",
            ".*head.*",
            ".*shoulder.*",
            ".*elbow.*",
        ]

        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        # Tight reset around crawl pose.
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

        # Crawl height command. This remains in obs through the height-command env.
        self.commands.ranges.base_height = (1.0 * H0, 1.0 * H0)

        # Start easy.
        self.commands.rel_standing_envs = 0.0
        self.commands.ranges.lin_vel_x = (0.10, 0.30)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (-0.25, 0.25)


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