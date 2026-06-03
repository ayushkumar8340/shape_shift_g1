import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

import mdp as mdp
from assets.unitree import G1_CFG

from envs.base.base_config import BaseSceneCfg
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from terrains import GRAVEL_TERRAINS_CFG

BASKETBALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.125,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),  # Orange
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.650),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=1.2,
            restitution=0.05,
            restitution_combine_mode="min",
            friction_combine_mode="max",
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.15, 0.15)),
)


def reset_ball_relative_to_robot(
    env, env_ids: torch.Tensor, ball_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
    offset_range_x: tuple[float, float], offset_range_y: tuple[float, float],
):
    """Resets the ball in front of the robot, accounting for the robot's yaw/position."""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]

    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]
    num_resets = len(env_ids)

    local_x = torch.rand(num_resets, device=env.device) * (offset_range_x[1] - offset_range_x[0]) + offset_range_x[0]
    local_y = torch.rand(num_resets, device=env.device) * (offset_range_y[1] - offset_range_y[0]) + offset_range_y[0]

    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    world_x = robot_pos[:, 0] + local_x * torch.cos(yaw) - local_y * torch.sin(yaw)
    world_y = robot_pos[:, 1] + local_x * torch.sin(yaw) + local_y * torch.cos(yaw)
    world_z = env.scene.env_origins[env_ids, 2] + 0.125  # rest on floor (center == radius)

    root_state = ball.data.default_root_state[env_ids].clone()
    root_state[:, 0] = world_x
    root_state[:, 1] = world_y
    root_state[:, 2] = world_z
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:13] = 0.0
    ball.write_root_state_to_sim(root_state, env_ids)


@configclass
class LiftSceneCfg(BaseSceneCfg):
    """Register the ball as a dataclass field on the scene."""
    ball: RigidObjectCfg = BASKETBALL_CFG


HANDS = SceneEntityCfg("robot", body_names=[".*left_rubber_hand.*", ".*right_rubber_hand.*"])


@configclass
class G1LiftRewardCfg:
    # --- Grasp (learn a real sandwich grip) ----------------------------------
    reach_surface = RewTerm(
        func=mdp.hands_to_ball_surface_tanh,
        weight=5.0,
        params={"asset_cfg": HANDS, "radius": 0.125, "std": 0.4},
    )
    # perfect_grasp = RewTerm(
    #     func=mdp.bimanual_sphere_grasp, 
    #     weight=10.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*left_rubber_hand.*", ".*right_rubber_hand.*"]), "radius": 0.125, "std": 0.1}
    # )
    grasp_quality = RewTerm(
        func=mdp.grasp_quality_reward,
        weight=8.0,
        params={"asset_cfg": HANDS, "radius": 0.125},
    )
    ball_height = RewTerm(
        func=mdp.ball_height_reward,
        weight=10.0,
        params={"asset_cfg": HANDS, "radius": 0.125, "target_h": 0.575},
    )

    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=3.0,
        params={"std": 0.12},
    )

    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-1.0,  # moderate: allow a forward lean to reach the floor ball
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    feet_flat = RewTerm(
        func=mdp.feet_flat_orientation,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*"])},
    )

    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    # feet_on_ground = RewTerm(
    #     func=mdp.penalize_lifted_feet,
    #     weight=-2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_.*"]), "max_height": 0.08},
    # )
    penalize_walking = RewTerm(func=mdp.penalize_walking, weight=-0.5)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    lost_ball = RewTerm(func=mdp.ball_rolled_away, weight=-20.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*(ankle|wrist|rubber_hand).*).*"),
            "threshold": 1.0,
        },
    )
    hand_kickstand_penalty = RewTerm(
        func=mdp.penalize_hand_kickstand,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*left_rubber_hand.*", ".*right_rubber_hand.*", ".*left_wrist.*", ".*right_wrist.*"],
            ),
            "min_height": 0.06,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class G1LiftFlatEnvCfg(BaseEnvCfg):
    scene: LiftSceneCfg = LiftSceneCfg()
    reward = G1LiftRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "plane"

        self.commands.ranges.base_height = (0.35, 0.78)

        # Remove HIP if needed
        self.robot.terminate_contacts_body_names = [
            ".*torso.*", ".*pelvis.*", ".*waist.*",
            ".*_hip_.*", ".*_shoulder_.*", ".*_elbow_.*", ".*head.*",
        ]

        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        self.domain_rand.events.reset_ball = EventTerm(
            func=reset_ball_relative_to_robot,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "offset_range_x": (0.30, 0.40),
                "offset_range_y": (-0.1, 0.1),
            },
        )


@configclass
class G1LiftFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_lift_flat"
    wandb_project: str = "g1_lift_flat"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.005
        self.policy.init_noise_std = 1.0
