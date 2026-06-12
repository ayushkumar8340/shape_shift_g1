from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
import torch

import mdp as mdp
from assets.unitree import G1_CFG
from envs.base.base_config import BaseSceneCfg
from envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from terrains import GRAVEL_TERRAINS_CFG

# --- 1. Define the Bouncy Basketball ---
BASKETBALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.125,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)), # Orange
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=10.0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.650),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.75,
            restitution_combine_mode="max",
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            # contact_offset=0.02,
            # rest_offset=0.01,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.15, 0.7)), # Spawn precisely under right hand
)

# --- 2. Custom Spawn Logic for the Ball ---
def reset_ball_under_hand(env, env_ids: torch.Tensor, ball_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg, offset_range_x: tuple[float, float], offset_range_y: tuple[float, float]):
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]

    num_resets = len(env_ids)

    # Spawn relative to base: 35cm forward, 15cm right (directly under the right hand)
    # OG WITHOUT RANDOMIZATION!! 
    # DO NOT DELETE!!
    # local_x = torch.ones(num_resets, device=env.device) * 0.35
    # local_y = torch.ones(num_resets, device=env.device) * -0.15

    # 2. Sample local forward (X) and left/right (Y) distances
    local_x = torch.rand(num_resets, device=env.device) * (offset_range_x[1] - offset_range_x[0]) + offset_range_x[0]
    local_y = torch.rand(num_resets, device=env.device) * (offset_range_y[1] - offset_range_y[0]) + offset_range_y[0]

    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    world_x = robot_pos[:, 0] + local_x * torch.cos(yaw) - local_y * torch.sin(yaw)
    world_y = robot_pos[:, 1] + local_x * torch.sin(yaw) + local_y * torch.cos(yaw)
    world_z = env.scene.env_origins[env_ids, 2] + 0.7 # Drop from 0.7 height so it bounces immediately

    root_state = ball.data.default_root_state[env_ids].clone()
    root_state[:, 0] = world_x
    root_state[:, 1] = world_y
    root_state[:, 2] = world_z
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:13] = 0.0

    ball.write_root_state_to_sim(root_state, env_ids)

@configclass
class DribbleSceneCfg(BaseSceneCfg):
    ball: RigidObjectCfg = BASKETBALL_CFG

@configclass
class G1DribbleRewardCfg:
    # --- Dribbling Mechanics ---
    track_hand = RewTerm(func=mdp.ball_under_hand_xy_tanh, weight=5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])})
    bounce_activity = RewTerm(func=mdp.ball_bounce_activity, weight=10.0, params={"max_reward_vel": 3.0})
    dribble_strike = RewTerm(func=mdp.dribble_strike,  weight=20.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])})

    # --- Dribbling Penalties ---
    pinning_penalty = RewTerm(
        func=mdp.penalize_pinning, 
        weight=-50.0, # Massive penalty for holding the ball down. Forces it to let go and wait!
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*right_rubber_hand.*"])}
    )
    wild_dribbling = RewTerm(func=mdp.wild_dribbling, weight=-1.0, params={"speed_limit": 3.0})
    ball_xy_drift = RewTerm(func=mdp.ball_xy_drift, weight=-5.0)

    # --- Standard Penalties ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # The new Head-over-Pelvis wall prevents the lateral snap entirely
    no_lean = RewTerm(
        func=mdp.penalize_lateral_lean, 
        weight=-50.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*head.*", ".*pelvis.*"])} # Ensure these match your URDF links
    )
    # Keep the left arm completely stiff at its side so it doesn't flail
    feet_on_ground = RewTerm(
        func=mdp.penalize_lifted_feet,
        weight=-200.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_.*"]), "max_height": 0.05}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_.*"),
        },
    )
    stand_still = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-20.0,  # -10.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*left_shoulder.*", ".*left_elbow.*", ".*left_wrist.*", ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", ".*waist.*"])}
    )

@configclass
class G1DribbleFlatEnvCfg(BaseEnvCfg):
    scene: DribbleSceneCfg = DribbleSceneCfg()
    reward = G1DribbleRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        new_joint_pos = {}
        new_joint_pos["left_elbow_joint"] = 1.5
        new_joint_pos["right_elbow_joint"] = 1.5
        new_joint_pos["right_shoulder_pitch_joint"] = 0.5
        new_joint_pos["right_wrist_roll_joint"] = -1.9
        new_joint_pos["right_wrist_yaw_joint"] = -0.6

        # Apply the safe dictionary to the robot config
        self.scene.robot = G1_CFG.replace(
            init_state=G1_CFG.init_state.replace(joint_pos=new_joint_pos)
        )
        # -------------------------------------------------------------

        self.scene.terrain_type = "plane"

        # --- STRICT TERMINATIONS ---
        # Explicitly list joints to avoid regex collisions and to safely exclude the left foot!
        self.robot.terminate_contacts_body_names = [
            ".*torso.*",
            ".*pelvis.*",
            ".*waist.*",
            ".*head.*",
            ".*_hip_.*",
            ".*_knee_.*",
            ".*_shoulder_.*",
            ".*_elbow_.*",
            ".*left_wrist.*",
            ".*left_rubber_hand.*"
        ]

        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        self.domain_rand.events.reset_ball = EventTerm(
            func=reset_ball_under_hand,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "offset_range_x": (0.30, 0.40), 
                "offset_range_y": (-0.05, 0.05), 
            }
        )

@configclass
class G1DribbleFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_dribble_flat"
