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

# --- 1. The basketball (same physical ball as the dribble task) ------------ #
BASKETBALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.125,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),  # Orange
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=8,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.650),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.4,
            restitution_combine_mode="max",
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
    # Nominal only - the reset event below places the ball root-relative.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.915)),
)


# --- 2. The two-hand V-cradle holding pose ---------------------------------- #
# FK-verified against the g1_29dof USD (see logs of the pose analysis): at the
# all-zero arm pose the G1's palms already face each other 0.253 m apart - the
# 0.25 m ball almost exactly fits. The cradle is therefore a SMALL perturbation
# of zero, not large joint values. This pose puts the palm plates at pelvis
# frame (0.297, +/-0.132, +0.070), inward normals tilted 14.6 deg upward (a V:
# the squeeze pushes the ball up, gravity seats it), with ~3.5 mm per-palm
# interference on a ball centered at pelvis-frame (0.297, 0, +0.105) so the PD
# actuators apply a standing grip force. Every joint keeps >=0.85 rad margin to
# its soft limit, so the +/-0.25-scaled actions retain full symmetric authority
# (a default AT a limit would kill half the DOF's range - the old commented
# pose had wrists at 99.9% of the hard stop).
#
# Sign notes (g1_29dof URDF): positive shoulder_pitch swings the hanging arm
# BACKWARD; positive elbow rotates the (forward-pointing) forearm DOWN. Keys
# must not overlap (regex resolution raises on multiple matches), so the arm
# joints are written out per side and the legs keep the G1 defaults.
HOLDING_JOINT_POS = {
    # legs: unchanged G1 defaults
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    # arms: FK-verified V-cradle
    "left_shoulder_pitch_joint": -0.05,
    "right_shoulder_pitch_joint": -0.05,
    "left_shoulder_roll_joint": -0.16,
    "right_shoulder_roll_joint": 0.16,
    "left_shoulder_yaw_joint": 0.10,
    "right_shoulder_yaw_joint": -0.10,
    "left_elbow_joint": 0.50,
    "right_elbow_joint": 0.50,
    "left_wrist_roll_joint": -0.20,
    "right_wrist_roll_joint": 0.20,
    "left_wrist_pitch_joint": -0.60,
    "right_wrist_pitch_joint": -0.60,
    "left_wrist_yaw_joint": -0.15,
    "right_wrist_yaw_joint": 0.15,
}


# --- 3. Place the ball between both hands at episode start ------------------ #
def reset_ball_in_hands(
    env,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    forward_offset: float,
    height_rel: float,
    lateral: float = 0.0,
):
    """Spawn the ball in the V-cradle, root-relative in ALL three axes.

    Body positions are not yet updated at reset-event time (sim.forward runs
    later), so the ball is placed from the root pose plus a yaw-rotated local
    offset rather than from the live hand positions. ``height_rel`` is relative
    to the CURRENT root z (the pelvis settles ~2 cm after reset and the ball
    settles with it into the pocket), sized to start ~1 cm above the cradle
    center so the ball drops in rather than spawning interpenetrated.
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_quat = robot.data.root_quat_w[env_ids]

    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    world_x = robot_pos[:, 0] + forward_offset * torch.cos(yaw) - lateral * torch.sin(yaw)
    world_y = robot_pos[:, 1] + forward_offset * torch.sin(yaw) + lateral * torch.cos(yaw)
    world_z = robot_pos[:, 2] + height_rel

    root_state = ball.data.default_root_state[env_ids].clone()
    root_state[:, 0] = world_x
    root_state[:, 1] = world_y
    root_state[:, 2] = world_z
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:13] = 0.0
    ball.write_root_state_to_sim(root_state, env_ids)


@configclass
class ThrowSceneCfg(BaseSceneCfg):
    # Documentation only: the scene wrapper does not read this field. The ball
    # actually enters the scene via ``SceneCfg.ball = BASKETBALL_CFG`` in
    # G1ThrowEnv.__init__ (same mechanism as the dribble task).
    ball: RigidObjectCfg = BASKETBALL_CFG


# --- 4. Task thresholds (read by G1ThrowEnv) ------------------------------- #
@configclass
class ThrowCfg:
    # Holding / set detection. Hand body origins sit at the palm HEEL, ~9 cm
    # behind the grip point; at the cradle each hand origin is ~0.16 m from the
    # ball center, so 0.25 is "hand on ball" with margin but well short of the
    # ball resting on the chest.
    hold_dist: float = 0.25         # per-hand ball-center distance that counts as "hand on ball"
    # dot(u_left, u_right) below this = hands not bunched on one side. The
    # side-side cradle scores ~-0.87, a natural shooting pocket (right hand
    # under-behind, left on the side) ~+0.19, a same-side carry ~+0.9. +0.3
    # admits both legitimate holds and rejects the carry; the old -0.2 made the
    # shooting pocket an illegal set.
    opposition_max: float = 0.3
    z_set: float = 0.40             # ball height above root that counts as the set position
    z_low: float = 0.105            # cradle height above root (bottom of the raise-reward ramp)
    set_sustain_steps: int = 5      # consecutive in-set steps (0.1 s) before the set latches
    # Release / throw detection
    release_dist: float = 0.35      # per-hand distance; BOTH hands beyond this = ball away
    rf_window: int = 5              # steps of (d_left - d_right) averaged for the right-hand finish
    # Retouch (second push) detection: any lower-arm body (palm/wrist/elbow)
    # back within retouch_dist of the ball after the follow-through grace
    # forfeits hit_bonus + ball_proximity. Must sit ABOVE the square palm-push
    # contact onset (ball r 0.125 + palm-heel offset ~0.09 = ~0.215): the old
    # 0.20 was below it and missed exactly the re-push it exists to catch.
    retouch_dist: float = 0.24
    retouch_grace_steps: int = 5    # 0.1 s follow-through grace after the release
    # Post-throw window and failure thresholds. The window must outlast the
    # real ballistic flight (a lofted 4 m shot flies >1 s), or the episode
    # times out before the ball arrives and legitimate hits are voided.
    stabilize_steps: int = 75       # post-throw steps before the clean timeout (1.5 s)
    tilt_threshold: float = 0.8     # |projected_gravity_xy|; ~0.8 -> ~53 deg lean = fail
    ball_ground_z: float = 0.25     # ball center below this without a legal throw = dropped
    step_xy_threshold: float = 0.10 # horizontal foot slide (m) that counts as a step (allows weight transfer)
    step_lift_threshold: float = 0.10  # foot lift (m) that counts as a step
    # Pre-set shaping time budget (steps; 0.02 s each). Full pay until 2.5 s,
    # linearly to zero by 3.5 s - camping in the cradle is finite by construction.
    budget_start_step: int = 125
    budget_end_step: int = 175
    # Target hit radius (also colors the debug marker green)
    hit_radius: float = 0.5


# --- 5. Rewards ------------------------------------------------------------- #
# Effective per-step pay = weight * raw * dt (dt = 0.02). Per-episode totals
# form an increasing ladder over the phases (INCREMENTALLY strict: from any
# state, the intended next transition is the argmax of what remains).
# UNITS: wandb/tensorboard logs Episode_Reward/<term> = per-episode effective
# total / max_episode_length_s (6.0) - dashboard values are 6x smaller than
# the ladder numbers below.
#   fumble ~0  <  best never-set camp ~9  >  set+drop ~ -1 (abandoned_set tax)
#   <  set+weak throw+stand ~17  <  near miss+stand ~23  <  two-hand hit ~29
#   <  right-hand-finish hit ~35
# (each rung includes the stuck_landing bonus eff 8.0; it is path-invariant
# across the accuracy rungs, so it widens throw-vs-camp without touching
# hit-vs-miss. Within EVERY rung, a wobbling/fallen finish trails a settled
# stand by ~9-13 units: bonus forfeited + remaining upright annuity + penalty.)
# No shortcut (swat, toss-aside, left-hand carry, camp, second push, or
# set->drop->restart farming) out-earns the real shot. set_bonus is kept SMALL
# (eff 2.0): at eff 5.0 the "reach the set, drop, restart" cycle out-rated the
# honest intermediate behaviors per wall-clock step, a sticky local optimum.
@configclass
class G1ThrowRewardCfg:
    # Pre-set shaping (dense from step 0, budget-limited)
    hold_ball = RewTerm(func=mdp.hold_ball, weight=1.5, params={"sigma": 0.25})
    raise_ball = RewTerm(func=mdp.raise_ball, weight=1.5, params={"carry_dist": 0.30})
    set_bonus = RewTerm(func=mdp.set_bonus, weight=100.0)  # one-time eff +2.0
    # Quit-after-set tax (one-time eff -3.0): terminating (not timing out) with
    # set_done & ~thrown - together with the small set_bonus this makes the
    # farm cycle net-negative (+2.0 - 1.0 - 3.0 < 0) while never touching an
    # episode that completes the throw.
    abandoned_set = RewTerm(func=mdp.abandoned_set, weight=-150.0)

    # Post-set, pre-release: net <= 0 (waiting bleeds; shooting is the only way up)
    shot_prep = RewTerm(func=mdp.shot_prep, weight=1.0, params={"sigma": 0.2, "hand_gap": 0.2})
    shoot_clock = RewTerm(func=mdp.shoot_clock, weight=-1.0)

    # The shot. throw_release is the ONLY ball-velocity payment (one-time,
    # eff <= 10); proximity scores the real flight; hit is form-scaled.
    throw_release = RewTerm(func=mdp.throw_release, weight=250.0)
    ball_proximity = RewTerm(func=mdp.ball_target_proximity, weight=8.0, params={"sigma": 0.6})
    hit_bonus = RewTerm(func=mdp.throw_hit_bonus, weight=300.0)  # one-time eff 4.2-6.0
    late_touch = RewTerm(func=mdp.late_touch, weight=-3.0)

    # Post-throw stabilization (gated by thrown - unlocked only by a legal shot)
    stabilize_upright = RewTerm(func=mdp.post_release_upright, weight=3.0)
    stabilize_base_motion = RewTerm(func=mdp.post_release_base_motion, weight=-1.0)
    stabilize_posture = RewTerm(
        func=mdp.post_release_posture,
        weight=-1.0,
        params={
            "grace_steps": 15,  # 0.3 s follow-through grace
            # Legs/waist only: the arms must stay free for the follow-through.
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*waist.*"]),
        },
    )
    # Bring the arms back to the default (cradle/ready) stance after the
    # follow-through. Without this, a raised static arm costs ~zero energy
    # while action_rate taxes the way down: an arms-up statue is a defended
    # local optimum. Soft weight + longer grace than the legs (0.5 s) so the
    # follow-through itself is never punished; worst-case flail (~-3 over the
    # window) stays well under the upright income (+4.4), so it can never
    # invert into a prefer-to-fall gradient.
    stabilize_arms = RewTerm(
        func=mdp.post_release_posture,
        weight=-0.5,
        params={
            "grace_steps": 25,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
            ),
        },
    )
    # One-time eff 8.0 on the clean countdown expiry, only if upright + base
    # quiet + both feet loaded: the terminal anchor for "stand properly", the
    # difference between surviving the recoil and sticking the landing.
    # Discounted to the release step (0.99^75 ~ 0.47) it is worth ~3.8 - on par
    # with the hit bonus after flight discounting, so the policy values the
    # landing about as much as the hit itself.
    stuck_landing = RewTerm(func=mdp.stuck_landing, weight=400.0)

    # Always-on. ball_body_contact prices out chest-wedging/forearm-cradling
    # (replaces the removed torso/pelvis/waist contact TERMINATIONS, which made
    # legal chest-height holds a reset lottery); the nominal cradle keeps the
    # ball 0.13 m clear of the chest so legal play pays zero.
    ball_body_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[".*torso.*", ".*waist.*", ".*pelvis.*", ".*head.*", ".*_shoulder_.*", ".*_elbow_.*"],
            ),
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    feet_on_ground = RewTerm(
        func=mdp.penalize_lifted_feet,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll.*"]), "max_height": 0.08},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)

    # Failure penalty (clean countdown timeouts excluded via is_terminated).
    # Kept SMALL on purpose: the old -200 made survival worth 13x a hit and
    # traded a 76%-hit-rate policy down to 0.15%. The real deterrent to falling
    # is forfeiting the ~24-33 units the rest of the episode pays.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)


# --- 6. Environment config --------------------------------------------------- #
@configclass
class G1ThrowFlatEnvCfg(BaseEnvCfg):
    scene: ThrowSceneCfg = ThrowSceneCfg()
    reward = G1ThrowRewardCfg()
    throw: ThrowCfg = ThrowCfg()

    def __post_init__(self):
        super().__post_init__()

        # Bake the V-cradle into the default pose: zero action = holding, the
        # joint-pos observations become deltas from the cradle, and the reset
        # event drops the ball straight into the pocket.
        self.scene.robot = G1_CFG.replace(
            init_state=G1_CFG.init_state.replace(joint_pos=HOLDING_JOINT_POS)
        )

        # Short episode: if the robot never throws it should reset before long.
        self.scene.max_episode_length_s = 6.0
        self.scene.terrain_type = "plane"

        # Fall indicators ONLY (head/hip/knee). Hands/arms touch the ball every
        # shot, and torso/pelvis/waist contact must NOT terminate either - the
        # ball is held at chest height, and a graze there was a reset lottery.
        # Chest contact is priced by the ball_body_contact penalty instead;
        # actual falls are caught by the tilt failure + these three.
        self.robot.terminate_contacts_body_names = [
            ".*head.*",
            ".*_hip_.*",
            ".*_knee_.*",
        ]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        # Deterministic, stable start: no base pose/velocity randomization and an
        # EXACT holding pose, so the ball begins truly cradled.
        # reset_joints_by_scale MULTIPLIES default_joint_pos by the sampled
        # range, so "exact default pose" is scale (1.0, 1.0) - a (0.0, 0.0)
        # range (the inherited base default, live in the failed run) silently
        # resets all 29 joints to ZERO: straight legs, arms off the cradle,
        # and feet origins captured on the wrong pose.
        # (Once training reaches a >50% hit rate, widen to ~(0.98, 1.02) and
        # add ~0.01 m ball-spawn jitter to robustify.)
        self.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.domain_rand.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        # No external pushes while it is trying to hold/shoot - a random impulse
        # mid-hold or mid-throw is not a perturbation this task should train
        # against (yet). Design choice, not an API workaround: in the installed
        # isaaclab 2.1.0, push_by_setting_velocity ADDS a sampled velocity delta
        # to the current root velocity (events.py:814-820), so if pushes are
        # later wanted for robustification, small ranges are safe to re-enable.
        # EventManager silently skips None terms, so removal is clean.
        self.domain_rand.events.push_robot = None

        # Drop the ball into the cradle on every reset (root-relative in z too).
        self.domain_rand.events.reset_ball = EventTerm(
            func=reset_ball_in_hands,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "forward_offset": 0.30,
                "height_rel": 0.115,
                "lateral": 0.0,
            },
        )


@configclass
class G1ThrowFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_throw_flat"
    wandb_project: str = "g1_throw_flat"
    # The failed 25k-iteration run peaked by ~iteration 2000, collapsed ~80% by
    # ~4500, and stayed dead for the remaining ~21k iterations. Stop long before
    # that; watch Episode_Reward/throw_release, hit_bonus and stuck_landing, and
    # stop on plateau. NOTE ON UNITS: the dashboard logs Episode_Reward/<term> =
    # per-episode effective total / max_episode_length_s (6.0), so the ladder
    # numbers in G1ThrowRewardCfg appear 6x smaller there. Expected healthy
    # plateaus in LOGGED units: throw_release ~1.2-1.7, hit_bonus ~0.7-1.0,
    # stuck_landing ~1.0-1.3.
    max_iterations: int = 8000
