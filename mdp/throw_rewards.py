"""Reward functions for the G1 "shoot the basketball at a target" task.

Phase layout (latches owned by ``G1ThrowEnv`` and updated in ``check_reset``,
which runs immediately before the reward manager every step):

    pre-set      hold_ball + raise_ball        dense, budget-limited shaping
    just_set     set_bonus                     one-time
    set->throw   shot_prep - shoot_clock       right-on/left-off guidance, net <= 0
    just_thrown  throw_release                 one-time, ballistic quality x right-hand finish
    thrown       ball_proximity, hit_bonus,    flight scoring + post-shot behavior
                 late_touch, stabilization

Design rules (from the forensic analysis of the failed swat run):

* Every ball->target payment requires ``env.thrown`` - a release from a legal
  two-hand set. A swat, fumble or early release earns exactly nothing from the
  throw terms, so the intended hold -> raise -> right-hand shot strictly
  dominates every shortcut.
* The ONLY ball-velocity-dependent payment is the one-time ``throw_release``.
  Nothing pays per-step ball speed: that ungated annuity was 87.9% of the old
  run's income and is precisely what taught the back-of-the-hand swat.
* Pre-set shaping is multiplied by ``env.pre_set_budget()`` (1.0 -> 0.0 between
  budget_start_step and budget_end_step), so cradle-camping income is finite
  and the shot is always worth more than waiting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from envs.base.throw_env import G1ThrowEnv


# --------------------------------------------------------------------------- #
# Pre-set shaping (budget-limited, dense from step 0)
# --------------------------------------------------------------------------- #
def hold_ball(env: "G1ThrowEnv", sigma: float = 0.25) -> torch.Tensor:
    """Keep BOTH hands on the ball while carrying it up to the set.

    Uses the worse of the two hand distances, so parking one hand near the ball
    and ignoring the other pays almost nothing. Budget-limited and switched off
    once the set is reached (shot_prep takes over from there).
    """
    gate = (~env.released) & (~env.set_done)
    d_max = torch.maximum(env.hand_ball_dist_l, env.hand_ball_dist_r)
    return gate.float() * torch.exp(-torch.square(d_max / sigma)) * env.pre_set_budget()


def raise_ball(env: "G1ThrowEnv", carry_dist: float = 0.30) -> torch.Tensor:
    """Reward ball height - but ONLY while both hands are carrying it.

    Height fraction ramps from the cradle height (``z_low``) to the set line
    (``z_set``), both relative to the robot root. Tossing the ball up without
    hands on it pays zero (the ``carry_dist`` gate), so this cannot be farmed
    by juggling.
    """
    gate = (~env.released) & (~env.set_done)
    d_max = torch.maximum(env.hand_ball_dist_l, env.hand_ball_dist_r)
    carried = (d_max < carry_dist).float()
    height_frac = torch.clamp((env.ball_z_rel - env.z_low) / (env.z_set - env.z_low), 0.0, 1.0)
    return gate.float() * carried * height_frac * env.pre_set_budget()


def set_bonus(env: "G1ThrowEnv") -> torch.Tensor:
    """One-time bonus the step the two-hand set latches (top of the dense ramp)."""
    return env.just_set.float()


# --------------------------------------------------------------------------- #
# Post-set, pre-release (net <= 0: waiting bleeds, shooting is the only way up)
# --------------------------------------------------------------------------- #
def shot_prep(env: "G1ThrowEnv", sigma: float = 0.2, hand_gap: float = 0.2) -> torch.Tensor:
    """Dense right-hand-on / left-hand-off guidance between set and release.

    Pays for keeping the RIGHT hand at the ball while the LEFT hand backs away,
    i.e. the transition from the two-hand set into a one-hand shooting grip.
    """
    gate = env.set_done & (~env.released)
    right_on = torch.exp(-torch.square(env.hand_ball_dist_r / sigma))
    left_off = torch.clamp((env.hand_ball_dist_l - env.hand_ball_dist_r) / hand_gap, 0.0, 1.0)
    return gate.float() * right_on * left_off


def shoot_clock(env: "G1ThrowEnv") -> torch.Tensor:
    """Penalty (negative weight) per step spent holding the set without shooting."""
    return (env.set_done & (~env.released)).float()


# --------------------------------------------------------------------------- #
# The shot
# --------------------------------------------------------------------------- #
def throw_release(env: "G1ThrowEnv") -> torch.Tensor:
    """One-time payment at the throw: ballistic accuracy x right-hand finish.

    ``release_quality`` is captured by the env on the ``just_thrown`` step from
    the release-instant ball state (see throw_env.check_reset). This is the
    single ball-velocity-dependent term in the task.
    """
    return env.just_thrown.float() * env.release_quality


def ball_target_proximity(env: "G1ThrowEnv", sigma: float = 0.6) -> torch.Tensor:
    """Score of the REAL flight after a legal throw, off the running closest
    approach. Deliberately NOT zeroed once the target is reached: a hit locks
    the best distance below the hit radius and keeps paying near-max for the
    rest of the stabilize window, so a hit is never worth less than a near miss.
    Forfeited entirely by a retouch: a guided second push must not be paid.
    """
    return (env.thrown & (~env.retouched)).float() * torch.exp(
        -torch.square(env.best_ball_target_dist / sigma)
    )


def throw_hit_bonus(env: "G1ThrowEnv") -> torch.Tensor:
    """One-time bonus when the real ball reaches the target, scaled by shot
    form: a right-hand finish (release_rf ~1) earns up to ~40% more than a
    two-hand push (release_rf ~0.5)."""
    return env.just_reached.float() * (0.5 + 0.5 * env.release_rf)


def late_touch(env: "G1ThrowEnv") -> torch.Tensor:
    """Penalty (negative weight) for chasing/tipping the ball after the shot.

    A short grace window right at the release (``retouch_grace_steps``) lets the
    follow-through brush the ball without cost; after that, any part of either
    lower arm back on the ball is a second push (``env.arm_ball_dist`` is the
    min distance over palms + wrists + elbows, cached by check_reset). The
    per-step tax here is a gradient hint - the real deterrent is the
    ``retouched`` latch (same predicate, owned by the env), which forfeits
    hit_bonus and ball_proximity outright."""
    return (
        env.thrown
        & (env.stabilize_counter > env.retouch_grace_steps)
        & (env.arm_ball_dist < env.retouch_dist)
    ).float()


def abandoned_set(env: "G1ThrowEnv") -> torch.Tensor:
    """Penalty (negative weight) for ENDING the episode in failure after
    reaching the set without ever throwing.

    Closes the set-farming cycle: raise -> bank set_bonus -> deliberately drop
    or step -> cheap reset -> repeat. With only the small generic termination
    penalty, that loop out-rates the honest intermediate behaviors and is a
    sticky local optimum. This tax makes any set-without-throw termination
    strictly unprofitable while leaving the honest path untouched (once
    ``thrown`` latches, this can never fire; clean countdown timeouts are
    excluded the same way as in ``is_terminated``)."""
    return (env.reset_buf & (~env.time_out_buf) & env.set_done & (~env.thrown)).float()


# --------------------------------------------------------------------------- #
# Post-throw stabilization (all gated by env.thrown - an early release unlocks
# nothing, so tossing the ball aside cannot farm these)
# --------------------------------------------------------------------------- #
def post_release_upright(env: "G1ThrowEnv") -> torch.Tensor:
    """Reward staying vertical after the throw. ``projected_gravity_b`` is
    ~(0, 0, -1) when upright; its xy-norm is sin(tilt). Capped per episode by
    the stabilize window (75 steps -> ~4.5 total at weight 3), so it is never
    worth more than shooting well."""
    proj_g = env.robot.data.projected_gravity_b
    tilt = torch.norm(proj_g[:, :2], dim=-1)
    return env.thrown.float() * (1.0 - tilt)


def post_release_base_motion(env: "G1ThrowEnv") -> torch.Tensor:
    """Penalty (negative weight) on base lin+ang speed after the throw."""
    lin = torch.norm(env.robot.data.root_lin_vel_b, dim=-1)
    ang = torch.norm(env.robot.data.root_ang_vel_b, dim=-1)
    return env.thrown.float() * (lin + ang)


def post_release_posture(
    env: "G1ThrowEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grace_steps: int = 15,
) -> torch.Tensor:
    """Penalty (negative weight): drift of the selected joints from their
    defaults, after a follow-through grace window. Used twice in the config
    with different joint sets and graces: legs/waist (grace 15) for the
    balance recovery, and arms (grace 25, softer weight) so the follow-through
    finishes freely but the arms then come back down to the ready stance -
    without the arm term, a raised static arm costs ~zero energy while
    action_rate taxes the way down, making an arms-up statue a defended local
    optimum."""
    asset = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return (env.thrown & (env.stabilize_counter > grace_steps)).float() * deviation


def stuck_landing(
    env: "G1ThrowEnv",
    tilt_max: float = 0.15,
    lin_vel_max: float = 0.15,
    ang_vel_max: float = 0.5,
) -> torch.Tensor:
    """One-time bonus on the clean countdown-expiry step for finishing the shot
    STANDING PROPERLY: upright (sin-tilt < tilt_max, vs the 0.8 kill line),
    base quiet (lin/ang speed below thresholds), and BOTH feet loaded.

    This is the terminal anchor of the stabilization objective: the per-step
    upright annuity distinguishes falling from surviving, but pays a wobbling
    survivor almost as much as a settled one, and the clean-timeout bootstrap
    values both finishes identically. This bonus is the difference between
    "didn't fall" and "stuck the landing".

    Exploit-safe: requires ``thrown`` (camping/drops never reach it), fires
    exactly once (the counter reaches stabilize_steps only on the terminal
    step, and rewards run before the reset), and is path-invariant across
    accuracy rungs (hits and misses both earn it), so it cannot make a weak
    throw out-earn an accurate one."""
    done = env.thrown & (env.stabilize_counter >= env.stabilize_steps)
    proj_g = env.robot.data.projected_gravity_b
    upright = torch.norm(proj_g[:, :2], dim=-1) < tilt_max
    slow = (torch.norm(env.robot.data.root_lin_vel_b, dim=-1) < lin_vel_max) & (
        torch.norm(env.robot.data.root_ang_vel_b, dim=-1) < ang_vel_max
    )
    forces = env.contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.norm(forces[:, :, env.feet_cfg.body_ids], dim=-1).max(dim=1).values > 1.0
    )
    return (done & upright & slow & feet_contact.all(dim=1)).float()
