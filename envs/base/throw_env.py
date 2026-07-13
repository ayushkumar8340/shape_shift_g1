"""Basketball-shot environment for the Unitree G1.

Task: start cradling the basketball in BOTH hands in front of the chest, raise
it into a "set" position above ``z_set``, then shoot it at the target with the
RIGHT hand (the left hand peels off as the support hand) and stay balanced
afterwards - the way a basketball player shoots a jump shot from a standstill.

The env owns a small latched state machine, updated once per step at the top of
``check_reset``. ``BaseEnv.step()`` calls ``check_reset`` right before the
reward manager, so the same step's rewards always see fresh latches:

* ``set_done`` - latched once BOTH hands cradled the ball (per-hand distance
  < ``hold_dist`` AND hands on opposite sides of the ball) above ``z_set`` for
  ``set_sustain_steps`` consecutive steps. This is the "legal shot" gate:
  nothing ball-to-target ever pays without it, which is what kills the old
  back-of-the-hand swat exploit.
* ``released`` - latched once BOTH hands are farther than ``release_dist``
  from the ball. Requiring both hands makes a left-hand carry impossible.
  A release without ``set_done`` earns nothing and ends as a ball-drop
  failure; it can never be upgraded to a throw.
* ``thrown``  - released while ``set_done``: a legal shot. On this transition
  we capture (a) the right-hand-finish factor ``release_rf`` from a short
  history of (d_left - d_right), so a shot finished by the right hand pays
  more than a symmetric two-hand push, and (b) ``release_quality``, the
  closest approach of the analytic ballistic arc of the release-instant ball
  state to the target - the ONLY ball-velocity-dependent payment in the task,
  paid exactly once. History entries only count while the right palm is ON
  the ball (d_right < hold_dist AND measured contact force): proximity alone
  can be laundered (park the right palm near the ball, shove with the left
  arm), and a two-hand shove could launder credit by letting the right hand
  chase the departing ball for a few steps.
* ``target_reached`` - the real (simulated) ball got within ``hit_radius`` of
  the target after the throw, tracked via the running closest approach.
* ``retouched`` - any lower-arm body (palm/wrist/elbow) came back onto the
  ball after the follow-through grace window. Latching this voids the hit
  bonus and proximity pay, so a weak release followed by a guided second push
  can never score.

Post-throw, a ``stabilize_steps`` countdown gives the robot time to absorb the
recoil; when it elapses the episode ends as a CLEAN TIMEOUT (both reset and
time-out flags), so rsl-rl bootstraps the value function and no termination
penalty is paid for finishing the job.

Failure terminations (with penalty): ball drops before a legal throw, torso
tilts past ``tilt_threshold``, a foot slides/lifts more than the step
thresholds, or a head/hip/knee contact fires (falling).
"""

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from envs.base.base_env import BaseEnv
from mdp.command_gen import ThrowCommand, ThrowCommandCfg
from utils.env_utils.scene import SceneCfg
from envs.g1.g1_throw_config import BASKETBALL_CFG


class G1ThrowEnv(BaseEnv):
    def __init__(self, cfg, headless):
        # Register this task's basketball into the shared scene wrapper right
        # before the scene is built. Done here (not at import) so it cannot clash
        # with the dribble task, which sets SceneCfg.ball to its own ball config.
        SceneCfg.ball = BASKETBALL_CFG
        super().__init__(cfg, headless)

    # --- command generator ------------------------------------------------- #
    def _create_command_generator(self):
        self.command_cfg = ThrowCommandCfg()
        self.command_generator = ThrowCommand(self.command_cfg, self)

    # --- buffers ----------------------------------------------------------- #
    def init_buffers(self):
        # Everything in this block must exist BEFORE super().init_buffers():
        # building the observation buffer evaluates ``command_generator.command``,
        # which reads the latches, the stabilize counter and the hand body ids.
        tcfg = self.cfg.throw
        self.hold_dist = tcfg.hold_dist
        self.opposition_max = tcfg.opposition_max
        self.z_set = tcfg.z_set
        self.z_low = tcfg.z_low
        self.set_sustain_steps = tcfg.set_sustain_steps
        self.release_dist = tcfg.release_dist
        self.rf_window = tcfg.rf_window
        self.stabilize_steps = tcfg.stabilize_steps
        self.tilt_threshold = tcfg.tilt_threshold
        self.ball_ground_z = tcfg.ball_ground_z
        self.step_xy_threshold = tcfg.step_xy_threshold
        self.step_lift_threshold = tcfg.step_lift_threshold
        self.budget_start_step = tcfg.budget_start_step
        self.budget_end_step = tcfg.budget_end_step
        self.hit_radius = tcfg.hit_radius
        self.retouch_dist = tcfg.retouch_dist
        self.retouch_grace_steps = tcfg.retouch_grace_steps

        # find_bodies returns ids in ASSET order regardless of query order, so
        # resolve each hand separately to keep the left/right identity straight.
        self.left_hand_id = self.robot.find_bodies(".*left_rubber_hand.*")[0][0]
        self.right_hand_id = self.robot.find_bodies(".*right_rubber_hand.*")[0][0]
        feet_ids, _ = self.robot.find_bodies(self.cfg.robot.feet_body_names)
        self.feet_body_ids = feet_ids
        # Retouch (second push) is checked against the WHOLE lower arm, not just
        # the palm origins: a square palm re-push contacts the ball at ~0.215 m
        # from the palm-heel origin, and a forearm/elbow nudge never moves the
        # palm distance at all - palm-only retouch detection misses exactly the
        # pushes it exists to catch. Order is irrelevant (min over all).
        touch_ids, _ = self.robot.find_bodies([".*_rubber_hand.*", ".*_wrist_.*", ".*_elbow_.*"])
        self.arm_touch_body_ids = touch_ids

        # Contact-sensor indices for the obs-only hand contact bits. The sensor
        # has its own body ordering - resolve through the scene, not the robot.
        left_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*left_rubber_hand.*")
        left_contact_cfg.resolve(self.scene)
        self.left_hand_contact_id = left_contact_cfg.body_ids[0]
        right_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*right_rubber_hand.*")
        right_contact_cfg.resolve(self.scene)
        self.right_hand_contact_id = right_contact_cfg.body_ids[0]

        n = self.num_envs
        # Latches (monotone within an episode)
        self.released = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.set_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.thrown = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.target_reached = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.retouched = torch.zeros(n, dtype=torch.bool, device=self.device)
        # One-step transition flags (pay the one-time bonuses)
        self.just_set = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.just_thrown = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.just_reached = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Counters / captured-at-release quantities
        self.set_counter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.stabilize_counter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.release_rf = torch.zeros(n, device=self.device)
        self.release_quality = torch.zeros(n, device=self.device)
        # Rolling window of (d_left - d_right), frozen at release: the
        # right-hand-finish evidence. Cradle value is ~0 -> rf factor 0.5.
        self.dlr_hist = torch.zeros(n, self.rf_window, device=self.device)
        self.best_ball_target_dist = torch.full((n,), 1e6, device=self.device)
        # Per-step caches written by check_reset, read by the reward functions.
        self.hand_ball_dist_l = torch.zeros(n, device=self.device)
        self.hand_ball_dist_r = torch.zeros(n, device=self.device)
        self.ball_z_rel = torch.zeros(n, device=self.device)
        self.arm_ball_dist = torch.full((n,), 1e6, device=self.device)
        # Initial world-frame foot positions, captured each reset, for step detection.
        self.initial_feet_pos_w = torch.zeros(n, len(feet_ids), 3, device=self.device)
        # Analytic ballistic grid for the release-quality capture: 200 samples
        # over 2 s of vacuum flight (10 ms spacing). A 50-sample grid over-reports
        # the closest approach by up to v*dt/2 ~ 0.14 m at 7 m/s - 29% of the
        # 0.5 m accuracy Gaussian's sigma; at 10 ms the error is ~0.035 m.
        self._ballistic_t = torch.linspace(0.0, 2.0, 200, device=self.device).view(1, -1, 1)
        self._ballistic_g = torch.tensor([0.0, 0.0, -9.81], device=self.device).view(1, 1, 3)

        super().init_buffers()

    # --- rendering / debug ------------------------------------------------- #
    def step(self, actions):
        out = super().step(actions)
        if not self.headless and hasattr(self, "draw_interface"):
            self._draw_debug_vis()
        return out

    def _draw_debug_vis(self):
        """Draw only the target location marker (red, turns green once hit).

        Deliberately no ball->target line: there is no intended trajectory. The
        policy must discover the shooting arc on its own from the rewards, so we
        show *where* the target is and nothing about *how* to reach it.
        """
        self.draw_interface.clear_points()

        target = self.command_generator.target_pos_w
        target_colors = [
            (0.0, 1.0, 0.0, 1.0) if hit else (1.0, 0.0, 0.0, 1.0)
            for hit in self.target_reached.tolist()
        ]
        self.draw_interface.draw_points(target.tolist(), target_colors, [12.0] * self.num_envs)

    # --- reset ------------------------------------------------------------- #
    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        # super().reset() has already written the holding pose + ball to sim and
        # called sim.forward(), so body_pos_w now reflects the fresh stance.
        self.released[env_ids] = False
        self.set_done[env_ids] = False
        self.thrown[env_ids] = False
        self.target_reached[env_ids] = False
        self.retouched[env_ids] = False
        self.just_set[env_ids] = False
        self.just_thrown[env_ids] = False
        self.just_reached[env_ids] = False
        self.set_counter[env_ids] = 0
        self.stabilize_counter[env_ids] = 0
        self.release_rf[env_ids] = 0.0
        self.release_quality[env_ids] = 0.0
        self.dlr_hist[env_ids] = 0.0
        self.best_ball_target_dist[env_ids] = 1e6
        feet_pos_w = self.robot.data.body_pos_w[:, self.feet_body_ids, :]
        self.initial_feet_pos_w[env_ids] = feet_pos_w[env_ids]

    # --- reward helpers ----------------------------------------------------- #
    def pre_set_budget(self) -> torch.Tensor:
        """Time budget multiplying the pre-set shaping terms: 1.0 until
        ``budget_start_step``, linearly down to 0.0 at ``budget_end_step``.
        Makes cradle-camping income finite by construction."""
        denom = float(max(self.budget_end_step - self.budget_start_step, 1))
        elapsed = (self.episode_length_buf - self.budget_start_step).float()
        return torch.clamp(1.0 - elapsed / denom, 0.0, 1.0)

    # --- termination + task state machine ---------------------------------- #
    def check_reset(self):
        reset_buf, time_out_buf = super().check_reset()

        ball = self.scene["ball"]
        ball_pos = ball.data.root_pos_w
        ball_vel = ball.data.root_lin_vel_w
        lh = self.robot.data.body_pos_w[:, self.left_hand_id]
        rh = self.robot.data.body_pos_w[:, self.right_hand_id]

        d_l = torch.norm(ball_pos - lh, dim=-1)
        d_r = torch.norm(ball_pos - rh, dim=-1)
        # Cache for the reward functions (they run right after this).
        self.hand_ball_dist_l = d_l
        self.hand_ball_dist_r = d_r
        self.ball_z_rel = ball_pos[:, 2] - self.robot.data.root_pos_w[:, 2]

        # 1) Fresh release condition FIRST, so the rf window below can freeze on
        #    the release-edge step itself. Freezing on the stale latch used to
        #    roll one guaranteed zero into the window on the edge step (the ball
        #    is already past hold_dist there), diluting the right-hand-finish
        #    signal by ~20% for marginal shots.
        ball_away = (d_l > self.release_dist) & (d_r > self.release_dist)
        release_edge = ball_away & (~self.released)

        # 2) Right-hand-finish history: roll in (d_left - d_right) while the
        #    ball is still in hand. Entries only count while the RIGHT palm is
        #    within hold_dist AND registering real contact force: proximity
        #    alone can be laundered (park the right palm 0.2 m off the ball and
        #    shove with the left forearm -> d_l - d_r large -> fake rf 1.0).
        #    Requiring measured right-palm force means full right-hand credit is
        #    only earned by a hand that is actually driving the ball. Contact
        #    flicker only zeroes single entries (never flips sign) and the
        #    window mean absorbs it; a genuine finishing push carries tens of
        #    newtons, far above the 1 N threshold.
        forces = self.contact_sensor.data.net_forces_w_history
        right_touching = (
            torch.norm(forces[:, :, self.right_hand_contact_id], dim=-1).max(dim=1).values > 1.0
        )
        entry = torch.where(
            (d_r < self.hold_dist) & right_touching, d_l - d_r, torch.zeros_like(d_r)
        )
        rolled = torch.cat([self.dlr_hist[:, 1:], entry.unsqueeze(1)], dim=1)
        self.dlr_hist = torch.where(
            (self.released | ball_away).unsqueeze(1), self.dlr_hist, rolled
        )

        # 3) SET latch: both hands on the ball, not bunched on the same side,
        #    above z_set, sustained. Opposition uses hand->ball unit vectors:
        #    the side-side cradle scores dot ~ -0.87 and a natural shooting
        #    pocket (right hand under-behind, left guiding on the side) scores
        #    dot ~ +0.19, while a same-side carry scores ~ +0.9. The threshold
        #    (+0.3) admits both legitimate two-hand holds and still rejects the
        #    carry - the old -0.2 outlawed the shooting pocket entirely.
        u_l = (ball_pos - lh) / (d_l.unsqueeze(-1) + 1e-6)
        u_r = (ball_pos - rh) / (d_r.unsqueeze(-1) + 1e-6)
        opposed = torch.sum(u_l * u_r, dim=-1) < self.opposition_max
        in_set = (
            (~self.released)
            & (d_l < self.hold_dist)
            & (d_r < self.hold_dist)
            & opposed
            & (self.ball_z_rel > self.z_set)
        )
        self.set_counter = torch.where(in_set, self.set_counter + 1, torch.zeros_like(self.set_counter))
        self.just_set = (self.set_counter >= self.set_sustain_steps) & (~self.set_done)
        self.set_done |= self.just_set

        # 4) RELEASE latch: BOTH hands clear of the ball (kills left-hand carries).
        self.released |= ball_away

        # 4) THROWN: a release from a legal set. An early release latches
        #    ``released`` but never ``thrown`` - it earns nothing downstream and
        #    ends as a ball-drop failure when the ball hits the floor.
        self.just_thrown = release_edge & self.set_done
        self.thrown |= self.just_thrown

        # 5) Right-hand-finish factor, captured at the throw from the frozen
        #    window mean: left hand off >=0.1 s early -> ~1.0, symmetric
        #    two-hand push -> ~0.5, left-hand-last -> 0.4 floor. (Numbering of
        #    the remaining steps is historical; order is what matters.)
        rf = torch.clamp(0.5 + self.dlr_hist.mean(dim=1) / 0.2, min=0.4, max=1.0)
        self.release_rf = torch.where(self.just_thrown, rf, self.release_rf)

        # 6) Release quality, captured at the throw: closest approach of the
        #    analytic ballistic arc (vacuum) of the release-instant ball state.
        #    Linear term gives a constant accuracy gradient out to 4 m; the
        #    Gaussian sharpens it near the target. Scaled by the right-hand
        #    finish, this is the single one-time ball-velocity payment.
        traj = (
            ball_pos.unsqueeze(1)
            + ball_vel.unsqueeze(1) * self._ballistic_t
            + 0.5 * self._ballistic_g * self._ballistic_t.square()
        )
        target = self.command_generator.target_pos_w
        miss = torch.norm(traj - target.unsqueeze(1), dim=-1).min(dim=1).values
        quality = self.release_rf * (
            torch.clamp(1.0 - miss / 4.0, 0.0, 1.0) + torch.exp(-torch.square(miss / 0.5))
        )
        self.release_quality = torch.where(self.just_thrown, quality, self.release_quality)

        # 7) Post-throw stabilization countdown -> ends the episode as a CLEAN
        #    timeout (both flags set; is_terminated stays False, rsl-rl bootstraps).
        self.stabilize_counter += self.thrown.long()
        countdown_done = self.stabilize_counter >= self.stabilize_steps

        # 8) Retouch latch: any part of either lower arm back on the ball after
        #    the follow-through grace is a second push. Latching it forfeits the
        #    hit bonus and the proximity annuity (both check ~retouched), so a
        #    "fake release then guided swat" earns nothing from the flight - the
        #    late_touch per-step tax alone would be ~50x too cheap a deterrent.
        #    Measured over palms + wrists + elbows (not palms alone: a square
        #    palm re-push contacts at ~0.215 m palm-origin distance and a
        #    forearm nudge never moves the palm distance at all), against a
        #    threshold ABOVE the palm-push contact onset. No false positives:
        #    after the grace the ball is already >0.4 m away and receding at
        #    3-6 m/s - only a deliberate chase re-enters 0.24 m.
        arm_pos = self.robot.data.body_pos_w[:, self.arm_touch_body_ids, :]
        self.arm_ball_dist = torch.norm(ball_pos.unsqueeze(1) - arm_pos, dim=-1).min(dim=1).values
        self.retouched |= (
            self.thrown
            & (self.stabilize_counter > self.retouch_grace_steps)
            & (self.arm_ball_dist < self.retouch_dist)
        )

        # 9) Closest approach of the REAL ball after a legal throw (running min,
        #    so a ball that whips through the target zone keeps its credit;
        #    frozen once retouched so a second push cannot improve it), and the
        #    one-time hit latch.
        target_dist = torch.norm(ball_pos - target, dim=-1)
        self.best_ball_target_dist = torch.where(
            self.thrown & (~self.retouched),
            torch.minimum(self.best_ball_target_dist, target_dist),
            self.best_ball_target_dist,
        )
        reached = self.thrown & (~self.retouched) & (self.best_ball_target_dist < self.hit_radius)
        self.just_reached = reached & (~self.target_reached)
        self.target_reached |= reached

        # 10) Failure: ball on the floor without a legal throw. Covers both a
        #     fumble out of the cradle and an early (pre-set) release.
        ball_dropped = (ball_pos[:, 2] < self.ball_ground_z) & (~self.thrown)

        # 11) Failure: robot tilted too far from vertical.
        proj_g = self.robot.data.projected_gravity_b
        tilted = torch.norm(proj_g[:, :2], dim=-1) > self.tilt_threshold

        # 12) Failure: robot took a step (a foot slid or lifted from its origin).
        feet_pos_w = self.robot.data.body_pos_w[:, self.feet_body_ids, :]
        xy_disp = torch.norm(feet_pos_w[:, :, :2] - self.initial_feet_pos_w[:, :, :2], dim=-1)
        z_lift = feet_pos_w[:, :, 2] - self.initial_feet_pos_w[:, :, 2]
        stepped = torch.any(xy_disp > self.step_xy_threshold, dim=1) | torch.any(
            z_lift > self.step_lift_threshold, dim=1
        )

        # Combine. countdown_done is a clean finish (timeout), the rest are failures.
        time_out_buf = time_out_buf | countdown_done
        reset_buf = reset_buf | ball_dropped | tilted | stepped | countdown_done

        return reset_buf, time_out_buf
