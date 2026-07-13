# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Lightweight, NO-policy previewer for the throw task. It builds the environment
# and steps it with zero, small random, or SCRIPTED actions so you can validate
# the setup *before* spending time on training:
#
#   * zero   - hold the configured cradle pose. The ball must stay cradled
#              (d_left/d_right ~0.16, never released) indefinitely. If it falls
#              out, the holding pose / ball spawn need tuning.
#   * random - small random actions, eyeball general stability.
#   * script - drive a keyframed shot cycle through the env's state machine:
#              hold (0-2 s) -> raise both arms (2-3.5 s, the set should latch)
#              -> settle (0.5 s) -> fling arms apart (release; ``thrown`` should
#              latch and the 75-step countdown should end the episode as a clean
#              timeout). This validates every latch without a trained policy.
#
# Run WITHOUT --headless to watch it; with --headless read the printed
# diagnostics (hand distances, z_rel, set/released/thrown counts, resets).
#
# Examples:
#   python scripts/inspect_throw.py --task g1_throw_flat --num_envs 9
#   python scripts/inspect_throw.py --task g1_throw_flat --num_envs 4 --action script --headless --steps 300
import argparse

import torch
from isaaclab.app import AppLauncher

import utils.cli_args as cli_args  # isort: skip
from utils import task_registry

parser = argparse.ArgumentParser(description="Preview the throw task without a trained policy.")
parser.add_argument("--task", type=str, default="g1_throw_flat", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps (0 = run until window closed).")
parser.add_argument("--action", type=str, default="zero", choices=["zero", "random", "script"], help="Action source.")
parser.add_argument("--log_every", type=int, default=25, help="Print diagnostics every N steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from envs import *  # noqa: F401, F403  (registers the tasks)


# Scripted shot keyframes (per-env, driven by episode_length_buf so envs that
# reset early just restart their own cycle).
HOLD_END = 100     # steps 0-99: hold the cradle
RAISE_END = 175    # 100-174: raise both arms (shoulder pitch -0.05 -> -1.0)
SETTLE_END = 200   # 175-199: hold the set (the set latch needs 5 sustained steps)
FLING_LEN = 10     # 200-209: fling both arms apart -> both-hands-clear release
RAISE_PITCH = -1.0  # ball z_rel ~0.43 at this shoulder pitch (set line is 0.40)
FLING_ROLL = 1.2    # outward shoulder roll at the end of the fling


def build_script_actions(env):
    """Absolute joint targets for the current per-env cycle step, mapped back to
    policy actions (action = (q_des - default) / action_scale)."""
    joint_names = env.robot.data.joint_names
    idx = {name: i for i, name in enumerate(joint_names)}
    default = env.robot.data.default_joint_pos

    t = env.episode_length_buf.float()
    alpha_raise = torch.clamp((t - HOLD_END) / (RAISE_END - HOLD_END), 0.0, 1.0)
    alpha_fling = torch.clamp((t - SETTLE_END) / FLING_LEN, 0.0, 1.0)

    q_des = default.clone()
    for side in ("left", "right"):
        sp = idx[f"{side}_shoulder_pitch_joint"]
        q_des[:, sp] = default[:, sp] + alpha_raise * (RAISE_PITCH - default[:, sp])
        sr = idx[f"{side}_shoulder_roll_joint"]
        outward = FLING_ROLL if side == "left" else -FLING_ROLL
        q_des[:, sr] = default[:, sr] + alpha_fling * (outward - default[:, sr])

    return (q_des - default) / env.action_scale


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    # Clean, repeatable preview conditions.
    env_cfg.noise.add_noise = False
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.max_episode_length_s = 40.0  # don't time out while you watch
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    obs, _ = env.get_observations()
    num_actions = env.num_actions

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.action == "random":
                actions = (2.0 * torch.rand(env.num_envs, num_actions, device=env.device) - 1.0) * 0.5
            elif args_cli.action == "script":
                actions = build_script_actions(env)
            else:
                actions = torch.zeros(env.num_envs, num_actions, device=env.device)
            obs, reward, dones, _ = env.step(actions)

        if args_cli.log_every and step % args_cli.log_every == 0:
            ball = env.scene["ball"].data.root_pos_w
            n = env.num_envs
            print(
                f"[{step:5d}] "
                f"ball_z(mean)={ball[:, 2].mean():.2f}  "
                f"z_rel(mean)={env.ball_z_rel.mean():.2f}  "
                f"d_L/d_R(mean)={env.hand_ball_dist_l.mean():.2f}/{env.hand_ball_dist_r.mean():.2f}  "
                f"set={int(env.set_done.sum())}/{n}  "
                f"released={int(env.released.sum())}/{n}  "
                f"thrown={int(env.thrown.sum())}/{n}  "
                f"hit={int(env.target_reached.sum())}/{n}  "
                f"quality(mean|thrown)={env.release_quality[env.thrown].mean().item() if env.thrown.any() else 0.0:.2f}  "
                f"resets={int(dones.sum())}  "
                f"reward(mean)={reward.mean():.2f}"
            )

        step += 1
        if args_cli.steps and step >= args_cli.steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
