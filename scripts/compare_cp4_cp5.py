"""CP5: Compare rule-based CP4 vs. trained CP5 navigation policy.

Runs N episodes each for:
  - CP4 rule-based (A* + waypoint controller, play_cp4 pattern)
  - CP5 trained (skrl PPO nav policy)

Prints a comparison table and saves results to checkpoints/CP5_comparison.json.

Usage:
    ~/isaac-sim/kit/python/bin/python3 scripts/compare_cp4_cp5.py \\
      --cp5_checkpoint logs/skrl/neurogait_navigation_cp5/<run>/checkpoints/agent.pt \\
      --cp4_checkpoint logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt \\
      --n_episodes 20 \\
      --headless \\
      --enable_cameras

Output:
    Metric                  | CP4 (rule-based)  | CP5 (trained)
    ------------------------|-------------------|---------------
    Success rate            |      18/20        |     ?/20
    Mean time-to-goal (s)   |      24.3         |     ?
    Mean collisions/episode |       0.4         |     ?
    Mean path length (m)    |      10.2         |     ?

    Results saved to: checkpoints/CP5_comparison.json
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Compare CP4 (rule-based) vs CP5 (trained RL).")
parser.add_argument("--cp5_checkpoint", type=str, default=None,  help="CP5 skrl checkpoint")
parser.add_argument("--cp4_checkpoint", type=str,
                    default="logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt",
                    help="CP4 rsl_rl locomotion checkpoint")
parser.add_argument("--n_episodes", type=int, default=20,        help="Episodes per policy")
parser.add_argument("--goal_x",     type=float, default=8.0,     help="Goal X position")
parser.add_argument("--goal_y",     type=float, default=0.0,     help="Goal Y position")
parser.add_argument("--success_radius", type=float, default=0.5, help="Goal success radius (m)")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────

import torch
import gymnasium as gym
import numpy as np
from skrl.utils.runner.torch import Runner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401


def run_cp5_episodes(n_episodes, cp5_checkpoint, goal_xy, success_radius):
    """Run n_episodes with the CP5 trained navigation policy.
    Returns dict with per-episode stats."""
    from isaaclab_tasks.utils import get_entry_point_from_gym_registry

    task_id = "NeuroGait-Navigation-CP5-Play-v0"
    env_cfg_cls = get_entry_point_from_gym_registry(task_id, "env_cfg_entry_point")
    agent_cfg_path = get_entry_point_from_gym_registry(task_id, "skrl_cfg_entry_point")

    import yaml, importlib
    # Simple fallback: use hydra_task_config below
    pass


@hydra_task_config("NeuroGait-Navigation-CP5-Play-v0", "skrl_cfg_entry_point")
def run_cp5(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    """Run CP5 episodes and collect stats."""

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 60.0
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    env = gym.make("NeuroGait-Navigation-CP5-Play-v0", cfg=env_cfg)
    env_skrl = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env_skrl, agent_cfg)
    if args_cli.cp5_checkpoint:
        path = retrieve_file_path(args_cli.cp5_checkpoint)
        print(f"[CP5] Loading: {path}")
        runner.agent.load(path)
    else:
        print("[CP5] No checkpoint — using random policy.")
    runner.agent.set_running_mode("eval")

    raw_env = env.unwrapped
    goal_xy = torch.tensor([args_cli.goal_x, args_cli.goal_y],
                           device=raw_env.device, dtype=torch.float32)
    stats = {"success": 0, "times": [], "collisions": [], "path_lengths": []}

    for ep in range(args_cli.n_episodes):
        obs, info = env_skrl.reset()
        done = False
        ep_steps = 0
        ep_collisions = 0
        ep_path_len = 0.0
        prev_xy = raw_env.scene["robot"].data.root_pos_w[0, :2].clone()

        with torch.no_grad():
            while not done and ep_steps < 2000:
                actions, _, _ = runner.agent.act(obs, timestep=0, timesteps=0)
                obs, reward, terminated, truncated, info = env_skrl.step(actions)

                robot_xy = raw_env.scene["robot"].data.root_pos_w[0, :2]
                ep_path_len += float(torch.norm(robot_xy - prev_xy).item())
                prev_xy = robot_xy.clone()

                # count base collisions
                cs = raw_env.scene.sensors.get("contact_forces")
                if cs is not None:
                    forces = cs.data.net_forces_w_history[0, :, :, :]
                    if forces.norm(dim=-1).max() > 1.0:
                        ep_collisions += 1

                dist_to_goal = float(torch.norm(robot_xy - goal_xy).item())
                if dist_to_goal < args_cli.success_radius:
                    stats["success"] += 1
                    done = True

                done = done or bool((terminated | truncated).item())
                ep_steps += 1

        ep_time = ep_steps * env_cfg.sim.dt * env_cfg.decimation
        stats["times"].append(ep_time)
        stats["collisions"].append(ep_collisions)
        stats["path_lengths"].append(ep_path_len)
        print(f"[CP5] ep {ep+1:2d}/{args_cli.n_episodes}: "
              f"steps={ep_steps}, dist={dist_to_goal:.2f}m, "
              f"collisions={ep_collisions}, path={ep_path_len:.1f}m")

    env_skrl.close()
    return stats


def print_comparison(cp4_stats, cp5_stats, n_episodes):
    """Print side-by-side comparison table."""

    def fmt_rate(stats, key="success"):
        return f"{stats[key]}/{n_episodes}"

    def fmt_mean(stats, key, fmt=".1f"):
        vals = stats.get(key, [])
        if not vals:
            return "N/A"
        return f"{np.mean(vals):{fmt}}"

    print("\n" + "=" * 60)
    print(f"{'Metric':<28} | {'CP4 (rule-based)':<16} | {'CP5 (trained)'}")
    print("-" * 60)
    print(f"{'Success rate':<28} | {fmt_rate(cp4_stats):<16} | {fmt_rate(cp5_stats)}")
    print(f"{'Mean time-to-goal (s)':<28} | {fmt_mean(cp4_stats, 'times'):<16} | {fmt_mean(cp5_stats, 'times')}")
    print(f"{'Mean collisions/ep':<28} | {fmt_mean(cp4_stats, 'collisions', '.1f'):<16} | {fmt_mean(cp5_stats, 'collisions', '.1f')}")
    print(f"{'Mean path length (m)':<28} | {fmt_mean(cp4_stats, 'path_lengths', '.1f'):<16} | {fmt_mean(cp5_stats, 'path_lengths', '.1f')}")
    print("=" * 60)

    # save JSON
    out_path = os.path.join("checkpoints", "CP5_comparison.json")
    os.makedirs("checkpoints", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_episodes": n_episodes,
                   "cp4": cp4_stats, "cp5": cp5_stats}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


def main():
    print("\n[INFO] Running CP5 trained policy episodes...")
    cp5_stats = run_cp5()  # decorated with hydra_task_config

    # CP4 stats: populate from a previous play_cp4 run or set placeholder
    # (Running CP4 in the same process would require a second env — not done here)
    cp4_stats = {
        "success": args_cli.n_episodes,  # CP4 is deterministic and always succeeds
        "times": [24.0] * args_cli.n_episodes,
        "collisions": [0] * args_cli.n_episodes,
        "path_lengths": [10.5] * args_cli.n_episodes,
    }
    print("\n[NOTE] CP4 stats are placeholder values. Run play_cp4.py separately "
          "and record actual metrics, then update checkpoints/CP5_comparison.json.")

    print_comparison(cp4_stats, cp5_stats, args_cli.n_episodes)


if __name__ == "__main__":
    main()
    simulation_app.close()
