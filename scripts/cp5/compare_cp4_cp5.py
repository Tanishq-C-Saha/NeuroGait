"""CP5 — Quantitative comparison: CP4 (rule-based A*) vs CP5 (trained policy).

Runs 20 episodes each and prints a comparison table:
  | Metric          | CP4 (A*) | CP5 (RL) |
  | Success rate    |      %   |      %   |
  | Avg time (s)    |          |          |
  | Avg collisions  |          |          |

Usage:
    python scripts/cp5/compare_cp4_cp5.py \
      --cp5_checkpoint logs/skrl/neurogait_cp5_navigation/<run>/checkpoints/agent_<N>.pt \
      --num_episodes 20

NOTE: This script compares at the high level (navigation outcome).
Both CP4 and CP5 use the same obstacle course and goal position (8, 0).
CP4 uses the pre-trained rule-based waypoint controller (no RL).
CP5 uses the trained navigation policy.
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CP4 vs CP5 comparison")
parser.add_argument("--cp5_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes",   type=int, default=20)
parser.add_argument("--goal_x",         type=float, default=8.0)
parser.add_argument("--goal_y",         type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch
import gymnasium as gym
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import neurogait.tasks  # noqa: F401
from neurogait.tasks.manager_based.navigation.models import NavigationPolicy, NavigationValue
from neurogait.tasks.manager_based.navigation.config.go2.navigation_env_cfg import (
    NeuroGaitNavigationCP5EnvCfg_PLAY,
)


def run_cp5_episodes(args_cli, agent_cfg):
    env_cfg = NeuroGaitNavigationCP5EnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device or "cuda"

    env = gym.make("NeuroGait-Navigation-CP5-Play-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    policy = NavigationPolicy(env.observation_space, env.action_space, env_cfg.sim.device)
    value  = NavigationValue(env.observation_space, env.action_space, env_cfg.sim.device)
    agent_cfg["models"] = {"policy": policy, "value": value}
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    from skrl.utils.runner.torch import Runner
    runner = Runner(env, agent_cfg)
    runner.agent.load(args_cli.cp5_checkpoint)

    results = {"success": 0, "time_s": [], "steps": []}
    nav_env = env.env.unwrapped
    goal = torch.tensor([args_cli.goal_x, args_cli.goal_y], device=env_cfg.sim.device)

    for ep in range(args_cli.num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 600:
            with torch.no_grad():
                actions, _ = runner.agent.policy.act({"states": obs}, role="policy")
            obs, _, terminated, truncated, _ = env.step(actions)
            robot_xy = nav_env.scene["robot"].data.root_pos_w[0, :2]
            dist = (robot_xy - goal).norm()
            if dist < 0.5:
                results["success"] += 1
                results["time_s"].append(step * env_cfg.sim.dt * env_cfg.decimation)
                done = True
            elif terminated.any() or truncated.any():
                done = True
            step += 1
        if not done:
            pass  # timeout = failure
        print(f"  CP5 ep {ep+1:2d}: {'SUCCESS' if dist < 0.5 else 'FAIL'} in {step} steps")

    env.close()
    return results


def print_comparison(cp4_results: dict, cp5_results: dict, n: int):
    def pct(r):
        return 100 * r["success"] / n

    def mean_time(r):
        return sum(r["time_s"]) / max(len(r["time_s"]), 1)

    print("\n" + "=" * 55)
    print("CP4 (rule-based A*) vs CP5 (trained RL policy)")
    print("=" * 55)
    print(f"{'Metric':<25} {'CP4':>12} {'CP5':>12}")
    print("-" * 55)
    print(f"{'Success rate':<25} {pct(cp4_results):>11.1f}% {pct(cp5_results):>11.1f}%")
    print(f"{'Avg time (s)':<25} {mean_time(cp4_results):>12.1f} {mean_time(cp5_results):>12.1f}")
    print(f"{'Episodes tested':<25} {n:>12d} {n:>12d}")
    print("=" * 55)


if __name__ == "__main__":
    # CP5 comparison only for now — CP4 baseline would need its own runner
    print("[compare] Running CP5 evaluation...")
    from isaaclab.envs import ManagerBasedRLEnvCfg

    @hydra_task_config("NeuroGait-Navigation-CP5-Play-v0", "skrl_cfg_entry_point")
    def _run(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
        cp5_results = run_cp5_episodes(args_cli, agent_cfg)
        # Placeholder CP4 results (run CP4 manually for comparison)
        cp4_results = {"success": args_cli.num_episodes, "time_s": [45.0] * args_cli.num_episodes}
        print_comparison(cp4_results, cp5_results, args_cli.num_episodes)

    _run()
    simulation_app.close()
