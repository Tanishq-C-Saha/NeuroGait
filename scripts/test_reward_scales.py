"""CP5: Test reward scales before training.

Runs 100 steps with RANDOM actions and prints per-term reward statistics.
Use this BEFORE any real training to ensure no reward dominates.

Rule of thumb: all weighted reward terms should be within 1 order of magnitude
of each other. If one term is >10x any other, adjust weights in CP5RewardsCfg.

Usage:
    ~/isaac-sim/kit/python/bin/python3 scripts/test_reward_scales.py \\
      --task NeuroGait-Navigation-CP5-v0 \\
      --num_envs 256 \\
      --headless \\
      --enable_cameras

Expected output (example):
    Term                | Weight |   Mean   | Weighted Mean | Std
    --------------------|--------|----------|---------------|-------
    termination_penalty | -200.0 |  0.0000  |    0.0000     | 0.0000
    progress            |   1.0  |  0.0021  |    0.0021     | 0.0045
    heading             |   0.3  |  0.0830  |    0.0249     | 0.4820
    collision           |   2.0  | -0.0050  |   -0.0100     | 0.0704
    smoothness          |   0.1  | -0.3200  |   -0.0320     | 0.1800

    WARNING if |weighted mean| > 10× any other term.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CP5: Reward scale calibration test.")
parser.add_argument("--task",     type=str, required=True, help="Gym task id")
parser.add_argument("--num_envs", type=int, default=256,   help="Number of parallel envs")
parser.add_argument("--steps",    type=int, default=100,   help="Steps to collect stats")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────

import torch
import gymnasium as gym
import numpy as np

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    """Collect reward stats with random actions."""

    env_cfg.scene.num_envs = args_cli.num_envs

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = raw_env.unwrapped  # ManagerBasedRLEnv

    env.reset()

    # storage: {term_name: list of per-step mean values}
    reward_buffer: dict[str, list[float]] = {}

    print(f"\n[INFO] Collecting {args_cli.steps} steps with random actions "
          f"({args_cli.num_envs} envs)...")

    for step in range(args_cli.steps):
        # sample random action in [-1, 1]^3
        actions = torch.rand(args_cli.num_envs, 3, device=env.device) * 2.0 - 1.0

        # step and pull per-term rewards from the reward manager
        obs_dict, total_reward, terminated, truncated, info = env.step({"pre_trained_policy_action": actions})

        # Isaac Lab stores per-term rewards in env.reward_manager
        for term_name, term_cfg in env.reward_manager._term_cfgs.items():
            val = env.reward_manager.get_term_value(term_name)  # (num_envs,)
            if val is not None:
                mean_val = float(val.mean().item())
                if term_name not in reward_buffer:
                    reward_buffer[term_name] = []
                reward_buffer[term_name].append(mean_val)

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Term':<22} | {'Weight':>7} | {'Mean':>9} | {'Wtd Mean':>12} | {'Std':>8}")
    print("-" * 75)

    weighted_means = {}
    for term_name, values in reward_buffer.items():
        arr = np.array(values)
        mean_raw = float(np.mean(arr))
        std_raw  = float(np.std(arr))

        # get weight from reward manager config
        term_cfg = env.reward_manager._term_cfgs.get(term_name)
        weight   = float(term_cfg.weight) if term_cfg is not None else 1.0
        mean_wtd = mean_raw * weight

        weighted_means[term_name] = mean_wtd
        print(f"{term_name:<22} | {weight:>7.1f} | {mean_raw:>9.4f} | {mean_wtd:>12.4f} | {std_raw:>8.4f}")

    print("=" * 75)

    # ── balance check ─────────────────────────────────────────────────────────
    abs_vals = [abs(v) for v in weighted_means.values() if abs(v) > 1e-8]
    if abs_vals:
        max_v = max(abs_vals)
        min_v = min(abs_vals)
        ratio = max_v / max(min_v, 1e-8)
        if ratio > 10.0:
            worst = [k for k, v in weighted_means.items() if abs(v) == max_v][0]
            print(f"\n⚠  WARNING: '{worst}' is {ratio:.1f}× larger than the smallest "
                  f"term. Consider rebalancing weights.")
        else:
            print(f"\n✓  Reward terms are balanced (max ratio {ratio:.1f}×).")

    env.close()
    raw_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
