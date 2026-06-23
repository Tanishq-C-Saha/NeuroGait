"""CP5 Step 0 — Verify reward term balance BEFORE training.

Runs 100 steps with random actions (256 envs, headless) and prints a table
of weighted reward magnitudes.  All terms should be within 10× of each other.
If any term dominates or is zero, adjust weights in CP5RewardsCfg before training.

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/cp5/test_reward_scales.py \
      --task NeuroGait-Navigation-CP5-v0 \
      --num_envs 256 \
      --headless
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CP5 reward scale test")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps",    type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import neurogait.tasks  # noqa: F401
from neurogait.tasks.manager_based.navigation.config.go2.navigation_env_cfg import (
    NeuroGaitNavigationCP5EnvCfg,
)


def main():
    env_cfg = NeuroGaitNavigationCP5EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or env_cfg.sim.device

    env = gym.make("NeuroGait-Navigation-CP5-v0", cfg=env_cfg)
    env.reset()

    reward_accum = {}
    for _ in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        actions = torch.rand(args_cli.num_envs, 3, device=env_cfg.sim.device) * 2 - 1
        actions[:, 2] *= math.pi   # scale heading to [-π, π]
        _, _, _, _, info = env.step(actions)

        for key, val in info.get("episode", {}).items():
            if key.startswith("reward/") or key.startswith("rew/"):
                name = key.split("/", 1)[-1]
                reward_accum[name] = reward_accum.get(name, 0.0) + float(val.mean())

    print("\n" + "=" * 60)
    print(f"CP5 Reward Scale Check  ({args_cli.num_envs} envs × {args_cli.steps} random-action steps)")
    print("=" * 60)
    print(f"{'Term':<35} {'Mean/step':>10}")
    print("-" * 60)
    total = 0.0
    for name in sorted(reward_accum):
        mean_val = reward_accum[name] / args_cli.steps
        print(f"  {name:<33} {mean_val:>10.4f}")
        total += mean_val
    print("-" * 60)
    print(f"  {'TOTAL':<33} {total:>10.4f}")
    print("=" * 60)
    print("\nAll weighted terms should be within 10× of each other.")
    print("If smoothness > 10× collision → reduce smoothness weight.")
    print("If velocity_toward_goal ≈ 0  → check obs order / camera.\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
