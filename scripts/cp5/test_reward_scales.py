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

    # Access reward manager directly — Isaac Lab only populates info["log"]
    # on episode resets, not every step.  _step_reward is filled every step:
    #   _step_reward[:, i] = func_output * weight  (already weighted)
    rm           = env.unwrapped.reward_manager
    term_names   = rm._term_names
    reward_accum = {name: 0.0 for name in term_names}

    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        actions = torch.rand(args_cli.num_envs, 3, device=env_cfg.sim.device) * 2 - 1
        actions[:, 2] *= math.pi   # scale heading to [-π, π]
        env.step(actions)

        for i, name in enumerate(term_names):
            reward_accum[name] += float(rm._step_reward[:, i].mean())

        # Print robot sanity every 20 steps
        if step % 20 == 0:
            pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2].cpu()
            print(f"  step {step:3d}  pos=({pos[0]:.2f}, {pos[1]:.2f})")

    print("\n" + "=" * 60)
    print(f"CP5 Reward Scale Check  ({args_cli.num_envs} envs × {args_cli.steps} steps)")
    print("=" * 60)
    print(f"  {'Term':<33} {'Mean/step':>10}  {'Total':>10}")
    print("-" * 60)
    grand_total = 0.0
    for name in term_names:
        mean_val  = reward_accum[name] / args_cli.steps
        total_val = reward_accum[name]
        print(f"  {name:<33} {mean_val:>10.4f}  {total_val:>10.4f}")
        grand_total += mean_val
    print("-" * 60)
    print(f"  {'TOTAL':<33} {grand_total:>10.4f}")
    print("=" * 60)
    print("\nAll weighted terms should be within 10× of each other.")
    print("If smoothness > 10× collision  → reduce smoothness weight.")
    print("If velocity_toward_goal ≈ 0    → check obs order / camera.\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
