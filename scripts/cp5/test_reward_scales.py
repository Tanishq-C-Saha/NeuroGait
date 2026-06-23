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
import sys
import os

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))

parser = argparse.ArgumentParser(description="CP5 reward scale test")
parser.add_argument("--task",     type=str, default="NeuroGait-Navigation-CP5-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps",    type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg

import neurogait.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Collect reward term sums over N steps
    reward_accum = {}
    for _ in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        actions = torch.rand(args_cli.num_envs, 3, device=env_cfg.sim.device) * 2 - 1
        # Scale heading to [-π, π]
        import math
        actions[:, 2] *= math.pi
        _, rewards, _, info = env.step(actions)

        for key, val in info.get("episode", {}).items():
            if key.startswith("reward/") or key.startswith("rew/"):
                name = key.split("/", 1)[-1]
                reward_accum[name] = reward_accum.get(name, 0.0) + float(val.mean())

    print("\n" + "=" * 60)
    print("CP5 Reward Scale Check  (256 envs × 100 random-action steps)")
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
