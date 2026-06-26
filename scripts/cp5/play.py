"""CP5 — Evaluate a trained navigation policy.

Loads the policy weights directly from a skrl checkpoint without using Runner.

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/cp5/play.py \
      --task NeuroGait-Navigation-CP5-Play-v0 \
      --checkpoint logs/skrl/neurogait_cp5_navigation/<run>/checkpoints/best_agent.pt

Output every 10 steps:
    Step   10 | CMD vx=+0.50 vy=+0.00 hdg=+0.78 | vel vx=+0.48 vy=+0.01 | POS (2.3, 0.5)
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CP5 navigation evaluation")
parser.add_argument("--task",       type=str, default="NeuroGait-Navigation-CP5-Play-v0")
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to skrl checkpoint (.pt file in checkpoints/)")
parser.add_argument("--max_steps",  type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import neurogait.tasks  # noqa: F401
from neurogait.tasks.manager_based.navigation.models import NavigationPolicy


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env_cfg.sim.device

    # ── Load policy ──────────────────────────────────────────────────────────
    policy = NavigationPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    print(f"[CP5-play] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)

    # skrl saves state dicts under per-model keys; print them once so it's
    # clear which key to use if the structure changes between skrl versions.
    print(f"[CP5-play] Checkpoint keys: {list(checkpoint.keys())}")

    # skrl PPO checkpoints store the policy state dict under "policy"
    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        # Fallback: checkpoint IS the state dict (some skrl versions save directly)
        policy.load_state_dict(checkpoint)

    policy.to(device)
    policy.eval()

    # ── Inference loop ───────────────────────────────────────────────────────
    obs, _ = env.reset()
    nav_env = env.env.unwrapped

    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break

        with torch.no_grad():
            # skrl passes obs under "observations" key (not "states")
            result = policy.act({"observations": obs}, role="policy")
            actions = result[0]

        obs, rewards, terminated, truncated, _ = env.step(actions)

        if step % 10 == 0:
            robot   = nav_env.scene["robot"]
            pos     = robot.data.root_pos_w[0].cpu()
            lin_vel = robot.data.root_lin_vel_b[0].cpu()
            cmd     = actions[0].cpu()
            print(
                f"Step {step:4d} | "
                f"CMD vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} hdg={cmd[2]:+.2f} | "
                f"vel vx={lin_vel[0]:+.2f} vy={lin_vel[1]:+.2f} | "
                f"POS ({pos[0]:.1f}, {pos[1]:.1f})"
            )

        if terminated.any() or truncated.any():
            print(f"[CP5-play] Episode ended at step {step}")
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
