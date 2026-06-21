"""CP5: Evaluate a trained navigation policy.

Loads a skrl PPO checkpoint and runs the trained navigation policy.
The nav policy outputs velocity commands → PreTrainedPolicyAction →
frozen locomotion policy → joint targets → physics (all handled by Isaac Lab).

Usage:
    ~/isaac-sim/kit/python/bin/python3 scripts/play_cp5.py \\
      --task NeuroGait-Navigation-CP5-Play-v0 \\
      --num_envs 1 \\
      --enable_cameras \\
      --checkpoint logs/skrl/neurogait_navigation_cp5/<run>/checkpoints/agent.pt

    # Without a checkpoint (random policy — for env smoke test):
    ~/isaac-sim/kit/python/bin/python3 scripts/play_cp5.py \\
      --task NeuroGait-Navigation-CP5-Play-v0 \\
      --num_envs 1 \\
      --enable_cameras
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CP5: Play trained navigation policy.")
parser.add_argument("--task",       type=str,  required=True, help="Gym task id")
parser.add_argument("--num_envs",   type=int,  default=1,     help="Number of envs")
parser.add_argument("--checkpoint", type=str,  default=None,  help="Path to skrl checkpoint (.pt)")
parser.add_argument("--max_steps",  type=int,  default=2000,  help="Max env steps to run")
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_cfg_entry_point",
    help="Agent config entry-point key",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────

import torch
import gymnasium as gym

from skrl.utils.runner.torch import Runner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    """Run the trained navigation policy."""

    env_cfg.scene.num_envs = args_cli.num_envs
    # Disable randomisation for evaluation
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # build agent (same architecture as training)
    runner = Runner(env, agent_cfg)

    if args_cli.checkpoint:
        path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint: {path}")
        runner.agent.load(path)
    else:
        print("[WARN] No checkpoint supplied — running random policy.")

    # switch to eval mode (disables exploration noise in GaussianMixin)
    runner.agent.enable_training_mode(False ,apply_to_models=True )

    # reset
    obs, info = env.reset()
    step = 0

    print(f"[INFO] Running CP5 nav policy for up to {args_cli.max_steps} steps...")
    with torch.no_grad():
        while simulation_app.is_running() and step < args_cli.max_steps:
            # nav policy inference
            #actions = runner.agent.act(obs,obs,timestep=0, timesteps=0)
            result = runner.agent.act(obs,obs, timestep=0, timesteps=0)
            actions = result[0] if isinstance(result, tuple) else result


            # step environment (PreTrainedPolicyAction handles loco internally)
            obs, rewards, terminated, truncated, info = env.step(actions)

            # ── velocity command verification (every 10 steps) ───────────────
            if step % 10 == 0:
                raw_env  = env.unwrapped
                nav_cmd  = actions[0].cpu().numpy()                              # [vx, vy, yaw]
                lin_vel  = raw_env.scene["robot"].data.root_lin_vel_b[0].cpu().numpy()
                ang_rate = float(raw_env.scene["robot"].data.root_ang_vel_b[0, 2].item())
                pos      = raw_env.scene["robot"].data.root_pos_w[0].cpu().numpy()
                print(
                    f"step {step:4d} | "
                    f"CMD vx={nav_cmd[0]:+.3f} vy={nav_cmd[1]:+.3f} yaw={nav_cmd[2]:+.3f} | "
                    f"ACT vx={lin_vel[0]:+.3f} vy={lin_vel[1]:+.3f} yaw={ang_rate:+.3f} | "
                    f"POS ({pos[0]:.2f}, {pos[1]:.2f})"
                )

            done = terminated | truncated
            if done.any():
                print(f"[step {step}] Episode done for {done.sum().item()} env(s).")

            step += 1

    print(f"[INFO] Done after {step} steps.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
