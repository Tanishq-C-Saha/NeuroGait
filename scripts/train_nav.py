"""CP5: Train navigation RL policy with skrl PPO.

Uses PreTrainedPolicyActionCfg to wrap the frozen locomotion checkpoint.
The navigation policy (3-dim velocity commands) is trained with skrl PPO.

Usage:
    # Smoke test (50 iterations, ~2 min):
    ~/isaac-sim/kit/python/bin/python3 scripts/train_nav.py \\
      --task NeuroGait-Navigation-CP5-v0 \\
      --num_envs 256 \\
      --headless \\
      --max_iterations 50

    # Full training (2000 iterations):
    ~/isaac-sim/kit/python/bin/python3 scripts/train_nav.py \\
      --task NeuroGait-Navigation-CP5-v0 \\
      --num_envs 1024 \\
      --headless \\
      --max_iterations 2000

    # Resume from checkpoint:
    ~/isaac-sim/kit/python/bin/python3 scripts/train_nav.py \\
      --task NeuroGait-Navigation-CP5-v0 \\
      --num_envs 1024 \\
      --headless \\
      --max_iterations 2000 \\
      --checkpoint logs/skrl/neurogait_navigation_cp5/<run>/agent.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# ── argument parsing (MUST happen before AppLauncher) ────────────────────────

parser = argparse.ArgumentParser(description="CP5: Train navigation policy with skrl PPO.")
parser.add_argument("--task",           type=str,   required=True,   help="Gym task id")
parser.add_argument("--num_envs",       type=int,   default=1024,    help="Number of parallel envs")
parser.add_argument("--max_iterations", type=int,   default=2000,    help="PPO update iterations")
parser.add_argument("--checkpoint",     type=str,   default=None,    help="Resume from checkpoint")
parser.add_argument("--seed",           type=int,   default=42,      help="Random seed")
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_cfg_entry_point",
    help="Agent config entry-point key (default: skrl_cfg_entry_point)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that need the simulator running ────────────────────────────────────

import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    """Train navigation policy with skrl PPO."""

    # apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed

    # adjust total timesteps from max_iterations
    rollouts = agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts * args_cli.num_envs
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # set up log directory
    log_root = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_torch"
    log_dir  = os.path.join(log_root, run_name)
    agent_cfg["agent"]["experiment"]["directory"]       = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    env_cfg.log_dir = log_dir

    print(f"[INFO] Logging to: {log_dir}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"),   env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # build Runner (instantiates PPO + networks from YAML)
    runner = Runner(env, agent_cfg)

    # optionally resume from checkpoint
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.agent.load(resume_path)

    # train
    t0 = time.time()
    runner.run()
    print(f"[INFO] Training finished in {round(time.time() - t0, 1)} s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
