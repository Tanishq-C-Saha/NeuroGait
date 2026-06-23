"""CP5 — Train the navigation policy with skrl PPO + CNN+MLP models.

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/cp5/train.py \
      --task NeuroGait-Navigation-CP5-v0 \
      --num_envs 512 \
      --headless

Training checklist (run in this order):
  1. python scripts/cp5/export_locomotion.py          # verify TorchScript shape
  2. python scripts/cp5/test_reward_scales.py --num_envs 256 --headless
  3. python scripts/cp5/train.py --num_envs 512 --headless --max_iterations 50  (smoke test)
  4. python scripts/cp5/train.py --num_envs 512 --headless                      (full run)
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CP5 navigation training with skrl PPO")
parser.add_argument("--task",           type=str, default="NeuroGait-Navigation-CP5-v0")
parser.add_argument("--num_envs",       type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override trainer timesteps (iterations × rollouts)")
parser.add_argument("--checkpoint",     type=str, default=None,
                    help="Resume from a previous skrl checkpoint (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from datetime import datetime
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.runner.torch import Runner

import neurogait.tasks  # noqa: F401
from neurogait.tasks.manager_based.navigation.models import NavigationPolicy, NavigationValue


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    if args_cli.max_iterations is not None:
        rollouts = agent_cfg.get("agent", {}).get("rollouts", 24)
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts

    # Close environment at exit is handled by our own close() call
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env_cfg.sim.device
    obs_space  = env.observation_space
    act_space  = env.action_space

    policy = NavigationPolicy(obs_space, act_space, device)
    value  = NavigationValue(obs_space, act_space, device)
    models = {"policy": policy, "value": value}

    # Inject custom models into agent_cfg so Runner picks them up
    agent_cfg["models"] = models

    log_dir = os.path.join(
        "logs", "skrl", "neurogait_cp5_navigation",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(
        os.path.join("logs", "skrl", "neurogait_cp5_navigation")
    )
    agent_cfg["agent"]["experiment"]["experiment_name"] = os.path.basename(log_dir)

    runner = Runner(env, agent_cfg)

    if args_cli.checkpoint:
        print(f"[CP5-train] Resuming from: {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)

    runner.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
