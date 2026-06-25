"""CP5 — Train the navigation policy with skrl PPO + CNN+MLP models.

Uses PPO + SequentialTrainer directly (bypasses skrl Runner) so custom
NavigationPolicy / NavigationValue model instances can be passed in.

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
                    help="Override total iterations (overrides YAML timesteps)")
parser.add_argument("--checkpoint",     type=str, default=None,
                    help="Resume from a previous skrl checkpoint (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer

import neurogait.tasks  # noqa: F401
from neurogait.tasks.manager_based.navigation.models import NavigationPolicy, NavigationValue


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env_cfg.sim.device
    obs_space = env.observation_space
    act_space = env.action_space

    # ── Models ───────────────────────────────────────────────────────────────
    # PPO takes instantiated model objects, not class references.
    # Runner's YAML "class:" syntax resolves to built-in skrl models only —
    # custom architectures must bypass Runner and use PPO directly.
    policy = NavigationPolicy(observation_space=obs_space, action_space=act_space, device=device)
    value  = NavigationValue(observation_space=obs_space, action_space=act_space, device=device)
    models = {"policy": policy, "value": value}

    # ── Memory ───────────────────────────────────────────────────────────────
    a = agent_cfg.get("agent", {})
    rollouts = int(a.get("rollouts", 24))
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    # ── Experiment logging ───────────────────────────────────────────────────
    log_dir = os.path.abspath(os.path.join(
        "logs", "skrl", "neurogait_cp5_navigation",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    ))

    # ── PPO config ───────────────────────────────────────────────────────────
    # YAML keys "state_preprocessor"/"learning_rate_scheduler" are strings
    # that Runner would resolve; we wire the actual classes here directly.
    ppo_cfg = {
        "rollouts":                      rollouts,
        "learning_epochs":               int(a.get("learning_epochs", 5)),
        "mini_batches":                  int(a.get("mini_batches", 4)),
        "discount_factor":               float(a.get("discount_factor", 0.99)),
        "gae_lambda":                    float(a.get("lambda", 0.95)),
        "learning_rate":                 float(a.get("learning_rate", 3e-4)),
        "learning_rate_scheduler":       KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
        "state_preprocessor":            RunningStandardScaler,
        "state_preprocessor_kwargs":     {"size": obs_space, "device": device},
        "value_preprocessor":            RunningStandardScaler,
        "value_preprocessor_kwargs":     {"size": 1, "device": device},
        "grad_norm_clip":                float(a.get("grad_norm_clip", 1.0)),
        "ratio_clip":                    float(a.get("ratio_clip", 0.2)),
        "value_clip":                    float(a.get("value_clip", 0.2)),
        "entropy_loss_scale":            float(a.get("entropy_loss_scale", 0.01)),
        "value_loss_scale":              float(a.get("value_loss_scale", 1.0)),
        "kl_threshold":                  0.0,
        "time_limit_bootstrap":          bool(a.get("time_limit_bootstrap", True)),
        "experiment": {
            "directory":           log_dir,
            "experiment_name":     "",
            "write_interval":      "auto",
            "checkpoint_interval": int(a.get("experiment", {}).get("checkpoint_interval", 200)),
        },
    }

    agent = PPO(
        models=models,
        memory=memory,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        cfg=ppo_cfg,
    )

    if args_cli.checkpoint:
        print(f"[CP5-train] Resuming from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    # ── Trainer ──────────────────────────────────────────────────────────────
    t = agent_cfg.get("trainer", {})
    timesteps = int(t.get("timesteps", 48000))
    if args_cli.max_iterations is not None:
        timesteps = args_cli.max_iterations * rollouts

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": timesteps, "close_environment_at_exit": False},
    )

    print(f"[CP5-train] Logging to: {log_dir}")
    print(f"[CP5-train] Training for {timesteps} timesteps ({timesteps // rollouts} iterations)")
    trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
