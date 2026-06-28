# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##



gym.register(
    id="NeuroGait-Navigation-Unitree-Go2-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NeuroGaitNavigationCP1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_locomotion_ppo_cfg:NeuroGaitNavigationCP1PPORunnerCfg",
    },
)

# ── CP5: Trained navigation policy (skrl PPO + CNN+MLP) ──────────────────────

gym.register(
    id="NeuroGait-Navigation-CP5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NeuroGaitNavigationCP5EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_ppo_cfg.yaml",
    },
)

gym.register(
    id="NeuroGait-Navigation-CP5-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NeuroGaitNavigationCP5EnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_ppo_cfg.yaml",
    },
)

# ── CP6: Generalized navigation (randomised obstacles, upgraded rewards) ──────

gym.register(
    id="NeuroGait-Navigation-CP6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NeuroGaitNavigationCP6EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_ppo_cfg.yaml",
    },
)

gym.register(
    id="NeuroGait-Navigation-CP6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NeuroGaitNavigationCP6EnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_ppo_cfg.yaml",
    },
)

# ── CP6.5: Path-first scene generation + 4-axis curriculum ───────────────────

gym.register(
    id="NeuroGait-Navigation-CP65-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NeuroGaitNavigationCP65EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_ppo_cfg.yaml",
    },
)

gym.register(
    id="NeuroGait-Navigation-CP65-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NeuroGaitNavigationCP65EnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_nav_ppo_cfg.yaml",
    },
)
