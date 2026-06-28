"""Curriculum functions for the NeuroGait navigation task.

Copied from isaaclab_tasks locomotion velocity mdp. Future CPs will add
navigation-specific curriculum terms here (e.g. obstacle density progression).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# rollouts: 24 from agents/skrl_nav_ppo_cfg.yaml — converts training iterations
# to env.common_step_counter units (which increments once per env.step() call)
_NAV_ROLLOUT_LENGTH: int = 24


def curriculum_obstacle_difficulty(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    ramp_iterations: int = 2_000,
    gap_start: float = 2.0,
    gap_end: float = 0.51,
    obs_start: int = 3,
    obs_end: int = 12,
    pad_start: float = 3.0,
    pad_end: float = 1.5,
    angle_start: float = 0.3,
    angle_end: float = 0.8,
) -> dict:
    """CurriculumTermCfg func: ramp obstacle difficulty over training.

    Isaac Lab calls this automatically inside _reset_idx(), before the reset
    EventTerm fires.  Writes env._curr_* attributes which
    cp65_reset_with_generated_scene reads.

    Args:
        ramp_iterations: training iterations until full difficulty.
                         Internally multiplied by _NAV_ROLLOUT_LENGTH (24) to
                         convert to env.common_step_counter units.
        gap_start / gap_end:   corridor width at t=0 / t=1 (metres).
        obs_start / obs_end:   obstacle count at t=0 / t=1.
        pad_start / pad_end:   arena padding at t=0 / t=1 (metres).
        angle_start / angle_end: half-angle of goal direction spread (radians).
    """
    t = min(1.0, env.common_step_counter / max(1, ramp_iterations * _NAV_ROLLOUT_LENGTH))

    def lerp(a: float, b: float) -> float:
        return a + t * (b - a)

    env._curr_gap_width     = lerp(gap_start, gap_end)
    env._curr_num_obstacles = int(round(lerp(obs_start, obs_end)))
    env._curr_arena_padding = lerp(pad_start, pad_end)
    env._curr_goal_dist     = (lerp(6.0, 5.0), lerp(7.0, 10.0))
    env._curr_goal_angle    = (-lerp(angle_start, angle_end), lerp(angle_start, angle_end))

    return {
        "t":             round(t, 3),
        "gap_width":     round(env._curr_gap_width, 3),
        "num_obstacles": env._curr_num_obstacles,
    }


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum: increase terrain difficulty when the robot walks far enough.

    Robots that walked more than half the terrain patch advance to harder levels.
    Robots that walked less than half their commanded distance regress to easier levels.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
