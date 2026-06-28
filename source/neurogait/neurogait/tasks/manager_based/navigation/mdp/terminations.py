"""Termination functions for the NeuroGait navigation task.

Copied from isaaclab_tasks locomotion velocity mdp. Future CPs will add
navigation-specific terminations here (e.g. goal reached, stuck detection).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cp6_goal_reached(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate episode when robot reaches the final waypoint.

    Returns True for envs whose robot centre is within `threshold` metres
    of the last waypoint in env._cp5_waypoints.

    threshold=0.5 m gives a generous acceptance radius that accounts for
    the 0.3 m body half-width plus 0.2 m navigation tolerance.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)

    robot_xy   = env.scene["robot"].data.root_pos_w[:, :2]   # (E, 2)
    final_goal = env._cp5_waypoints[:, -1, :]                 # (E, 2)
    dist       = torch.norm(robot_xy - final_goal, dim=-1)    # (E,)
    return dist < threshold


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
) -> torch.Tensor:
    """Terminate when the robot moves too close to the edge of the terrain."""
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        map_width  = n_rows * grid_width  + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        asset: RigidObject = env.scene[asset_cfg.name]
        x_out = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width  - distance_buffer
        y_out = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out, y_out)
    else:
        raise ValueError("Unsupported terrain type — must be 'plane' or 'generator'.")
