"""Waypoint management for CP5 hierarchical navigation.

Provides init_waypoints() — an EventTerm function (mode="reset") that:
  1. Builds a global occupancy grid from the env's rigid objects (cached).
  2. Runs A* from (0,0) to the fixed goal (8,0) (cached across all resets).
  3. Resets per-env waypoint tracking tensors for the given env_ids.

Per-env tensors set on the env object:
  env._waypoints_tensor    (num_waypoints, 2) float32  — A* path in world coords
  env._curr_waypoint_idx   (num_envs,)      long      — current waypoint per env
  env._curr_waypoint_pos   (num_envs, 2)    float32   — current waypoint world XY
  env._prev_waypoint_dist  (num_envs,)      float32   — dist to current wp (prev step)
  env._prev_nav_action     (num_envs, 3)    float32   — action at previous step
  env._nav_goal_xy         (2,)             float32   — fixed world goal position

Waypoint advancement (advance when robot within 0.3 m) is handled inside
mdp/rewards.py::reward_progress() so it runs every step without a separate event.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from neurogait.tasks.manager_based.navigation.planning.global_grid import build_global_grid
from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner

_NAV_GOAL_X: float = 8.0
_NAV_GOAL_Y: float = 0.0
_WAYPOINT_ADVANCE_RADIUS: float = 0.3   # metres — advance when this close
_GRID_RESOLUTION: float = 0.2           # metres per cell (must match global_grid)
_GRID_SIZE: int = 200                   # cells (40 m × 40 m arena)


def init_waypoints(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset waypoint state for env_ids.

    Called as::

        EventTerm(func=nav_mdp.init_waypoints, mode="reset")

    The global grid and A* path are built once and cached on env._waypoints_tensor.
    Only the per-env index / position / distance tensors are reset for env_ids.
    """
    device = env.device

    # ── build grid + run A* once (module-level cache on the env object) ─────────
    if not hasattr(env, "_waypoints_tensor") or env._waypoints_tensor is None:
        print("[CP5] Building global occupancy grid and running A* (first reset)...")
        grid, origin, _ = build_global_grid(
            env, grid_resolution=_GRID_RESOLUTION, grid_size=_GRID_SIZE
        )
        planner = AStarPlanner(grid, origin, resolution=_GRID_RESOLUTION)
        waypoints = planner.plan((0.0, 0.0), (_NAV_GOAL_X, _NAV_GOAL_Y))

        if not waypoints:
            # fallback: straight-line path every 1 m
            print("[CP5] WARNING: A* found no path — using straight-line fallback.")
            waypoints = [
                (float(x), 0.0) for x in range(0, int(_NAV_GOAL_X) + 1)
            ]

        env._waypoints_tensor = torch.tensor(
            waypoints, dtype=torch.float32, device=device
        )  # (num_waypoints, 2)
        env._nav_goal_xy = torch.tensor(
            [_NAV_GOAL_X, _NAV_GOAL_Y], dtype=torch.float32, device=device
        )
        print(f"[CP5] A* path cached: {len(waypoints)} waypoints, "
              f"goal = ({_NAV_GOAL_X}, {_NAV_GOAL_Y})")

    n = env.num_envs

    # ── initialise per-env tensors if they don't exist yet ───────────────────────
    if not hasattr(env, "_curr_waypoint_idx"):
        first_wp = env._waypoints_tensor[0]  # (2,)
        env._curr_waypoint_idx = torch.zeros(n, dtype=torch.long, device=device)
        env._curr_waypoint_pos = first_wp.unsqueeze(0).expand(n, 2).clone()
        env._prev_waypoint_dist = torch.full((n,), 1e6, dtype=torch.float32, device=device)
        env._prev_nav_action = torch.zeros(n, 3, dtype=torch.float32, device=device)

    if len(env_ids) == 0:
        return

    # ── reset only the envs being reset ──────────────────────────────────────────
    first_wp = env._waypoints_tensor[0]
    env._curr_waypoint_idx[env_ids] = 0
    env._curr_waypoint_pos[env_ids] = first_wp
    env._prev_nav_action[env_ids] = 0.0

    # initial distance from robot spawn to first waypoint
    robot_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]
    env._prev_waypoint_dist[env_ids] = torch.norm(
        robot_xy - first_wp.unsqueeze(0), dim=-1
    )
