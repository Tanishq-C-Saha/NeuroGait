"""Navigation event functions for Isaac Lab manager-based envs.

CP6: obstacle randomization + A* replanning on episode reset.
All obstacles shift by a SHARED random offset (same across all parallel envs)
so the global occupancy grid remains consistent and A* replanning is
only needed once per reset call.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cp6_randomize_obstacles_and_replan(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: dict | None = None,
    goal_local_xy: tuple = (8.0, 0.0),
) -> None:
    """EventTerm (mode="reset"): randomize obstacle positions then replan A*.

    Applies a single shared (dx, dy) offset to every obstacle in every
    parallel env so the local layout is identical across envs.  Rebuilds
    the occupancy grid from the new positions (without re-reading from PhysX
    to avoid timing issues) and replans A*.  Retries up to 3 times if A*
    cannot find a path.  Falls back to a straight line if all retries fail.

    State updated:
        env._cp5_waypoints    — (E, W_new, 2) replanned for all envs
        env._cp5_wp_idx       — clamped for running envs, zeroed for env_ids
        env._cp5_prev_dist    — reset to inf for env_ids
        env._cp5_prev_action  — zeroed for env_ids
        env._cp5_pos_history  — zeroed for env_ids
    """
    if position_range is None:
        position_range = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}

    from neurogait.tasks.manager_based.navigation.planning.global_grid import (
        _get_shape_info,
        _INFLATION_M,
    )
    from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state

    # Ensure waypoint-state tensors exist before writing into them.
    # On the very first reset this also builds the initial A* path — we will
    # immediately overwrite it with the randomised plan below.
    _cp5_init_waypoint_state(env)

    env_origins    = env.scene.env_origins           # (E, 3)
    env_origins_xy = env_origins[:, :2]              # (E, 2)
    env0_origin_xy = env_origins_xy[0].cpu().numpy() # (2,)
    all_ids        = torch.arange(env.num_envs, device=env.device)

    # Use env-0 origin as the A* start (robot spawns within ±0.5 m of origin)
    start_xy  = (float(env0_origin_xy[0]), float(env0_origin_xy[1]))
    goal_world = (
        float(env0_origin_xy[0]) + goal_local_xy[0],
        float(env0_origin_xy[1]) + goal_local_xy[1],
    )

    # Grid parameters — must match build_global_grid defaults
    grid_resolution  = 0.2
    grid_size        = 200
    half             = grid_size * grid_resolution / 2.0
    grid_origin      = (-half, -half)

    lo_x, hi_x = position_range["x"]
    lo_y, hi_y = position_range["y"]

    waypoints = None
    dx = dy   = 0.0

    for attempt in range(3):
        dx = lo_x + torch.rand(1).item() * (hi_x - lo_x)
        dy = lo_y + torch.rand(1).item() * (hi_y - lo_y)

        # ── move obstacles + collect new env-0 world positions for grid ────────
        grid          = np.zeros((grid_size, grid_size), dtype=np.uint8)
        obs_world_env0 = {}   # name → (ox, oy, obj)

        for name, obj in env.scene.rigid_objects.items():
            if not any(k in name.lower() for k in ("obstacle", "cube", "cyl")):
                continue

            default_local = torch.tensor(
                obj.cfg.init_state.pos, dtype=torch.float32, device=env.device
            )  # (3,) local position relative to env origin

            new_lx = float(default_local[0]) + dx
            new_ly = float(default_local[1]) + dy
            new_lz = float(default_local[2])

            # World positions for every env (same local offset, different origins)
            new_pos_w           = torch.zeros(env.num_envs, 3, device=env.device)
            new_pos_w[:, 0]     = env_origins_xy[:, 0] + new_lx
            new_pos_w[:, 1]     = env_origins_xy[:, 1] + new_ly
            new_pos_w[:, 2]     = env_origins[:, 2]    + new_lz

            orig_quat = obj.data.root_quat_w.clone()  # preserve orientation
            pose      = torch.cat([new_pos_w, orig_quat], dim=-1)  # (E, 7)
            obj.write_root_pose_to_sim(pose, env_ids=all_ids)

            # Zero velocities so teleported obstacles don't fly off
            obj.write_root_velocity_to_sim(
                torch.zeros(env.num_envs, 6, device=env.device), env_ids=all_ids
            )

            # Record env-0 world position for grid building
            ox = float(env_origins_xy[0, 0]) + new_lx
            oy = float(env_origins_xy[0, 1]) + new_ly
            obs_world_env0[name] = (ox, oy, obj)

        # ── build occupancy grid from new positions (bypass PhysX read-back) ──
        for _name, (ox, oy, obj) in obs_world_env0.items():
            shape, sx, sy = _get_shape_info(obj)
            inflate_r = max(sx, sy) / 2.0 + _INFLATION_M

            c_min = max(0, int((ox - inflate_r - grid_origin[0]) / grid_resolution))
            c_max = min(grid_size - 1, int((ox + inflate_r - grid_origin[0]) / grid_resolution))
            r_min = max(0, int((oy - inflate_r - grid_origin[1]) / grid_resolution))
            r_max = min(grid_size - 1, int((oy + inflate_r - grid_origin[1]) / grid_resolution))

            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    cx_w = grid_origin[0] + (c + 0.5) * grid_resolution
                    cy_w = grid_origin[1] + (r + 0.5) * grid_resolution
                    if (cx_w - ox) ** 2 + (cy_w - oy) ** 2 <= inflate_r ** 2:
                        grid[r, c] = 1

        planner   = AStarPlanner(grid, grid_origin, resolution=grid_resolution)
        waypoints = planner.plan(start_xy, goal_world)
        if waypoints:
            break

    if not waypoints:
        # Straight-line fallback
        dx_g  = goal_world[0] - start_xy[0]
        dy_g  = goal_world[1] - start_xy[1]
        dist  = math.sqrt(dx_g ** 2 + dy_g ** 2)
        n     = max(int(dist), 2)
        waypoints = [
            (start_xy[0] + dx_g * i / n, start_xy[1] + dy_g * i / n)
            for i in range(1, n + 1)
        ]
        print("[CP6] WARNING: A* failed after 3 attempts — using straight-line path")

    # ── convert env-0 world waypoints → local → broadcast to all envs ─────────
    local_wp_np = np.array(waypoints, dtype=np.float32) - env0_origin_xy  # (W, 2)
    local_wp    = torch.tensor(local_wp_np, dtype=torch.float32, device=env.device)
    new_wps     = local_wp.unsqueeze(0) + env_origins_xy.unsqueeze(1)     # (E, W, 2)

    env._cp5_waypoints = new_wps
    W_new = new_wps.shape[1]

    # Running envs: clamp wp_idx so it stays within the new path length
    env._cp5_wp_idx.clamp_(max=W_new - 1)

    # Resetting envs: full state reset
    env._cp5_wp_idx[env_ids]      = 0
    env._cp5_prev_dist[env_ids]   = float("inf")
    env._cp5_prev_action[env_ids] = 0.0
    env._cp5_pos_history[env_ids] = 0.0

    print(
        f"[CP6] Obstacles randomized (dx={dx:+.2f} m, dy={dy:+.2f} m) | "
        f"{len(waypoints)} waypoints | {len(env_ids)}/{env.num_envs} envs reset"
    )
