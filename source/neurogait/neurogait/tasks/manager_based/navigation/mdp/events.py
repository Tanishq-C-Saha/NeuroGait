"""Navigation event functions for Isaac Lab manager-based envs.

CP6:   obstacle randomization + A* replanning on episode reset.
CP6.5: path-first scene generation + curriculum (no A* needed).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_last_curriculum_log: float = 0.0   # rate-limit curriculum prints to 1 per 60 s


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

    # Grid parameters — must match build_global_grid defaults.
    # Centre on env-0 origin (same as robot spawn) so the grid covers the
    # actual environment when env_spacing places it far from world (0, 0).
    grid_resolution  = 0.2
    grid_size        = 200
    half             = grid_size * grid_resolution / 2.0
    grid_origin      = (float(env0_origin_xy[0]) - half,
                        float(env0_origin_xy[1]) - half)

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
        pass  # silenced — straight-line fallback is rare and recoverable

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

    # Only log on the very first replan — with 512 envs resetting frequently
    # this print would otherwise flood the terminal during training.
    if not hasattr(env, "_cp6_replan_logged"):
        print(
            f"[CP6] First replan: dx={dx:+.2f} m dy={dy:+.2f} m | "
            f"{len(waypoints)} waypoints | {len(env_ids)}/{env.num_envs} envs reset"
        )
        env._cp6_replan_logged = True


def cp65_reset_with_generated_scene(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """EventTerm (mode="reset"): obstacles-first scene generation.

    Reads difficulty from env._curr_* attributes written by the
    CurriculumTerm (curriculum_obstacle_difficulty), which Isaac Lab
    calls automatically before this reset event fires inside _reset_idx().

    Each reset:
      1. Reads current difficulty from env._curr_* (set by CurriculumTerm)
      2. Generates a random goal + places obstacles with BFS connectivity check
      3. Runs A* for the optimal path (same algorithm as deployment)
      4. Writes obstacle world positions to all envs in the sim
      5. Stores waypoints as env._cp5_waypoints (E, W, 2)
      6. Resets waypoint tracking state for the resetting envs

    Last waypoint == goal, so all CP5/CP6 reward and termination functions
    work unchanged.
    """
    global _last_curriculum_log

    from neurogait.tasks.manager_based.navigation.scene import (
        generate_scene,
        apply_scene_to_env,
    )
    from neurogait.tasks.manager_based.navigation.mdp.observations import (
        _cp5_init_waypoint_state,
    )

    # Ensure base waypoint tensors exist before we overwrite them
    _cp5_init_waypoint_state(env)

    # ── 1. Read difficulty params (written by CurriculumTerm before us) ───────
    gap_width     = getattr(env, "_curr_gap_width",     2.0)
    num_obstacles = getattr(env, "_curr_num_obstacles",  3)
    arena_padding = getattr(env, "_curr_arena_padding",  3.0)
    goal_dist     = getattr(env, "_curr_goal_dist",     (6.0, 7.0))
    goal_angle    = getattr(env, "_curr_goal_angle",    (-0.3, 0.3))

    # ── 2. Generate scene (obstacles-first, random goal, local coords) ────────
    obstacles, waypoints, goal_local = generate_scene(
        start_xy         = (0.0, 0.0),
        goal_xy          = None,
        num_obstacles    = num_obstacles,
        min_gap_width    = gap_width,
        arena_padding    = arena_padding,
        goal_dist_range  = goal_dist,
        goal_angle_range = goal_angle,
    )

    # ── 3. Apply obstacles to sim (all envs, shared layout) ───────────────────
    apply_scene_to_env(env, obstacles, env.device)

    # ── 4. Store waypoints: local → per-env world (E, W, 2) ──────────────────
    env_origins_xy = env.scene.env_origins[:, :2]       # (E, 2)
    local_wp = torch.tensor(
        waypoints, dtype=torch.float32, device=env.device
    )                                                     # (W, 2)
    new_wps = local_wp.unsqueeze(0) + env_origins_xy.unsqueeze(1)  # (E, W, 2)

    env._cp5_waypoints = new_wps
    W_new = new_wps.shape[1]

    # ── 5. Reset waypoint tracking state ──────────────────────────────────────
    env._cp5_wp_idx.clamp_(max=W_new - 1)
    env._cp5_wp_idx[env_ids]      = 0
    env._cp5_prev_dist[env_ids]   = float("inf")
    env._cp5_prev_action[env_ids] = 0.0
    env._cp5_pos_history[env_ids] = 0.0

    if hasattr(env, "_cp6_prev_action_1"):
        env._cp6_prev_action_1[env_ids] = 0.0
    if hasattr(env, "_cp6_prev_action_2"):
        env._cp6_prev_action_2[env_ids] = 0.0

    # ── Rate-limited logging (max 1 per 60 s) ─────────────────────────────────
    now = time.time()
    if now - _last_curriculum_log > 60.0:
        from neurogait.tasks.manager_based.navigation.mdp.curriculums import _NAV_ROLLOUT_LENGTH
        t = getattr(env, "_curr_gap_width", None)   # proxy: set means curriculum ran
        t_pct = 0.0
        if hasattr(env, "common_step_counter"):
            ramp_steps = 2_000 * _NAV_ROLLOUT_LENGTH
            t_pct = min(100.0, 100.0 * env.common_step_counter / max(1, ramp_steps))
        print(
            f"[CP6.5] curriculum {t_pct:.0f}% | "
            f"gap={gap_width:.2f}m obs={num_obstacles} pad={arena_padding:.1f}m | "
            f"goal=({goal_local[0]:.1f},{goal_local[1]:.1f}) | "
            f"{W_new} waypoints"
        )
        _last_curriculum_log = now
