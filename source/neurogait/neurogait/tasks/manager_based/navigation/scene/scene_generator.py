"""
Path-first procedural scene generation for navigation training.

Algorithm:
  1. Generate a random smooth path from start to goal (cubic spline)
  2. Define a protected corridor of width W around the path
  3. Place obstacles randomly OUTSIDE the corridor
  4. Path is ALWAYS traversable by construction — no A* needed

Corridor width controls difficulty:
  Easy:    2.0m  (robot can wander freely)
  Medium:  1.0m  (needs to follow the path)
  Hard:    0.7m  (0.20m clearance each side of the 0.31m-wide body)
  Extreme: 0.51m (just squeezable — 0.10m clearance each side)

Cite: path-first traversability guarantee adapted from graph-based
passage networks (Beyond Specialization, Lutz et al., 2026).

All coordinates are LOCAL to the env frame: start=(0,0), goal=(8,0).
`apply_scene_to_env` adds env_origins to get world positions.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

# ── Go2 body dimensions (actual spec, not estimates) ─────────────────────────
GO2_BODY_WIDTH  = 0.31   # metres (relevant for lateral clearance)
GO2_BODY_LENGTH = 0.70   # metres
GO2_BODY_HEIGHT = 0.40   # metres
SAFETY_MARGIN   = 0.10   # clearance on each side beyond robot edge
MIN_CORRIDOR    = GO2_BODY_WIDTH + 2 * SAFETY_MARGIN   # 0.51 m — just squeezable


def generate_scene(
    start_xy: tuple = (0.0, 0.0),
    goal_xy: tuple  = (8.0, 0.0),
    num_obstacles: int = 9,
    corridor_width: float = 1.5,
    num_control_points: int = 3,
    max_lateral_deviation: float = 2.5,
    arena_bounds: tuple = (-1.0, 9.0, -4.0, 4.0),
    obstacle_size_range: tuple = (0.4, 1.5),
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict], list[tuple]]:
    """Generate a guaranteed-traversable obstacle scene.

    Args:
        start_xy: start position in local env frame (usually (0, 0))
        goal_xy:  goal position in local env frame (usually (8, 0))
        num_obstacles:          how many obstacles to place
        corridor_width:         protected width around path in metres
                                MIN_CORRIDOR (0.51m) = just squeezable for Go2
        num_control_points:     interior spline control points (path complexity)
        max_lateral_deviation:  max Y deviation of control points from straight line
        arena_bounds:           (x_min, x_max, y_min, y_max) for obstacle placement
        obstacle_size_range:    (min, max) obstacle footprint dimension in metres
        seed:                   RNG seed for reproducibility (None = random)

    Returns:
        path_points: (N, 2) dense path array — defines the protected corridor
        obstacles:   list of {"pos": (lx,ly,lz), "size": (sx,sy,sz), "shape": str}
                     in LOCAL env coordinates
        waypoints:   downsampled [(lx, ly), ...] for use as env._cp5_waypoints
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    path_points, waypoints = _generate_random_path(
        start_xy, goal_xy, num_control_points, max_lateral_deviation
    )
    obstacles = _place_obstacles(
        path_points, corridor_width, num_obstacles, arena_bounds, obstacle_size_range
    )
    return path_points, obstacles, waypoints


def _generate_random_path(
    start_xy: tuple,
    goal_xy:  tuple,
    num_control_points: int,
    max_deviation: float,
) -> tuple[np.ndarray, list[tuple]]:
    """Cubic spline through random interior control points."""
    try:
        from scipy.interpolate import CubicSpline
        _use_scipy = True
    except ImportError:
        _use_scipy = False

    n_pts = num_control_points + 2   # includes start and goal
    cx = np.linspace(start_xy[0], goal_xy[0], n_pts)
    cy = np.zeros(n_pts)
    cy[0]  = start_xy[1]
    cy[-1] = goal_xy[1]
    for i in range(1, n_pts - 1):
        cy[i] = start_xy[1] + random.uniform(-max_deviation, max_deviation)

    t_ctrl  = np.linspace(0.0, 1.0, n_pts)
    t_dense = np.linspace(0.0, 1.0, 300)

    if _use_scipy:
        path_x = CubicSpline(t_ctrl, cx)(t_dense)
        path_y = CubicSpline(t_ctrl, cy)(t_dense)
    else:
        path_x = np.interp(t_dense, t_ctrl, cx)
        path_y = np.interp(t_dense, t_ctrl, cy)

    path_points = np.stack([path_x, path_y], axis=1)  # (300, 2)

    # Downsample to ~1 m waypoint spacing
    seg_lens = np.sqrt(np.sum(np.diff(path_points, axis=0) ** 2, axis=1))
    total_len = float(seg_lens.sum())
    n_wp = max(3, int(total_len))
    wp_idx = np.linspace(0, len(path_points) - 1, n_wp, dtype=int)
    waypoints = [(float(path_points[i, 0]), float(path_points[i, 1])) for i in wp_idx]

    return path_points, waypoints


def _place_obstacles(
    path_points: np.ndarray,
    corridor_width: float,
    num_obstacles: int,
    arena_bounds: tuple,
    size_range: tuple,
) -> list[dict]:
    """Place obstacles outside the protected corridor. Never fails — fewer is OK."""
    x_min, x_max, y_min, y_max = arena_bounds
    half_corridor = corridor_width / 2.0
    obstacles: list[dict] = []
    max_attempts = num_obstacles * 50

    for _ in range(max_attempts):
        if len(obstacles) >= num_obstacles:
            break

        ox  = random.uniform(x_min, x_max)
        oy  = random.uniform(y_min, y_max)
        sx  = random.uniform(size_range[0], size_range[1])
        sy  = random.uniform(size_range[0], size_range[1])
        sz  = random.uniform(0.3, 0.8)
        shape = random.choice(["cube", "cube", "cylinder"])

        # Every corner of the obstacle must be outside the corridor
        corners_clear = True
        for dx in (-sx / 2, sx / 2):
            for dy in (-sy / 2, sy / 2):
                px, py = ox + dx, oy + dy
                dists = np.sqrt(
                    (path_points[:, 0] - px) ** 2 + (path_points[:, 1] - py) ** 2
                )
                if dists.min() < half_corridor:
                    corners_clear = False
                    break
            if not corners_clear:
                break

        if not corners_clear:
            continue

        # No overlap with already-placed obstacles
        for ex_obs in obstacles:
            ex, ey = ex_obs["pos"][0], ex_obs["pos"][1]
            esx, esy = ex_obs["size"][0], ex_obs["size"][1]
            if (abs(ox - ex) < (sx + esx) / 2 + 0.2 and
                    abs(oy - ey) < (sy + esy) / 2 + 0.2):
                corners_clear = False
                break

        if corners_clear:
            obstacles.append({
                "pos":   (ox, oy, sz / 2),   # LOCAL coords; z = half-height above ground
                "size":  (sx, sy, sz),
                "shape": shape,
            })

    return obstacles


def apply_scene_to_env(
    env,
    obstacles: list[dict],
    device: str | torch.device,
) -> None:
    """Write generated obstacle positions to the simulator for ALL parallel envs.

    Obstacles are specified in LOCAL env coordinates (same for every env).
    This function broadcasts to world coords by adding each env's origin.
    Unused scene objects (when len(obstacles) < num_scene_objects) are moved
    to y=100 (effectively hidden).
    """
    env_origins_xy = env.scene.env_origins[:, :2]   # (E, 2)
    E = env.num_envs
    all_ids = torch.arange(E, device=device)

    scene_objs = [
        (name, obj)
        for name, obj in env.scene.rigid_objects.items()
        if any(k in name.lower() for k in ("cube", "cyl", "obstacle"))
    ]

    for i, (name, obj) in enumerate(scene_objs):
        if i < len(obstacles):
            lx, ly, lz = obstacles[i]["pos"]
            local_pos = torch.tensor([[lx, ly, lz]], dtype=torch.float32, device=device)
            # Broadcast: world_xy = local_xy + env_origin  (z is absolute)
            world_pos = torch.zeros(E, 3, device=device)
            world_pos[:, 0] = env_origins_xy[:, 0] + lx
            world_pos[:, 1] = env_origins_xy[:, 1] + ly
            world_pos[:, 2] = env.scene.env_origins[:, 2] + lz
        else:
            # Park unused objects far from any env
            world_pos = torch.zeros(E, 3, device=device)
            world_pos[:, 1] = 1000.0

        orig_quat = obj.data.root_quat_w.clone()
        pose = torch.cat([world_pos, orig_quat], dim=-1)
        obj.write_root_pose_to_sim(pose, env_ids=all_ids)
        obj.write_root_velocity_to_sim(
            torch.zeros(E, 6, device=device), env_ids=all_ids
        )
