"""
Obstacles-first scene generation with guaranteed start→goal connectivity.

Algorithm
─────────
1. Place obstacles randomly inside a tight arena around the start→goal line
2. After each placement, BFS flood-fill verifies start and goal are still
   connected (O(N) — faster than A*)
3. If the new obstacle blocks connectivity, reject it and try another position
4. A* runs once on the final grid to find the optimal path
   — same algorithm used at deployment, so training = deployment

Why obstacles-first?
   Path-first (previous): path curves for no reason; obstacles decorative
   Obstacles-first (now):  obstacles CREATE the gaps the policy must thread

Go2 C-space
───────────
   Body width  : 0.31 m
   Safety margin: 0.10 m each side
   Min corridor: 0.51 m  (just squeezable — 0.10 m clearance per side)

All positions are LOCAL to the env frame (start ≈ (0, 0)).
`apply_scene_to_env` adds env_origins to get world positions.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque

import numpy as np

# ── Go2 C-space constants ─────────────────────────────────────────────────────
GO2_WIDTH    = 0.31    # actual body width (m)
GO2_LENGTH   = 0.70    # actual body length (m)
GO2_HEIGHT   = 0.40    # actual body height (m)
SAFETY_MARGIN = 0.10   # per-side clearance beyond body edge
MIN_GAP      = GO2_WIDTH + 2 * SAFETY_MARGIN   # 0.51 m — just squeezable
# Backwards-compat alias used in curriculum / viz
MIN_CORRIDOR = MIN_GAP


def random_goal(
    start_xy: tuple,
    dist_range: tuple = (5.0, 10.0),
    angle_range: tuple = (-0.8, 0.8),
) -> tuple:
    """Random goal at variable distance and angle from start.

    angle_range in radians:
      ±0.3 ≈ ±17°  (nearly straight, easy)
      ±0.8 ≈ ±46°  (diagonal, moderate)
    """
    dist  = random.uniform(*dist_range)
    angle = random.uniform(*angle_range)
    return (
        start_xy[0] + dist * math.cos(angle),
        start_xy[1] + dist * math.sin(angle),
    )


def generate_scene(
    start_xy: tuple = (0.0, 0.0),
    goal_xy:  tuple | None = None,
    num_obstacles: int = 9,
    min_gap_width: float = 1.0,
    arena_padding: float = 2.5,
    obstacle_size_range: tuple = (0.5, 1.8),
    grid_resolution: float = 0.1,
    goal_dist_range: tuple = (5.0, 10.0),
    goal_angle_range: tuple = (-0.8, 0.8),
    seed: int | None = None,
) -> tuple[list[dict], list[tuple], tuple]:
    """Generate a scene with randomly placed obstacles and guaranteed connectivity.

    Args:
        start_xy:             robot start (local frame, usually (0, 0))
        goal_xy:              goal position; None = sampled randomly
        num_obstacles:        target obstacle count
        min_gap_width:        minimum navigable corridor in metres.
                              0.51m = Go2 just-squeezable; 1.0m = comfortable
        arena_padding:        how far arena extends beyond start↔goal bounding box.
                              Smaller = obstacles forced nearer the path = harder
        obstacle_size_range:  (min, max) footprint dimension per side
        grid_resolution:      metres per cell for the connectivity grid
        goal_dist_range:      (min, max) for random goal distance
        goal_angle_range:     (min, max) for random goal angle in radians
        seed:                 RNG seed (None = random)

    Returns:
        obstacles:  list of {"pos": (lx,ly,lz), "size": (sx,sy,sz), "shape": str}
                    in LOCAL env coordinates
        waypoints:  list of (lx, ly) tuples from A* — ends at goal
        goal_xy:    the (possibly randomised) goal in local env coordinates
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if goal_xy is None:
        goal_xy = random_goal(start_xy, goal_dist_range, goal_angle_range)

    # Arena: tight box around start↔goal
    pad   = arena_padding
    x_min = min(start_xy[0], goal_xy[0]) - pad
    x_max = max(start_xy[0], goal_xy[0]) + pad
    y_min = min(start_xy[1], goal_xy[1]) - pad
    y_max = max(start_xy[1], goal_xy[1]) + pad

    # Binary occupancy grid for connectivity checks
    cols   = int((x_max - x_min) / grid_resolution) + 1
    rows   = int((y_max - y_min) / grid_resolution) + 1
    origin = (x_min, y_min)

    def _to_cell(xy: tuple) -> tuple:
        c = int((xy[0] - origin[0]) / grid_resolution)
        r = int((xy[1] - origin[1]) / grid_resolution)
        return (max(0, min(rows - 1, r)), max(0, min(cols - 1, c)))

    grid: np.ndarray = np.zeros((rows, cols), dtype=np.uint8)
    inflate = min_gap_width / 2.0   # C-space inflation = half corridor width

    obstacles: list[dict] = []

    for _ in range(num_obstacles * 30):
        if len(obstacles) >= num_obstacles:
            break

        ox  = random.uniform(x_min + 0.5, x_max - 0.5)
        oy  = random.uniform(y_min + 0.5, y_max - 0.5)
        sx  = random.uniform(*obstacle_size_range)
        sy  = random.uniform(obstacle_size_range[0], obstacle_size_range[1] * 0.8)
        sz  = random.uniform(0.3, 0.8)

        # Minimum clearance from start and goal
        if math.hypot(ox - start_xy[0], oy - start_xy[1]) < 1.2:
            continue
        if math.hypot(ox - goal_xy[0], oy - goal_xy[1]) < 1.2:
            continue

        # No overlap with existing obstacles
        overlaps = False
        for ex_obs in obstacles:
            ex, ey = ex_obs["pos"][0], ex_obs["pos"][1]
            esx, esy = ex_obs["size"][0], ex_obs["size"][1]
            if (abs(ox - ex) < (sx + esx) / 2 + 0.2 and
                    abs(oy - ey) < (sy + esy) / 2 + 0.2):
                overlaps = True
                break
        if overlaps:
            continue

        # Stamp obstacle + inflation onto a test grid
        test_grid = grid.copy()
        _stamp(test_grid, ox, oy, sx, sy, inflate, origin, grid_resolution, rows, cols)

        # BFS connectivity check — keep only if start↔goal still reachable
        if _bfs_connected(test_grid, _to_cell(start_xy), _to_cell(goal_xy)):
            grid = test_grid
            obstacles.append({
                "pos":   (ox, oy, sz / 2),
                "size":  (sx, sy, sz),
                "shape": random.choice(["cube", "cube", "cylinder"]),
            })

    # A* for the optimal path (same algorithm used at deployment)
    start_cell = _to_cell(start_xy)
    goal_cell  = _to_cell(goal_xy)
    path_cells = _astar(grid, start_cell, goal_cell)

    if path_cells is None:
        waypoints = [start_xy, goal_xy]
    else:
        step = max(1, int(1.0 / grid_resolution))   # ~1 m spacing
        sampled = path_cells[::step]
        if sampled[-1] != path_cells[-1]:
            sampled = list(sampled) + [path_cells[-1]]
        waypoints = [
            (origin[0] + (c + 0.5) * grid_resolution,
             origin[1] + (r + 0.5) * grid_resolution)
            for r, c in sampled
        ]

    return obstacles, waypoints, goal_xy


# ── Internal helpers ──────────────────────────────────────────────────────────

def _stamp(
    grid: np.ndarray,
    ox: float, oy: float,
    sx: float, sy: float,
    inflate: float,
    origin: tuple,
    res: float,
    rows: int, cols: int,
) -> None:
    """Rasterise obstacle + C-space inflation onto `grid` (in-place)."""
    c_min = max(0, int((ox - sx / 2 - inflate - origin[0]) / res))
    c_max = min(cols - 1, int((ox + sx / 2 + inflate - origin[0]) / res))
    r_min = max(0, int((oy - sy / 2 - inflate - origin[1]) / res))
    r_max = min(rows - 1, int((oy + sy / 2 + inflate - origin[1]) / res))
    grid[r_min:r_max + 1, c_min:c_max + 1] = 1


def _bfs_connected(grid: np.ndarray, start: tuple, goal: tuple) -> bool:
    """BFS flood-fill — True iff start and goal are in the same free region."""
    rows, cols = grid.shape
    if grid[start[0], start[1]] or grid[goal[0], goal[1]]:
        return False
    visited: set = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and not grid[nr, nc]:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def _astar(grid: np.ndarray, start: tuple, goal: tuple) -> list | None:
    """4-directional A* with Manhattan heuristic. Returns path or None."""
    rows, cols = grid.shape
    if grid[start[0], start[1]] or grid[goal[0], goal[1]]:
        return None
    open_heap = [(0, start)]
    came_from: dict = {}
    g: dict = {start: 0}
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return list(reversed(path))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < rows and 0 <= nb[1] < cols and not grid[nb[0], nb[1]]:
                ng = g[cur] + 1
                if nb not in g or ng < g[nb]:
                    g[nb] = ng
                    h = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
                    heapq.heappush(open_heap, (ng + h, nb))
                    came_from[nb] = cur
    return None


def apply_scene_to_env(env, obstacles: list[dict], device) -> None:
    """Write generated obstacle positions to the simulator for ALL parallel envs.

    Obstacles are in LOCAL env coordinates (same layout for every env).
    Broadcasts to world coords by adding each env's origin.
    Unused scene objects (len(obstacles) < scene object count) are parked
    at y = 1000 m (effectively hidden).

    `torch` is imported lazily so this module works standalone (no Isaac Sim).
    """
    import torch

    env_origins    = env.scene.env_origins             # (E, 3)
    env_origins_xy = env_origins[:, :2]                # (E, 2)
    E              = env.num_envs
    all_ids        = torch.arange(E, device=device)

    scene_objs = [
        (name, obj)
        for name, obj in env.scene.rigid_objects.items()
        if any(k in name.lower() for k in ("cube", "cyl", "obstacle"))
    ]

    for i, (name, obj) in enumerate(scene_objs):
        if i < len(obstacles):
            lx, ly, lz = obstacles[i]["pos"]
            world_pos = torch.zeros(E, 3, device=device)
            world_pos[:, 0] = env_origins_xy[:, 0] + lx
            world_pos[:, 1] = env_origins_xy[:, 1] + ly
            world_pos[:, 2] = env_origins[:, 2] + lz
        else:
            world_pos = torch.zeros(E, 3, device=device)
            world_pos[:, 1] = 1000.0

        orig_quat = obj.data.root_quat_w.clone()
        pose = torch.cat([world_pos, orig_quat], dim=-1)
        obj.write_root_pose_to_sim(pose, env_ids=all_ids)
        obj.write_root_velocity_to_sim(torch.zeros(E, 6, device=device), env_ids=all_ids)
