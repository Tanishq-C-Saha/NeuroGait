"""
Build a 2D global occupancy grid from env.scene rigid object positions.

Convention:
    row  ↔  Y axis   (row 0 = min Y = origin_y)
    col  ↔  X axis   (col 0 = min X = origin_x)

Obstacle inflation (C-space mapping):
    Each obstacle is inflated by (robot_half_width + safety_margin) = 0.30 m.
    This maps the workspace problem into C-space: A* plans for the robot *centre*,
    and the inflated cells guarantee the robot body never touches an obstacle edge.

    Go2 body:       ~0.67 m long × 0.31 m wide
    Half-width:      0.155 m  (perpendicular to travel — relevant dimension)
    Safety margin:   0.145 m  (14.5 cm of true clearance from obstacle edge)
    Total inflation: 0.30 m  (1.5 cells at 0.20 m/cell)

    Using the diagonal half-extent (0.37 m) was too conservative: it caused
    all obstacles to merge into one black blob with no navigable corridors.
    For a robot travelling primarily forward, the half-WIDTH (not diagonal)
    is the relevant clearance dimension.
"""

import numpy as np

# --- robot body constants (Unitree Go2) ---
_ROBOT_HALF_WIDTH = 0.155   # half of body width (0.31 m) — relevant when travelling forward
_SAFETY_MARGIN    = 0.145   # additional safety buffer beyond robot body edge
_INFLATION_M      = _ROBOT_HALF_WIDTH + _SAFETY_MARGIN   # 0.30 m total C-space inflation

_DEFAULT_RADIUS   = 0.50    # fallback footprint when config cannot be read


def build_global_grid(env, grid_resolution=0.2, grid_size=200):
    """
    Reads obstacle positions from env.scene rigid objects and rasterises
    them into a 2D binary occupancy grid.

    Each obstacle is inflated by _INFLATION_M (0.30 m) so planning on this
    grid gives the robot body (not just its centre) sufficient clearance.

    Args:
        env            : ManagerBasedRLEnv instance
        grid_resolution: metres per cell
        grid_size      : cells per side

    Returns:
        grid          : np.ndarray (grid_size, grid_size) uint8, 0=free 1=occupied
        origin        : (float, float) world (x, y) of grid cell (0, 0)
        obstacle_info : list of dicts {name, x, y, shape, sx, sy, inflate_r}
    """
    print("[CP4] Building global occupancy grid...")

    half = grid_size * grid_resolution / 2.0
    origin = (-half, -half)

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    obstacle_info = []

    rigid_objects = env.scene.rigid_objects
    n_rasterized = 0

    for name, obj in rigid_objects.items():
        name_lower = name.lower()
        if "robot" in name_lower:
            continue
        if not any(k in name_lower for k in ("obstacle", "cube", "cyl")):
            continue

        pos_w = obj.data.root_pos_w[0].cpu().numpy()
        ox, oy = float(pos_w[0]), float(pos_w[1])

        shape, sx, sy = _get_shape_info(obj)
        obs_radius = max(sx, sy) / 2.0          # obstacle half-extent
        inflate_r  = obs_radius + _INFLATION_M  # C-space inflation

        c_min = max(0, int((ox - inflate_r - origin[0]) / grid_resolution))
        c_max = min(grid_size - 1, int((ox + inflate_r - origin[0]) / grid_resolution))
        r_min = max(0, int((oy - inflate_r - origin[1]) / grid_resolution))
        r_max = min(grid_size - 1, int((oy + inflate_r - origin[1]) / grid_resolution))

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                cx = origin[0] + (c + 0.5) * grid_resolution
                cy = origin[1] + (r + 0.5) * grid_resolution
                if (cx - ox) ** 2 + (cy - oy) ** 2 <= inflate_r ** 2:
                    grid[r, c] = 1

        obstacle_info.append({
            "name": name, "x": ox, "y": oy,
            "shape": shape, "sx": sx, "sy": sy,
            "inflate_r": inflate_r,
        })
        n_rasterized += 1

    occ_cells = int(grid.sum())
    print(f"[CP4] Grid built: {grid_size}x{grid_size} cells @ {grid_resolution}m/cell, "
          f"{n_rasterized} obstacles rasterized, {occ_cells} occupied cells "
          f"(C-space inflation per obstacle = {_INFLATION_M:.2f} m)")
    return grid, origin, obstacle_info


def _get_shape_info(obj):
    """Return (shape_str, sx, sy) for the object's spawn config."""
    try:
        spawn = obj.cfg.spawn
        if hasattr(spawn, "size"):
            sx, sy, _ = spawn.size
            return "cuboid", float(sx), float(sy)
        if hasattr(spawn, "radius"):
            r = float(spawn.radius)
            return "cylinder", r * 2, r * 2
    except Exception:
        pass
    return "unknown", _DEFAULT_RADIUS * 2, _DEFAULT_RADIUS * 2


def world_to_grid(world_xy, origin, resolution):
    """Convert (x, y) world coords to (row, col) grid indices."""
    col = int((world_xy[0] - origin[0]) / resolution)
    row = int((world_xy[1] - origin[1]) / resolution)
    return (row, col)


def grid_to_world(row_col, origin, resolution):
    """Convert (row, col) grid indices to (x, y) world coords (cell centre)."""
    x = origin[0] + (row_col[1] + 0.5) * resolution
    y = origin[1] + (row_col[0] + 0.5) * resolution
    return (x, y)
