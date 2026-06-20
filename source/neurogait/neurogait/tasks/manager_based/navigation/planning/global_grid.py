"""
Build a 2D global occupancy grid from env.scene rigid object positions.

Convention:
    row  ↔  Y axis   (row 0 = min Y = origin_y)
    col  ↔  X axis   (col 0 = min X = origin_x)
"""

import numpy as np


_INFLATION_M = 0.2   # extra clearance around every obstacle (one cell at 0.2 m/cell)
_DEFAULT_RADIUS = 0.5  # fallback footprint if size cannot be read from config


def build_global_grid(env, grid_resolution=0.2, grid_size=200):
    """
    Reads obstacle positions from env.scene rigid objects and rasterises
    them into a 2D binary occupancy grid.

    Args:
        env            : ManagerBasedRLEnv instance
        grid_resolution: metres per cell
        grid_size      : cells per side

    Returns:
        grid   : np.ndarray (grid_size, grid_size) uint8, 0=free 1=occupied
        origin : (float, float) world (x, y) of grid cell (0, 0)
    """
    print("[CP4] Building global occupancy grid...")

    half = grid_size * grid_resolution / 2.0
    origin = (-half, -half)

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    rigid_objects = env.scene.rigid_objects  # dict[str, RigidObject]
    n_rasterized = 0

    for name, obj in rigid_objects.items():
        name_lower = name.lower()
        if "robot" in name_lower:
            continue
        if not any(k in name_lower for k in ("obstacle", "cube", "cyl")):
            continue

        # world (x, y) of this obstacle in env 0
        pos_w = obj.data.root_pos_w[0].cpu().numpy()
        ox, oy = float(pos_w[0]), float(pos_w[1])

        # footprint radius from spawn config
        radius = _get_footprint_radius(obj)
        inflate_r = radius + _INFLATION_M

        # bounding box of cells to check
        c_min = max(0, int((ox - inflate_r - origin[0]) / grid_resolution))
        c_max = min(grid_size - 1, int((ox + inflate_r - origin[0]) / grid_resolution))
        r_min = max(0, int((oy - inflate_r - origin[1]) / grid_resolution))
        r_max = min(grid_size - 1, int((oy + inflate_r - origin[1]) / grid_resolution))

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                # world centre of this cell
                cx = origin[0] + (c + 0.5) * grid_resolution
                cy = origin[1] + (r + 0.5) * grid_resolution
                if (cx - ox) ** 2 + (cy - oy) ** 2 <= inflate_r ** 2:
                    grid[r, c] = 1

        n_rasterized += 1

    print(f"[CP4] Grid built: {grid_size}x{grid_size}, {n_rasterized} obstacles rasterized")
    return grid, origin


def _get_footprint_radius(obj):
    """Return the half-width footprint radius of a rigid object."""
    try:
        spawn = obj.cfg.spawn
        # CuboidCfg has .size = (sx, sy, sz)
        if hasattr(spawn, "size"):
            sx, sy, _ = spawn.size
            return max(sx, sy) / 2.0
        # CylinderCfg / SphereCfg have .radius
        if hasattr(spawn, "radius"):
            return float(spawn.radius)
    except Exception:
        pass
    return _DEFAULT_RADIUS


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
