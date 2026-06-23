"""
Navigation observation terms for Isaac Lab manager-based envs.

CP3 functions (occupancy_grid_obs, occupancy_grid_obs_gpu) — use scene["camera"].
CP5 functions — use scene["front_cam"] (MultiMeshRayCasterCamera) + waypoint state.

CP5 observation layout (1615 dims total):
  [0    : 1600]  occupancy_grid_obs_cp5  — 40×40 asymmetric grid (robot at row 10)
  [1600 : 1609]  future_waypoints_obs    — next 3 waypoints in robot frame (9 dims)
  [1609 : 1612]  robot_velocity_obs      — [vx, vy, yaw_rate] (3 dims)
  [1612 : 1615]  projected_gravity       — standard Isaac Lab obs term (3 dims)

Ordering is CRITICAL: CNN in models/navigation_policy.py splits at index 1600.
"""

from __future__ import annotations

import math
import sys
import os
import numpy as np
import torch
from typing import TYPE_CHECKING

# Allow importing the concept module even without a package install
_concept_dir = os.path.join(os.path.dirname(__file__), "..", "concept")
if _concept_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_concept_dir))

from occupancy_grid import points_to_occupancy_grid

from isaaclab.utils.math import unproject_depth, transform_points, quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ── grid constants (must match verify_cp3.py) ─────────────────────────────────
_GRID_SIZE_M       = 8.0   # 8 m × 8 m arena centred on robot
_RESOLUTION_M      = 0.2   # 0.2 m per cell → 40 × 40 = 1600 cells
_MIN_HEIGHT_M      = 0.05  # filter out near-ground returns
_MAX_HEIGHT_M      = 2.0   # filter out sky / ceiling returns
_N_CELLS           = int(_GRID_SIZE_M / _RESOLUTION_M)   # 40
_FLAT_OBS_DIM      = _N_CELLS * _N_CELLS                 # 1600


def occupancy_grid_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Observation term: depth camera → 40×40 occupancy grid, flattened.

    Pipeline per step:
      depth (num_envs, H, W) + K (num_envs, 3, 3)
        → camera-frame 3D points   [unproject_depth]
        → world-frame 3D points    [transform_points using camera pose]
        → robot-frame 3D points    [subtract robot pos, rotate by quat_inv]
        → 40×40 binary grid        [points_to_occupancy_grid, vectorised numpy]
        → (num_envs, 1600) tensor  [flatten + to torch float32]

    Returns zeros on first step if camera pose contains NaN (known Isaac Lab
    bug: https://github.com/isaac-sim/IsaacLab/issues/3004).
    """
    camera = env.scene["camera"]
    robot  = env.scene["robot"]

    # ── 1. read raw depth and intrinsics ─────────────────────────────────────
    # depth  : (num_envs, H, W)   — distance to image plane in metres
    # K_mats : (num_envs, 3, 3)   — per-env intrinsic matrix
    depth  = camera.data.output["distance_to_image_plane"]   # (E, H, W)
    K_mats = camera.data.intrinsic_matrices                   # (E, 3, 3)

    # ── 2. NaN guard — camera pose is NaN on the very first physics step ─────
    pos_w  = camera.data.pos_w         # (E, 3)
    quat_w = camera.data.quat_w_ros    # (E, 4)

    if torch.isnan(pos_w).any() or torch.isnan(quat_w).any():
        return torch.zeros(env.num_envs, _FLAT_OBS_DIM,
                           device=env.device, dtype=torch.float32)

    # ── 3. unproject depth → camera-frame 3D points ──────────────────────────
    # unproject_depth returns (E, N, 3) already flattened, where N = H*W.
    #   pt_cam = K⁻¹ · [u, v, 1]ᵀ · depth[u,v]
    points_cam = unproject_depth(depth, K_mats)   # (E, N, 3)

    # ── 4. camera frame → world frame ────────────────────────────────────────
    # transform_points applies the rigid transform T = (pos_w, quat_w):
    #   pt_world = quat_apply(quat_w, pt_cam) + pos_w
    E, N, _ = points_cam.shape   # N = H*W, points already flattened
    points_world = transform_points(points_cam, pos_w, quat_w)  # (E, N, 3)

    # ── 5. world frame → robot frame ─────────────────────────────────────────
    # Robot frame is defined by the base link pose: subtract its position,
    # then rotate by the inverse of its orientation quaternion.
    robot_pos  = robot.data.root_pos_w    # (E, 3)
    robot_quat = robot.data.root_quat_w   # (E, 4)

    # subtract robot origin (broadcast over N points)
    points_rel = points_world - robot_pos.unsqueeze(1)          # (E, N, 3)

    # rotate by quat⁻¹ to go from world axes to robot-body axes
    quat_inv_r = quat_inv(robot_quat)                           # (E, 4)
    quat_inv_r = quat_inv_r.unsqueeze(1).expand(-1, N, -1)     # (E, N, 4)
    points_robot = quat_apply(quat_inv_r, points_rel)           # (E, N, 3)

    # ── 6. build occupancy grid per environment ───────────────────────────────
    # points_to_occupancy_grid works on CPU numpy arrays; we loop over envs.
    # At 10 Hz with E=1 (play mode) this costs ~4.5 ms total — acceptable.
    # For training with many envs, use occupancy_grid_obs_gpu() below instead.
    grids = []
    for i in range(E):
        pts_np = points_robot[i].cpu().numpy()    # (N, 3)
        g = points_to_occupancy_grid(
            pts_np, _GRID_SIZE_M, _RESOLUTION_M,
            max_height_m=_MAX_HEIGHT_M,
            min_height_m=_MIN_HEIGHT_M,
        )                                          # (40, 40), uint8
        grids.append(g.ravel())                    # (1600,)

    grid_np = np.stack(grids, axis=0)             # (E, 1600)
    return torch.tensor(grid_np, dtype=torch.float32, device=env.device)


# ── GPU-native path (no numpy, no per-env loop) ───────────────────────────────
# Use this in ObsTerm instead of occupancy_grid_obs when training with many
# parallel environments — it keeps all computation on the GPU device.

def occupancy_grid_obs_gpu(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Same output as occupancy_grid_obs but implemented entirely in torch so
    the entire pipeline stays on the GPU.  Returns (num_envs, 1600) float32.
    """
    camera = env.scene["camera"]
    robot  = env.scene["robot"]

    depth  = camera.data.output["distance_to_image_plane"]
    K_mats = camera.data.intrinsic_matrices
    pos_w  = camera.data.pos_w
    quat_w = camera.data.quat_w_ros

    if torch.isnan(pos_w).any() or torch.isnan(quat_w).any():
        return torch.zeros(env.num_envs, 1600, device=env.device, dtype=torch.float32)

    points_cam   = unproject_depth(depth, K_mats)                       # (E, N, 3)
    E, N, _      = points_cam.shape                                     # N = H*W, already flat
    points_world = transform_points(points_cam, pos_w, quat_w)         # (E, N, 3)

    robot_pos  = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    points_rel = points_world - robot_pos.unsqueeze(1)
    q_inv      = quat_inv(robot_quat).unsqueeze(1).expand(-1, N, -1)
    pts_robot  = quat_apply(q_inv, points_rel)                         # (E, N, 3)

    # ── torch binning ─────────────────────────────────────────────────────────
    n_cells     = int(_GRID_SIZE_M / _RESOLUTION_M)   # 40
    center_cell = n_cells // 2                         # 20

    x = pts_robot[..., 0]   # (E, N)
    y = pts_robot[..., 1]
    z = pts_robot[..., 2]

    # height filter
    valid = (z >= _MIN_HEIGHT_M) & (z <= _MAX_HEIGHT_M)   # (E, N) bool

    # cell indices
    cols = center_cell + torch.round(x / _RESOLUTION_M).long()   # (E, N)
    rows = center_cell - torch.round(y / _RESOLUTION_M).long()

    in_bounds = (
        valid
        & (rows >= 0) & (rows < n_cells)
        & (cols >= 0) & (cols < n_cells)
    )   # (E, N)

    # flatten (row, col) → linear index into a 1600-element grid
    linear_idx = rows * n_cells + cols                  # (E, N)
    linear_idx = linear_idx.clamp(0, n_cells * n_cells - 1)
    linear_idx[~in_bounds] = 0                          # sentinel; will be masked

    # scatter 1s into a (E, 1600) grid
    grids = torch.zeros(E, n_cells * n_cells, device=env.device, dtype=torch.float32)
    grids.scatter_(1, linear_idx, in_bounds.float())

    return grids


# ─────────────────────────────────────────────────────────────────────────────
# CP5 observation functions
# ─────────────────────────────────────────────────────────────────────────────

# Grid constants (same as CP3 but asymmetric robot placement)
_CP5_GRID_SIZE_M  = 8.0
_CP5_RESOLUTION_M = 0.2
_CP5_N_CELLS      = int(_CP5_GRID_SIZE_M / _CP5_RESOLUTION_M)   # 40
_CP5_ROBOT_ROW    = 10    # 2 m behind, 6 m ahead (see concept/06_asymmetric_grid.md)
_CP5_ROBOT_COL    = _CP5_N_CELLS // 2   # 20 — centred laterally
_CP5_MIN_H        = 0.05
_CP5_MAX_H        = 2.0

_CP5_WAYPOINT_ADVANCE_DIST = 0.3   # m — advance to next waypoint when within this radius
_CP5_WP_LOOKAHEAD          = 3     # number of future waypoints to encode
_CP5_GOAL_XY               = (8.0, 0.0)


def _cp5_init_waypoint_state(env) -> None:
    """Lazy-init A* waypoint state on the env object.

    Runs once on first call.  All envs share the same path (same obstacles).
    Waypoint state tensors are stored as attributes on env so reward and obs
    functions can share them without re-running A* every step.
    """
    if hasattr(env, "_cp5_waypoints"):
        return   # already initialised

    from neurogait.tasks.manager_based.navigation.planning.global_grid import build_global_grid
    from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner

    robot = env.scene["robot"]
    robot_pos_np = robot.data.root_pos_w[0].cpu().numpy()
    start_xy = (float(robot_pos_np[0]), float(robot_pos_np[1]))

    grid, origin, _ = build_global_grid(env)
    planner = AStarPlanner(grid, origin, resolution=_CP5_RESOLUTION_M)
    waypoints = planner.plan(start_xy, _CP5_GOAL_XY)

    if not waypoints:
        # Fallback: straight line from start to goal at 1 m intervals
        import numpy as np
        dx = _CP5_GOAL_XY[0] - start_xy[0]
        dy = _CP5_GOAL_XY[1] - start_xy[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        n = max(int(dist), 2)
        waypoints = [(start_xy[0] + dx * i / n, start_xy[1] + dy * i / n)
                     for i in range(1, n + 1)]
        print("[CP5] Warning: A* failed, using straight-line fallback path")

    env._cp5_waypoints = torch.tensor(waypoints, dtype=torch.float32, device=env.device)  # (W, 2)
    env._cp5_wp_idx    = torch.zeros(env.num_envs, dtype=torch.long,    device=env.device)  # (E,)
    env._cp5_prev_dist = torch.full((env.num_envs,), float("inf"),
                                    dtype=torch.float32, device=env.device)
    env._cp5_prev_action = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    # Position history for stuck detection: (E, 20, 2)
    env._cp5_pos_history = torch.zeros(env.num_envs, 20, 2, dtype=torch.float32, device=env.device)
    env._cp5_pos_hist_idx = 0
    print(f"[CP5] Waypoint state initialised: {len(waypoints)} waypoints, "
          f"goal={_CP5_GOAL_XY}")


def _cp5_reset_waypoint_state(env, env_ids) -> None:
    """Reset waypoint tracking for terminated environments."""
    if not hasattr(env, "_cp5_wp_idx"):
        return
    env._cp5_wp_idx[env_ids] = 0
    env._cp5_prev_dist[env_ids] = float("inf")
    env._cp5_prev_action[env_ids] = 0.0
    env._cp5_pos_history[env_ids] = 0.0


def quat_to_yaw_batch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Batch quaternion (w,x,y,z) → yaw angle. Shape: (E,4) → (E,)."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wp_in_robot_frame(robot_pos: torch.Tensor, robot_yaw: torch.Tensor,
                       wp_world: torch.Tensor) -> torch.Tensor:
    """Transform waypoints from world to robot frame.

    Args:
        robot_pos:  (E, 2) world x,y
        robot_yaw:  (E,)   yaw in radians
        wp_world:   (E, 2) waypoint world x,y

    Returns:
        (E, 3) [dir_x, dir_y, norm_dist]  — normalised direction + distance/10
    """
    delta = wp_world - robot_pos    # (E, 2)
    cos_y = torch.cos(-robot_yaw)
    sin_y = torch.sin(-robot_yaw)
    dx_r  =  cos_y * delta[:, 0] + sin_y * delta[:, 1]
    dy_r  = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
    dist  = torch.sqrt(dx_r ** 2 + dy_r ** 2).clamp(min=1e-4)
    dir_x = dx_r / dist
    dir_y = dy_r / dist
    norm_dist = (dist / 10.0).clamp(max=1.0)
    return torch.stack([dir_x, dir_y, norm_dist], dim=-1)   # (E, 3)


def occupancy_grid_obs_cp5(env) -> torch.Tensor:
    """CP5 asymmetric 40×40 occupancy grid, robot placed at row 10.

    Uses scene["front_cam"] (MultiMeshRayCasterCamera).
    Returns (num_envs, 1600) float32.
    """
    _cp5_init_waypoint_state(env)

    camera = env.scene["front_cam"]
    robot  = env.scene["robot"]

    depth  = camera.data.output["distance_to_image_plane"]   # (E, H, W)
    K_mats = camera.data.intrinsic_matrices                   # (E, 3, 3)
    pos_w  = camera.data.pos_w
    quat_w = camera.data.quat_w_ros

    if torch.isnan(pos_w).any() or torch.isnan(quat_w).any():
        return torch.zeros(env.num_envs, _CP5_N_CELLS ** 2,
                           device=env.device, dtype=torch.float32)

    from isaaclab.utils.math import unproject_depth, transform_points, quat_inv, quat_apply

    points_cam   = unproject_depth(depth, K_mats)                         # (E, N, 3)
    E, N, _      = points_cam.shape
    points_world = transform_points(points_cam, pos_w, quat_w)            # (E, N, 3)

    robot_pos  = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    pts_rel    = points_world - robot_pos.unsqueeze(1)
    q_inv      = quat_inv(robot_quat).unsqueeze(1).expand(-1, N, -1)
    pts_robot  = quat_apply(q_inv, pts_rel)                               # (E, N, 3)

    x = pts_robot[..., 0]
    y = pts_robot[..., 1]
    z = pts_robot[..., 2]
    valid = (z >= _CP5_MIN_H) & (z <= _CP5_MAX_H)

    # Asymmetric placement: robot at (_CP5_ROBOT_ROW, _CP5_ROBOT_COL)
    # Forward = +x → increases col; Left = +y → decreases row
    cols = _CP5_ROBOT_COL + torch.round(x / _CP5_RESOLUTION_M).long()
    rows = _CP5_ROBOT_ROW - torch.round(y / _CP5_RESOLUTION_M).long()

    in_bounds = (
        valid
        & (rows >= 0) & (rows < _CP5_N_CELLS)
        & (cols >= 0) & (cols < _CP5_N_CELLS)
    )

    linear_idx = rows * _CP5_N_CELLS + cols
    linear_idx = linear_idx.clamp(0, _CP5_N_CELLS ** 2 - 1)
    linear_idx[~in_bounds] = 0

    grids = torch.zeros(E, _CP5_N_CELLS ** 2, device=env.device, dtype=torch.float32)
    grids.scatter_(1, linear_idx, in_bounds.float())
    return grids


def future_waypoints_obs(env) -> torch.Tensor:
    """Next 3 waypoints encoded in robot frame.

    Returns (num_envs, 9): [dir_x, dir_y, norm_dist] × 3 waypoints.
    Provides implicit lookahead — policy sees upcoming turns before reaching
    the current waypoint (pure pursuit principle).
    """
    _cp5_init_waypoint_state(env)

    robot     = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]                   # (E, 2)
    robot_yaw = quat_to_yaw_batch(robot.data.root_quat_w)      # (E,)

    # Advance waypoint index when within threshold
    W = len(env._cp5_waypoints)
    curr_wp  = env._cp5_waypoints[env._cp5_wp_idx.clamp(max=W - 1)]   # (E, 2)
    dist_now = torch.norm(curr_wp - robot_pos, dim=-1)                 # (E,)
    advance  = (dist_now < _CP5_WAYPOINT_ADVANCE_DIST) & (env._cp5_wp_idx < W - 1)
    env._cp5_wp_idx[advance] += 1

    # Encode 3 future waypoints
    obs_parts = []
    for offset in range(_CP5_WP_LOOKAHEAD):
        idx = (env._cp5_wp_idx + offset).clamp(max=W - 1)   # (E,)
        wp  = env._cp5_waypoints[idx]                         # (E, 2)
        obs_parts.append(_wp_in_robot_frame(robot_pos, robot_yaw, wp))

    return torch.cat(obs_parts, dim=-1)   # (E, 9)


def robot_velocity_obs(env) -> torch.Tensor:
    """Robot base velocity in body frame: [vx, vy, yaw_rate]. Shape (E, 3)."""
    robot  = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_b[:, :2]   # (E, 2)  — body-frame xy
    yaw_rate = robot.data.root_ang_vel_b[:, 2:3] # (E, 1)  — body-frame z (yaw)
    return torch.cat([lin_vel, yaw_rate], dim=-1) # (E, 3)
