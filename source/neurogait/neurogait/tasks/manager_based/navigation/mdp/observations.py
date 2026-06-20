"""
CP3 — Occupancy grid observation term for Isaac Lab manager-based envs.

Plugs into ObservationGroupCfg as a regular observation term:

    from neurogait.tasks.manager_based.navigation.mdp import occupancy_grid_obs

    @configclass
    class MyObsCfg(ObsGroup):
        occupancy_grid = ObsTerm(func=occupancy_grid_obs)   # → (num_envs, 1600)

The function reads the depth camera that must be added to scene["camera"]
(already done in navigation_env_cfg.py).  Output is a flattened 40×40
binary occupancy grid (1600 floats), one per environment.
"""

from __future__ import annotations

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


# ── CP5: navigation policy observations ───────────────────────────────────────


def _robot_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from Isaac Lab (w, x, y, z) quaternion batch."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def goal_vector_obs(env) -> torch.Tensor:
    """Goal vector in robot frame: [dir_x, dir_y, norm_distance].

    Returns (num_envs, 3) float32. Normalised direction to current A* waypoint
    in the robot's base frame, plus distance clamped to [0,1] (10 m range).
    Returns zeros if waypoints are not initialised yet.
    """
    if not hasattr(env, "_curr_waypoint_pos"):
        return torch.zeros(env.num_envs, 3, device=env.device)

    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    robot_quat = env.scene["robot"].data.root_quat_w
    yaw = _robot_yaw(robot_quat)

    dx = env._curr_waypoint_pos[:, 0] - robot_xy[:, 0]
    dy = env._curr_waypoint_pos[:, 1] - robot_xy[:, 1]
    dist = torch.sqrt(dx * dx + dy * dy).clamp(min=0.1)

    # rotate world-frame delta into robot frame
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy

    dir_x = local_x / dist
    dir_y = local_y / dist
    norm_dist = dist.clamp(max=10.0) / 10.0

    return torch.stack([dir_x, dir_y, norm_dist], dim=-1)


def robot_velocity_obs(env) -> torch.Tensor:
    """Robot base velocity in base frame: [vx, vy, yaw_rate].

    Returns (num_envs, 3) float32.
    """
    lin_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]   # (E, 2)
    ang_vel = env.scene["robot"].data.root_ang_vel_b[:, 2:3]  # (E, 1)
    return torch.cat([lin_vel, ang_vel], dim=-1)
