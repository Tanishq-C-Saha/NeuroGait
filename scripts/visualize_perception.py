"""CP5 — Perception pipeline visualization.

Shows the full chain from raw depth image to the occupancy grid the RL policy
actually sees. Saves a 6-panel figure to perception_pipeline_debug.png.

Usage:
    ~/isaac-sim/kit/python/bin/python3 scripts/visualize_perception.py \\
      --task NeuroGait-Navigation-CP5-v0 --num_envs 1 --headless --steps 10
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# ── argument parsing (before AppLauncher) ─────────────────────────────────────

parser = argparse.ArgumentParser(description="CP5 perception pipeline debug.")
parser.add_argument("--task",     type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps",    type=int, default=10,
                    help="Warm-up steps before grabbing a frame")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────

import torch
import numpy as np
import gymnasium as gym

import matplotlib
matplotlib.use("Agg")  # non-interactive, works headless
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from isaaclab.utils.math import unproject_depth, transform_points, quat_inv, quat_apply
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401

# observation layout (must match _CP5ObservationsCfg.PolicyCfg order)
_OBS_LIN_VEL      = slice(0, 3)    # base_lin_vel
_OBS_GRAVITY      = slice(3, 6)    # projected_gravity
_OBS_GOAL_VEC     = slice(6, 9)    # goal_vector
_OBS_ROBOT_VEL    = slice(9, 12)   # robot_velocity
_OBS_OCC_GRID     = slice(12, 1612)  # occupancy_grid (40×40)

_GRID_SIZE_M  = 8.0
_RESOLUTION_M = 0.2
_N_CELLS      = 40
_MIN_H        = 0.05
_MAX_H        = 2.0


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    raw_gym = gym.make(args_cli.task, cfg=env_cfg)
    env     = SkrlVecEnvWrapper(raw_gym, ml_framework="torch")
    base    = raw_gym.unwrapped

    obs, _ = env.reset()

    # warm-up: let robot settle and sensor initialise
    print(f"[INFO] Warming up for {args_cli.steps} steps...")
    with torch.no_grad():
        for _ in range(args_cli.steps):
            action = torch.zeros(args_cli.num_envs,
                                 raw_gym.action_space.shape[-1],
                                 device=base.device)
            obs, _, _, _, _ = env.step(action)

    # ── 1. robot state ─────────────────────────────────────────────────────────
    robot_pos  = base.scene["robot"].data.root_pos_w[0].cpu().numpy()
    print(f"Robot pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")

    # ── 2. raw depth from camera sensor ────────────────────────────────────────
    camera = base.scene["camera"]
    depth  = camera.data.output["distance_to_image_plane"]   # (E, H, W) or (E, H, W, 1)
    K_mats = camera.data.intrinsic_matrices                   # (E, 3, 3)
    pos_w  = camera.data.pos_w                               # (E, 3)
    quat_w = getattr(camera.data, "quat_w_ros", camera.data.quat_w_world)  # (E, 4)

    # squeeze channel dim if present (CameraCfg adds it, RayCasterCamera doesn't)
    if depth.ndim == 4:
        depth = depth.squeeze(-1)

    depth_np = depth[0].cpu().numpy()   # (H, W)
    print(f"Depth shape: {depth_np.shape}, "
          f"valid range: [{float(depth_np[np.isfinite(depth_np) & (depth_np > 0)].min()):.2f}, "
          f"{float(depth_np[np.isfinite(depth_np)].max()):.2f}] m")

    valid_mask = np.isfinite(depth_np) & (depth_np > 0.1) & (depth_np < 10.0)

    # ── 3. unproject to camera-frame point cloud ────────────────────────────────
    points_cam   = unproject_depth(depth, K_mats)                       # (E, N, 3)
    E, N, _      = points_cam.shape
    points_world = transform_points(points_cam, pos_w, quat_w)         # (E, N, 3)

    robot_pos_t  = base.scene["robot"].data.root_pos_w       # (E, 3)
    robot_quat   = base.scene["robot"].data.root_quat_w      # (E, 4)
    points_rel   = points_world - robot_pos_t.unsqueeze(1)   # (E, N, 3)
    q_inv        = quat_inv(robot_quat).unsqueeze(1).expand(-1, N, -1)
    pts_robot    = quat_apply(q_inv, points_rel)             # (E, N, 3) robot frame

    pts_np       = pts_robot[0].cpu().numpy()                # (N, 3) for env 0

    # ── 4. height filter ────────────────────────────────────────────────────────
    h_mask     = (pts_np[:, 2] > _MIN_H) & (pts_np[:, 2] < _MAX_H)
    filtered   = pts_np[h_mask]
    print(f"Points: total={N}, after height filter={len(filtered)}")

    # ── 5. manual occupancy grid (reference) ────────────────────────────────────
    half = _GRID_SIZE_M / 2.0
    manual_grid = np.zeros((_N_CELLS, _N_CELLS), dtype=np.float32)
    center = _N_CELLS // 2
    for p in filtered:
        col = center + int(round(p[0] / _RESOLUTION_M))
        row = center - int(round(p[1] / _RESOLUTION_M))
        if 0 <= row < _N_CELLS and 0 <= col < _N_CELLS:
            manual_grid[row, col] = 1.0

    # ── 6. policy's actual grid (from the RL observation) ───────────────────────
    obs_np     = obs[0].cpu().numpy()                       # (1612,)
    policy_grid = obs_np[_OBS_OCC_GRID].reshape(_N_CELLS, _N_CELLS)
    goal_vec    = obs_np[_OBS_GOAL_VEC]
    lin_vel     = obs_np[_OBS_LIN_VEL]
    robot_vel   = obs_np[_OBS_ROBOT_VEL]

    print(f"Manual grid occupied: {int(manual_grid.sum())} cells")
    print(f"Policy grid occupied: {int(policy_grid.sum())} cells")
    print(f"Goal vector:          {goal_vec}")
    print(f"Lin vel obs:          {lin_vel}")

    # ── 7. plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("NeuroGait CP5 — Perception Pipeline Debug", fontsize=14, fontweight="bold")
    bw = ListedColormap(["white", "black"])

    # Panel 1: raw depth
    ax = axes[0, 0]
    d_disp = depth_np.copy()
    d_disp[~np.isfinite(d_disp)] = 0.0
    im = ax.imshow(d_disp, cmap="plasma", vmin=0, vmax=5.0)
    plt.colorbar(im, ax=ax, label="Distance (m)")
    ax.set_title(f"1. Raw Depth ({depth_np.shape[1]}×{depth_np.shape[0]})")

    # Panel 2: full point cloud top-down (robot frame)
    ax = axes[0, 1]
    if len(pts_np):
        ax.scatter(pts_np[:, 0], pts_np[:, 1], c=pts_np[:, 2].clip(0, 3),
                   cmap="viridis", s=0.5, alpha=0.4)
    ax.plot(0, 0, "g*", markersize=14, label="robot")
    ax.set_title(f"2. Full Point Cloud ({N} pts)")
    ax.set_xlabel("X fwd (m)"); ax.set_ylabel("Y left (m)")
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3); ax.legend()

    # Panel 3: height-filtered
    ax = axes[0, 2]
    if len(filtered):
        ax.scatter(filtered[:, 0], filtered[:, 1], c="red", s=1, alpha=0.7)
    ax.plot(0, 0, "g*", markersize=14)
    ax.set_title(f"3. Height Filtered [{_MIN_H}–{_MAX_H} m]  ({len(filtered)} pts)")
    ax.set_xlabel("X fwd (m)"); ax.set_ylabel("Y left (m)")
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Panel 4: manual occupancy grid
    ax = axes[1, 0]
    ax.imshow(manual_grid, cmap=bw, origin="lower",
              extent=[-half, half, -half, half])
    ax.plot(0, 0, "go", markersize=10)
    ax.set_title(f"4. Manual Grid ({int(manual_grid.sum())} cells)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.grid(True, alpha=0.2)

    # Panel 5: policy's actual grid
    ax = axes[1, 1]
    ax.imshow(policy_grid, cmap=bw, origin="lower",
              extent=[-half, half, -half, half])
    ax.plot(0, 0, "go", markersize=10)
    # draw goal direction from goal_vector obs
    if abs(goal_vec[0]) + abs(goal_vec[1]) > 0.01:
        ax.annotate("", xy=(goal_vec[0] * 2, goal_vec[1] * 2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.set_title(f"5. Policy Grid ({int(policy_grid.sum())} cells)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.grid(True, alpha=0.2)

    # Panel 6: stats
    ax = axes[1, 2]
    ax.axis("off")
    diff = int(np.abs(manual_grid - policy_grid).sum())
    stats = (
        f"Pipeline Stats\n{'='*38}\n\n"
        f"Robot pos:        ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})\n"
        f"Depth shape:      {depth_np.shape}\n"
        f"Valid pixels:     {int(valid_mask.sum())}/{depth_np.size}\n\n"
        f"Raw cloud pts:    {N}\n"
        f"Height filtered:  {len(filtered)}\n\n"
        f"Manual grid occ:  {int(manual_grid.sum())}\n"
        f"Policy grid occ:  {int(policy_grid.sum())}\n"
        f"Grid diff cells:  {diff}\n\n"
        f"Goal vec obs:     [{goal_vec[0]:.3f}, {goal_vec[1]:.3f}, {goal_vec[2]:.3f}]\n"
        f"  dir ({goal_vec[0]:.2f}, {goal_vec[1]:.2f}), "
        f"dist {goal_vec[2]*10:.1f} m\n\n"
        f"Lin vel obs:      [{lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}]\n"
        f"Robot vel obs:    [{robot_vel[0]:.3f}, {robot_vel[1]:.3f}, {robot_vel[2]:.3f}]"
    )
    ax.text(0.05, 0.97, stats, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = "perception_pipeline_debug.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")

    env.close()
    raw_gym.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
