"""CP5 — Visualise the perception pipeline for debugging.

Produces a 6-panel figure:
  1. Raw depth image (80×60)
  2. 3D point cloud (robot frame)
  3. Height-filtered points
  4. Manual 40×40 grid (symmetric, CP3-style)
  5. CP5 grid (asymmetric, robot at row 10)
  6. Stats: min/max depth, obstacle coverage, robot position

Run (requires display or headless with Agg backend):
    ~/isaac-sim/kit/python/bin/python3 scripts/cp5/visualize_perception.py \
      --task NeuroGait-Navigation-CP5-Play-v0 \
      --num_envs 1

Saves output to maps/cp5_perception_debug.png.
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CP5 perception visualisation")
parser.add_argument("--task",     type=str, default="NeuroGait-Navigation-CP5-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps",    type=int, default=5,
                    help="Warmup steps before capturing frame")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import unproject_depth, transform_points, quat_inv, quat_apply
from isaaclab_tasks.utils.hydra import hydra_task_config

import neurogait.tasks  # noqa: F401

_MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "maps")


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    env    = gym.make(args_cli.task, cfg=env_cfg)
    nav_env = env.unwrapped
    obs, _ = env.reset()

    # Warmup to let camera stabilise
    for _ in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        random_action = torch.zeros(1, 3, device=env_cfg.sim.device)
        obs, _, _, _, _ = env.step(random_action)

    # Capture one frame
    camera = nav_env.scene["front_cam"]
    robot  = nav_env.scene["robot"]

    depth  = camera.data.output["distance_to_image_plane"][0].cpu().numpy()   # (H, W)
    K      = camera.data.intrinsic_matrices[0].cpu().numpy()                   # (3, 3)
    pos_w  = camera.data.pos_w[0].cpu()
    quat_w = camera.data.quat_w_ros[0].cpu()
    robot_pos  = robot.data.root_pos_w[0].cpu()
    robot_quat = robot.data.root_quat_w[0].cpu()

    # Re-derive point cloud for visualisation
    depth_t   = camera.data.output["distance_to_image_plane"]   # (1, H, W)
    K_t       = camera.data.intrinsic_matrices                   # (1, 3, 3)
    pts_cam   = unproject_depth(depth_t, K_t)[0].cpu()           # (N, 3)
    pts_world = (transform_points(pts_cam.unsqueeze(0),
                                  camera.data.pos_w,
                                  camera.data.quat_w_ros)[0].cpu())
    q_inv     = quat_inv(robot.data.root_quat_w[0]).cpu()
    pts_rel   = pts_world - robot.data.root_pos_w[0].cpu()
    pts_robot = quat_apply(q_inv.unsqueeze(0).expand(pts_rel.shape[0], -1), pts_rel).numpy()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        ax = axes.ravel()

        # Panel 1: raw depth
        depth_disp = np.where(np.isnan(depth), 0, depth)
        ax[0].imshow(depth_disp, cmap="viridis")
        ax[0].set_title("1. Raw depth (m)")

        # Panel 2: 3D point cloud (robot frame, top-down xy)
        mask = ~np.isnan(pts_robot[:, 0])
        ax[1].scatter(pts_robot[mask, 0], pts_robot[mask, 1], s=0.5, alpha=0.3)
        ax[1].set_title("2. Point cloud (robot frame, top-down)")
        ax[1].set_aspect("equal"); ax[1].set_xlabel("X (m)"); ax[1].set_ylabel("Y (m)")

        # Panel 3: height-filtered
        z = pts_robot[:, 2]
        hf = mask & (z > 0.05) & (z < 2.0)
        ax[2].scatter(pts_robot[hf, 0], pts_robot[hf, 1], s=0.5, alpha=0.5, c="orange")
        ax[2].set_title("3. Height-filtered (0.05–2.0 m)")
        ax[2].set_aspect("equal")

        # Panel 4: symmetric 40×40 grid (CP3 style)
        n_cells, res, center = 40, 0.2, 20
        grid4 = np.zeros((n_cells, n_cells))
        xi = (pts_robot[hf, 0] / res).astype(int) + center
        yi = center - (pts_robot[hf, 1] / res).astype(int)
        valid = (xi >= 0) & (xi < n_cells) & (yi >= 0) & (yi < n_cells)
        grid4[yi[valid], xi[valid]] = 1
        ax[3].imshow(grid4, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        ax[3].set_title("4. Symmetric grid (robot at row 20)")
        ax[3].axhline(20, color="g", lw=1.5, label="robot row")

        # Panel 5: asymmetric 40×40 grid (CP5 style, robot at row 10)
        grid5 = np.zeros((n_cells, n_cells))
        xi5 = (pts_robot[hf, 0] / res).astype(int) + 20
        yi5 = 10 - (pts_robot[hf, 1] / res).astype(int)
        valid5 = (xi5 >= 0) & (xi5 < n_cells) & (yi5 >= 0) & (yi5 < n_cells)
        grid5[yi5[valid5], xi5[valid5]] = 1
        ax[4].imshow(grid5, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        ax[4].set_title("5. Asymmetric grid (robot at row 10)")
        ax[4].axhline(10, color="g", lw=1.5, label="robot row")

        # Panel 6: stats
        ax[5].axis("off")
        stats_text = (
            f"Depth: min={np.nanmin(depth):.2f}m  max={np.nanmax(depth):.2f}m\n"
            f"Valid depth pixels: {(~np.isnan(depth)).sum()}/{depth.size}\n"
            f"Pts in robot frame: {mask.sum()}\n"
            f"Height-filtered:    {hf.sum()}\n"
            f"Grid4 occupied:     {int(grid4.sum())}/1600\n"
            f"Grid5 occupied:     {int(grid5.sum())}/1600\n"
            f"\nRobot pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})\n"
            f"Camera pos: ({pos_w[0].item():.2f}, {pos_w[1].item():.2f})\n"
        )
        ax[5].text(0.05, 0.95, stats_text, va="top", fontfamily="monospace",
                   transform=ax[5].transAxes, fontsize=10)
        ax[5].set_title("6. Stats")

        plt.tight_layout()
        os.makedirs(_MAPS_DIR, exist_ok=True)
        out_path = os.path.join(_MAPS_DIR, "cp5_perception_debug.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[visualize] Saved → {out_path}")

    except ImportError:
        print("[visualize] matplotlib not available")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
