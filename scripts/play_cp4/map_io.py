"""Persist the occupancy grid and A* path to the maps/ directory.

No simulator imports — safe to call at any point.
"""

import csv
import math
import os

import numpy as np

# maps/ lives at the project root (two levels above this file: play_cp4/ → scripts/ → project)
_MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "maps")


def save_map(grid, origin, resolution, path_world, start_xy, goal_xy, obstacle_info=None):
    """Save occupancy grid + A* path as a .npy, a .csv, and a PNG."""
    os.makedirs(_MAPS_DIR, exist_ok=True)

    npy_path = os.path.join(_MAPS_DIR, "global_grid.npy")
    np.save(npy_path, grid)
    print(f"[CP4] Grid array saved  → {npy_path}")

    if path_world:
        csv_path = os.path.join(_MAPS_DIR, "path_waypoints.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(path_world)
        print(f"[CP4] Path CSV saved    → {csv_path}")

    _save_png(grid, origin, resolution, path_world, start_xy, goal_xy, obstacle_info)


def _save_png(grid, origin, resolution, path_world, start_xy, goal_xy, obstacle_info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_x = [start_xy[0], goal_xy[0]]
        all_y = [start_xy[1], goal_xy[1]]
        if obstacle_info:
            all_x += [o["x"] for o in obstacle_info]
            all_y += [o["y"] for o in obstacle_info]
        if path_world:
            all_x += [p[0] for p in path_world]
            all_y += [p[1] for p in path_world]

        pad = 1.5
        view_xmin, view_xmax = min(all_x) - pad, max(all_x) + pad
        view_ymin, view_ymax = min(all_y) - pad, max(all_y) + pad

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("white")

        rows, cols = grid.shape
        extent = [
            origin[0], origin[0] + cols * resolution,
            origin[1], origin[1] + rows * resolution,
        ]
        ax.imshow(grid, origin="lower", cmap="gray_r", extent=extent,
                  vmin=0, vmax=1, zorder=1)

        if path_world and len(path_world) >= 2:
            xs = [p[0] for p in path_world]
            ys = [p[1] for p in path_world]
            ax.plot(xs, ys, color="#1565C0", linewidth=2.0, zorder=3,
                    solid_capstyle="round", label="A* path")

        ax.scatter(*start_xy, s=180, color="#00C853", zorder=5,
                   edgecolors="black", linewidths=1.2, label="Start")
        ax.scatter(*goal_xy,  s=180, color="#D32F2F", zorder=5,
                   edgecolors="black", linewidths=1.2, label="Goal")

        ax.set_xlim(view_xmin, view_xmax)
        ax.set_ylim(view_ymin, view_ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("X  (m)", fontsize=11)
        ax.set_ylabel("Y  (m)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, color="#cccccc", linewidth=0.4, linestyle="--")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        n_wp  = len(path_world) if path_world else 0
        n_obs = len(obstacle_info) if obstacle_info else 0
        d     = math.sqrt((goal_xy[0] - start_xy[0])**2 + (goal_xy[1] - start_xy[1])**2)
        ax.set_title(
            f"NeuroGait CP4 — Global Occupancy Grid + A* Path\n"
            f"start=({start_xy[0]:.1f}, {start_xy[1]:.1f})  "
            f"goal=({goal_xy[0]:.1f}, {goal_xy[1]:.1f})  "
            f"d={d:.1f} m   waypoints={n_wp}   obstacles={n_obs}",
            fontsize=10,
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        sb_x0 = view_xmax - 1.8
        sb_y0 = view_ymin + 0.4
        ax.plot([sb_x0, sb_x0 + 1.0], [sb_y0, sb_y0], "k-", lw=2.5, zorder=6)
        ax.text(sb_x0 + 0.5, sb_y0 + 0.15, "1 m",
                ha="center", va="bottom", fontsize=8, zorder=6)

        plt.tight_layout()
        png_path = os.path.join(_MAPS_DIR, "global_grid.png")
        plt.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[CP4] Map image saved → {png_path}")

    except ImportError:
        print("[CP4] matplotlib not available — PNG skipped, raw .npy saved")
    except Exception as exc:
        print(f"[CP4] Map rendering error: {exc}")
