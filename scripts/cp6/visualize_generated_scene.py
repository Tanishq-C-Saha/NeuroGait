#!/usr/bin/env python3
"""
Visualise generated scenes across 6 difficulty levels.

Usage:
    python scripts/cp6/visualize_generated_scene.py

Produces a 6-panel matplotlib figure (2 rows × 3 cols) showing:
  - robot path as a smooth curve
  - corridor half-width shaded in transparent green
  - obstacles as coloured rectangles
  - goal marker

Difficulty levels correspond to curriculum progress: 0%, 20%, 40%, 60%, 80%, 100%.
"""

from __future__ import annotations

import sys
import os

# Allow import without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "neurogait"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from neurogait.tasks.manager_based.navigation.scene import (
    NavigationCurriculum,
    generate_scene,
    MIN_CORRIDOR,
)


_PROGRESS_LEVELS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
_RAMP_STEPS = 24_576_000

# Distinct colours for obstacles
_OBS_COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
    "#3498db", "#9b59b6", "#1abc9c", "#e91e63",
    "#ff5722", "#795548", "#607d8b", "#009688",
]


def _draw_panel(ax: plt.Axes, progress: float, seed: int) -> None:
    curriculum = NavigationCurriculum(ramp_steps=_RAMP_STEPS)
    current_steps = int(progress * _RAMP_STEPS)
    d = curriculum.get_difficulty(current_steps)

    path_points, obstacles, waypoints = generate_scene(
        start_xy=(0.0, 0.0),
        goal_xy=(8.0, 0.0),
        num_obstacles=d["num_obstacles"],
        corridor_width=d["corridor_width"],
        num_control_points=d["num_control_points"],
        max_lateral_deviation=d["max_lateral_deviation"],
        arena_bounds=(-1.0, 9.0, -4.0, 4.0),
        seed=seed,
    )

    # Corridor shading
    half_w = d["corridor_width"] / 2.0
    px, py = path_points[:, 0], path_points[:, 1]

    # Compute per-point normals for ribbon
    tangents = np.diff(path_points, axis=0)
    norms = np.sqrt(tangents[:, 0] ** 2 + tangents[:, 1] ** 2)[:, None]
    tangents = np.divide(tangents, norms, where=norms > 0)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    midpts = (path_points[:-1] + path_points[1:]) / 2

    upper = midpts + normals * half_w
    lower = midpts - normals * half_w

    corridor_x = np.concatenate([upper[:, 0], lower[::-1, 0]])
    corridor_y = np.concatenate([upper[:, 1], lower[::-1, 1]])
    ax.fill(corridor_x, corridor_y, alpha=0.25, color="#2ecc71", zorder=1)

    # Path
    ax.plot(px, py, color="#27ae60", linewidth=2, zorder=2, label="path")

    # Waypoints
    wp_arr = np.array(waypoints)
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], c="#16a085", s=20, zorder=3)

    # Obstacles
    for i, obs in enumerate(obstacles):
        ox, oy, oz = obs["pos"]
        sx, sy, _  = obs["size"]
        color = _OBS_COLORS[i % len(_OBS_COLORS)]
        rect = patches.Rectangle(
            (ox - sx / 2, oy - sy / 2), sx, sy,
            linewidth=1, edgecolor="black", facecolor=color, alpha=0.8, zorder=4,
        )
        ax.add_patch(rect)

    # Start / Goal
    ax.scatter([0.0], [0.0], marker="o", s=80, c="#3498db", zorder=5, label="start")
    ax.scatter([8.0], [0.0], marker="*", s=160, c="#e74c3c", zorder=5, label="goal")

    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title(
        f"{progress*100:.0f}% — corridor={d['corridor_width']:.2f}m, "
        f"obs={d['num_obstacles']}, ctrl-pts={d['num_control_points']}",
        fontsize=8,
    )
    ax.set_xlabel("X (m)", fontsize=7)
    ax.set_ylabel("Y (m)", fontsize=7)
    ax.tick_params(labelsize=6)

    # MIN_CORRIDOR annotation on last panel only
    if progress == 1.0:
        ax.axhline(MIN_CORRIDOR / 2, color="red", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axhline(-MIN_CORRIDOR / 2, color="red", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.text(8.5, MIN_CORRIDOR / 2, f"min={MIN_CORRIDOR:.2f}m", fontsize=5, color="red", va="bottom")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "CP6.5 Curriculum — Generated Scenes Across Difficulty Levels\n"
        f"Go2 body width={0.31}m  |  MIN_CORRIDOR={MIN_CORRIDOR:.2f}m  |  corridor = body + 2×0.10m",
        fontsize=10,
    )

    base_seed = 42
    for idx, (ax, progress) in enumerate(zip(axes.flat, _PROGRESS_LEVELS)):
        _draw_panel(ax, progress, seed=base_seed + idx * 7)

    # Shared legend from last axis
    handles, labels = axes[1, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=3)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints", "cp65_curriculum_scenes.png"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {os.path.abspath(out_path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
