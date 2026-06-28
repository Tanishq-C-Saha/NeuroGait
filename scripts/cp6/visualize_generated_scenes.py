"""
Visualise obstacles-first scene generation across 6 curriculum difficulty levels.

Creates maps/cp6_5_obstacles_first_difficulty.png showing:
  - Black obstacles (cubes) and grey cylinders
  - Red dotted C-space inflation ring around each obstacle
  - Blue A* path threading through gaps
  - Green circles showing Go2 body width along path
  - Random goal direction and distance per panel

No Isaac Sim needed — runs standalone with plain Python + matplotlib.

Usage:
    python3 scripts/cp6/visualize_generated_scenes.py
"""

import os
import sys

# Direct import to avoid pulling in Isaac Lab / pxr
_SCENE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../source/neurogait/neurogait/tasks/manager_based/navigation",
    )
)
sys.path.insert(0, _SCENE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from scene.scene_generator import generate_scene, GO2_WIDTH, MIN_GAP
from scene.curriculum import difficulty_at


_PROGRESS_PCT = [0, 20, 40, 60, 80, 100]   # curriculum percentage per panel


def _draw_panel(
    ax: plt.Axes,
    obstacles: list[dict],
    waypoints: list[tuple],
    start_xy: tuple,
    goal_xy: tuple,
    title: str,
    min_gap: float,
) -> None:
    # ── Obstacles + C-space rings ─────────────────────────────────────────────
    for obs in obstacles:
        ox, oy  = obs["pos"][0], obs["pos"][1]
        sx, sy  = obs["size"][0], obs["size"][1]
        is_cube = obs["shape"] == "cube"
        body_color = "#3d3d3a" if is_cube else "#6b6b66"
        inflate = min_gap / 2.0

        if is_cube:
            body = patches.Rectangle(
                (ox - sx / 2, oy - sy / 2), sx, sy,
                facecolor=body_color, edgecolor="black", linewidth=0.5,
                alpha=0.85, zorder=3,
            )
        else:
            body = plt.Circle(
                (ox, oy), sx / 2,
                facecolor=body_color, edgecolor="black", linewidth=0.5,
                alpha=0.85, zorder=3,
            )
        ax.add_patch(body)

        # C-space inflation (dotted red box)
        ring = patches.Rectangle(
            (ox - sx / 2 - inflate, oy - sy / 2 - inflate),
            sx + 2 * inflate, sy + 2 * inflate,
            linewidth=0.6, edgecolor="#e74c3c", facecolor="none",
            linestyle=":", alpha=0.45, zorder=2,
        )
        ax.add_patch(ring)

    # ── A* path ───────────────────────────────────────────────────────────────
    if len(waypoints) >= 2:
        wx = [p[0] for p in waypoints]
        wy = [p[1] for p in waypoints]
        ax.plot(wx, wy, color="#2980b9", linewidth=1.8, zorder=4, alpha=0.85)
        ax.scatter(wx, wy, c="#2980b9", s=12, zorder=5, alpha=0.7)

    # ── Go2 body width along path (green circles) ─────────────────────────────
    for wp in waypoints[::3]:
        body_circle = plt.Circle(
            wp, GO2_WIDTH / 2,
            facecolor="#2ecc71", alpha=0.25, zorder=1, linewidth=0,
        )
        ax.add_patch(body_circle)

    # ── Start / Goal markers ──────────────────────────────────────────────────
    ax.scatter(*start_xy, s=160, c="#3498db", marker="o",
               edgecolors="black", linewidths=1.2, zorder=6)
    ax.scatter(*goal_xy, s=200, c="#e74c3c", marker="*",
               edgecolors="black", linewidths=1.0, zorder=6)

    # Faint arrow showing general start→goal direction
    ax.annotate(
        "", xy=goal_xy, xytext=start_xy,
        arrowprops=dict(arrowstyle="->", color="#e74c3c", alpha=0.15, lw=1.2),
    )

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlabel("X (m)", fontsize=7)
    ax.set_ylabel("Y (m)", fontsize=7)
    ax.grid(True, alpha=0.12)
    ax.tick_params(labelsize=6)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "CP6.5 — Obstacles-First Generator: Curriculum Progression\n"
        f"Go2 width={GO2_WIDTH}m | MIN_GAP={MIN_GAP:.2f}m | "
        "● start  ★ goal  — A* path  · C-space  ○ body width",
        fontsize=11, fontweight="bold",
    )

    start_xy = (0.0, 0.0)

    for ax, pct in zip(axes.flat, _PROGRESS_PCT):
        diff = difficulty_at(pct / 100.0)

        obstacles, waypoints, goal_xy = generate_scene(
            start_xy         = start_xy,
            goal_xy          = None,
            num_obstacles    = diff["num_obstacles"],
            min_gap_width    = diff["min_gap_width"],
            arena_padding    = diff["arena_padding"],
            goal_dist_range  = diff["goal_dist"],
            goal_angle_range = diff["goal_angle"],
            seed             = pct + 42,   # reproducible per panel
        )

        title = (
            f"{pct}% — gap={diff['min_gap_width']:.2f}m, "
            f"obs={diff['num_obstacles']}, "
            f"pad={diff['arena_padding']:.1f}m"
        )
        _draw_panel(ax, obstacles, waypoints, start_xy, goal_xy, title, diff["min_gap_width"])

        # Fit view to arena + small margin
        pad = diff["arena_padding"] + 0.8
        ax.set_xlim(min(start_xy[0], goal_xy[0]) - pad, max(start_xy[0], goal_xy[0]) + pad)
        ax.set_ylim(min(start_xy[1], goal_xy[1]) - pad, max(start_xy[1], goal_xy[1]) + pad)

    # Legend proxy artists on last axis
    last_ax = axes.flat[-1]
    legend_items = [
        patches.Patch(facecolor="#3d3d3a", label="Obstacle (cube)"),
        patches.Patch(facecolor="#6b6b66", label="Obstacle (cyl)"),
        patches.Patch(edgecolor="#e74c3c", facecolor="none",
                      linestyle=":", label="C-space inflation"),
        plt.Line2D([0], [0], color="#2980b9", linewidth=1.8, label="A* path"),
        patches.Patch(facecolor="#2ecc71", alpha=0.5, label=f"Go2 body (⌀{GO2_WIDTH}m)"),
    ]
    last_ax.legend(handles=legend_items, fontsize=7, loc="lower right", framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs("maps", exist_ok=True)
    out_path = "maps/cp6_5_obstacles_first_difficulty.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(out_path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
