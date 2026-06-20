"""CP4: Rule-based A→B navigation through obstacle field.

Frozen locomotion checkpoint drives low-level locomotion.
A* planner + proportional heading controller drive velocity commands.

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/play_cp4.py \
      --task NeuroGait-Navigation-Unitree-Go2-Play-v1 \
      --num_envs 1 \
      --checkpoint logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt \
      --enable_cameras \
      --goal_x 8.0 --goal_y 0.0
"""

import argparse
import csv
import math
import sys
import os

import numpy as np

from isaaclab.app import AppLauncher

# cli_args lives in scripts/rsl_rl/ — add it to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

# ── argument parsing (must happen before AppLauncher) ────────────────────────

parser = argparse.ArgumentParser(description="CP4: A→B navigation demo")
parser.add_argument("--task",       type=str, required=True,  help="Gym task id")
parser.add_argument("--num_envs",   type=int, default=1,      help="Number of envs")
parser.add_argument("--goal_x",     type=float, default=8.0,  help="Goal X (world m)")
parser.add_argument("--goal_y",     type=float, default=0.0,  help="Goal Y (world m)")
parser.add_argument("--agent",      type=str,
                    default="rsl_rl_cfg_entry_point",
                    help="Agent config entry-point key")
parser.add_argument("--max_steps",  type=int, default=2000,   help="Step budget")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# launch simulator (MUST be before any omni/isaac imports)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that need the sim running ────────────────────────────────────────

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner, DistillationRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401

from neurogait.tasks.manager_based.navigation.planning.global_grid import build_global_grid
from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner
from neurogait.tasks.manager_based.navigation.control.waypoint_controller import WaypointController

# maps/ directory at project root (one level above scripts/)
_MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")


# ── helpers ───────────────────────────────────────────────────────────────────

def quat_to_yaw(quat_wxyz):
    """Extract yaw from Isaac Lab quaternion (w, x, y, z)."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def save_map(grid, origin, resolution, path_world, start_xy, goal_xy, obstacle_info=None):
    """Save occupancy-grid + A* path as a clean B&W PNG plus supporting files."""
    os.makedirs(_MAPS_DIR, exist_ok=True)

    # ── raw grid ──────────────────────────────────────────────────────────────
    npy_path = os.path.join(_MAPS_DIR, "global_grid.npy")
    np.save(npy_path, grid)
    print(f"[CP4] Grid array saved  → {npy_path}")

    # ── path CSV ──────────────────────────────────────────────────────────────
    if path_world:
        csv_path = os.path.join(_MAPS_DIR, "path_waypoints.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(path_world)
        print(f"[CP4] Path CSV saved    → {csv_path}")

    # ── PNG ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # tight view window around all relevant objects
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

        # --- occupancy grid: white = free, black = occupied (standard robotics convention) ---
        rows, cols = grid.shape
        extent = [
            origin[0], origin[0] + cols * resolution,
            origin[1], origin[1] + rows * resolution,
        ]
        ax.imshow(grid, origin="lower", cmap="gray_r", extent=extent,
                  vmin=0, vmax=1, zorder=1)

        # --- A* path (blue line) ---
        if path_world and len(path_world) >= 2:
            xs = [p[0] for p in path_world]
            ys = [p[1] for p in path_world]
            ax.plot(xs, ys, color="#1565C0", linewidth=2.0, zorder=3,
                    solid_capstyle="round", label="A* path")

        # --- start: filled green circle ---
        ax.scatter(*start_xy, s=180, color="#00C853", zorder=5,
                   edgecolors="black", linewidths=1.2, label="Start")

        # --- goal: filled red circle ---
        ax.scatter(*goal_xy, s=180, color="#D32F2F", zorder=5,
                   edgecolors="black", linewidths=1.2, label="Goal")

        # --- axes and labels ---
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

        # --- 1 m scale bar ---
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


def spawn_marker(prim_path, xyz, color_rgb):
    """Spawn a tall coloured visual-only pillar at world xyz.  No physics.

    Uses a flat prim path directly under /World/ so the parent always exists.
    """
    cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 1.8),   # tall thin pillar — visible above obstacles
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color_rgb,
            opacity=1.0,
        ),
        # no rigid_props / collision_props → purely visual
    )
    cfg.func(prim_path, cfg, translation=xyz)


# ── main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """CP4 end-to-end play loop."""

    # ── 1. configure env ─────────────────────────────────────────────────────
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── 2. create env ────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 3. load frozen locomotion checkpoint ─────────────────────────────────
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[CP4] Loading checkpoint: {resume_path}")

    if agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── 4. reset env ─────────────────────────────────────────────────────────
    obs = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    robot_pos_np = robot.data.root_pos_w[0].cpu().numpy()
    start_xy = (float(robot_pos_np[0]), float(robot_pos_np[1]))
    print(f"[CP4] Robot start position: ({start_xy[0]:.2f}, {start_xy[1]:.2f})")

    # ── 5. build global grid ─────────────────────────────────────────────────
    grid, origin, obstacle_info = build_global_grid(env.unwrapped)

    # ── 6. plan path ─────────────────────────────────────────────────────────
    goal_xy = (args_cli.goal_x, args_cli.goal_y)
    planner = AStarPlanner(grid, origin, resolution=0.2)
    waypoints = planner.plan(start_xy, goal_xy)

    if not waypoints:
        print("[CP4] No path found — check obstacle layout or goal position. Exiting.")
        env.close()
        return

    # ── 7. save map + path to maps/ ──────────────────────────────────────────
    save_map(grid, origin, 0.2, waypoints, start_xy, goal_xy, obstacle_info)

    # ── 8. spawn visual markers in the sim viewport ───────────────────────────
    # Flat paths directly under /World/ — parent always exists.
    # /World/Markers/start would fail because /World/Markers doesn't exist yet.
    spawn_marker(
        "/World/marker_start",
        xyz=(start_xy[0], start_xy[1], 0.9),   # z=0.9 → pillar base at ground, top at 2.7m
        color_rgb=(0.05, 0.95, 0.1),            # bright green
    )
    spawn_marker(
        "/World/marker_goal",
        xyz=(goal_xy[0], goal_xy[1], 0.9),
        color_rgb=(0.95, 0.05, 0.1),            # bright red
    )
    print(f"[CP4] Markers spawned — green={start_xy}, red={goal_xy}")

    # ── 9. create controller ─────────────────────────────────────────────────
    controller = WaypointController(planner)

    # ── 10. get reference to velocity command term for injection ──────────────
    vel_term = env.unwrapped.command_manager.get_term("base_velocity")

    # ── 11. main loop ─────────────────────────────────────────────────────────
    MAX_STEPS = args_cli.max_steps
    goal_reached_count = 0

    for step in range(MAX_STEPS):
        if not simulation_app.is_running():
            break

        # robot state
        robot_pos_np = robot.data.root_pos_w[0].cpu().numpy()
        robot_quat_np = robot.data.root_quat_w[0].cpu().numpy()   # (w, x, y, z)
        robot_xy = (float(robot_pos_np[0]), float(robot_pos_np[1]))
        robot_yaw = quat_to_yaw(robot_quat_np)

        # velocity command from waypoint controller
        vx, vy, yaw_rate = controller.step(robot_xy, robot_yaw)

        # goal check
        if vx == 0.0 and vy == 0.0 and yaw_rate == 0.0:
            goal_reached_count += 1
            if goal_reached_count >= 3:
                print(f"[CP4] Goal reached in {step} steps!")
                break
        else:
            goal_reached_count = 0

        # inject velocity command before obs compute
        vel_term.vel_command_b[0, 0] = vx
        vel_term.vel_command_b[0, 1] = vy
        vel_term.vel_command_b[0, 2] = yaw_rate

        # compute observations with overridden command, forward through policy
        with torch.no_grad():
            obs = env.get_observations()
            actions = policy(obs)

        # step env (randomises command internally; we override each iteration)
        obs, _, dones, _ = env.step(actions)

        # status print every 50 steps
        if step % 50 == 0:
            wp_idx = controller.current_waypoint_idx
            total = len(planner.path_world)
            target = planner.get_waypoint_world(wp_idx)
            if target is not None:
                dist = math.sqrt(
                    (robot_xy[0] - target[0]) ** 2 + (robot_xy[1] - target[1]) ** 2
                )
                print(f"Step {step:4d} | waypoint {wp_idx:2d}/{total} | dist {dist:.2f}m "
                      f"| vx={vx:.2f} yaw={yaw_rate:.2f}")
            else:
                print(f"Step {step:4d} | goal reached")
    else:
        print(f"[CP4] Max steps ({MAX_STEPS}) exceeded — did not reach goal")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
