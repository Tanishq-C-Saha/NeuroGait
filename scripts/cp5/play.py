"""CP5 — Evaluate a trained navigation policy.

Loads policy weights directly (no Runner), spawns start/goal markers,
records the robot trajectory, and saves a comparison plot:
  maps/cp5_trajectory.png  — occupancy grid + A* plan (blue) + RL path (red)

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/cp5/play.py \
      --task NeuroGait-Navigation-CP5-Play-v0 \
      --checkpoint logs/skrl/neurogait_cp5_navigation/<run>/checkpoints/best_agent.pt

Output every 10 steps:
    Step   10 | CMD vx=+0.50 vy=+0.00 hdg=+0.78 | vel vx=+0.48 vy=+0.01 | POS (2.3, 0.5)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CP5 navigation evaluation")
parser.add_argument("--task",       type=str, default="NeuroGait-Navigation-CP5-Play-v0")
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to skrl checkpoint (.pt file in checkpoints/)")
parser.add_argument("--max_steps",  type=int, default=2000)
parser.add_argument("--goal_x",     type=float, default=8.0)
parser.add_argument("--goal_y",     type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from neurogait.tasks.manager_based.navigation.planning.global_grid import build_global_grid
from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner
import neurogait.tasks  # noqa: F401
from neurogait.tasks.manager_based.navigation.models import NavigationPolicy

_MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "maps")


def _spawn_marker(prim_path: str, xyz: tuple, color_rgb: tuple) -> None:
    """Tall coloured pillar — visual only, no physics."""
    cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 1.8),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_rgb, opacity=1.0),
    )
    cfg.func(prim_path, cfg, translation=xyz)


def _save_trajectory_csv(trajectory, save_path):
    """Write raw (x, y) points to a CSV — fast and survives matplotlib failures."""
    try:
        with open(save_path, "w") as f:
            f.write("x,y\n")
            for x, y in trajectory:
                f.write(f"{x:.4f},{y:.4f}\n")
        print(f"[CP5-play] Trajectory CSV saved → {save_path}")
    except Exception as e:
        print(f"[CP5-play] WARNING: CSV save failed: {e}")


def _save_trajectory_plot(grid, origin, resolution, astar_path, trajectory,
                          start_xy, goal_xy, reached, save_path):
    print(f"[CP5-play] Saving trajectory map ({len(trajectory)} points) → {save_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows, cols = grid.shape
        extent = [
            origin[0], origin[0] + cols * resolution,
            origin[1], origin[1] + rows * resolution,
        ]

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.imshow(grid, origin="lower", cmap="gray_r", extent=extent, vmin=0, vmax=1)

        if astar_path and len(astar_path) >= 2:
            ax.plot([p[0] for p in astar_path], [p[1] for p in astar_path],
                    color="#1565C0", linewidth=2.0, label="A* planned path",
                    solid_capstyle="round")

        if trajectory and len(trajectory) >= 2:
            ax.plot([p[0] for p in trajectory], [p[1] for p in trajectory],
                    color="#D32F2F", linewidth=1.5, label="RL actual trajectory",
                    solid_capstyle="round", alpha=0.8)

        ax.scatter(*start_xy, s=180, color="#00C853", edgecolors="black",
                   linewidths=1.2, zorder=5, label="Start")
        ax.scatter(*goal_xy,  s=180, color="#FF1744", edgecolors="black",
                   linewidths=1.2, zorder=5, label="Goal")

        if trajectory:
            final = trajectory[-1]
            ax.scatter(*final, s=120, color="#FF9100", marker="D",
                       edgecolors="black", linewidths=1.2, zorder=5,
                       label=f"Final pos ({final[0]:.1f}, {final[1]:.1f})")

        n_steps = len(trajectory)
        dist_covered = sum(
            ((trajectory[i + 1][0] - trajectory[i][0]) ** 2
             + (trajectory[i + 1][1] - trajectory[i][1]) ** 2) ** 0.5
            for i in range(n_steps - 1)
        ) if n_steps > 1 else 0.0
        status = "REACHED" if reached else "NOT REACHED"
        ax.set_title(
            f"NeuroGait CP5 — A* Plan vs RL Trajectory\n"
            f"Goal: {status} | Steps: {n_steps} | Distance covered: {dist_covered:.1f} m",
            fontsize=10,
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")

        # Zoom to the relevant area instead of showing the full 40 m grid
        pad = 2.0
        all_x = [p[0] for p in trajectory] + [start_xy[0], goal_xy[0]]
        all_y = [p[1] for p in trajectory] + [start_xy[1], goal_xy[1]]
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, color="#cccccc", linewidth=0.4, linestyle="--")

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[CP5-play] Trajectory map saved → {save_path}")

    except Exception as e:
        print(f"[CP5-play] WARNING: plot save failed ({type(e).__name__}: {e})")


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device  = env_cfg.sim.device
    nav_env = env.env.unwrapped

    # ── Load policy ──────────────────────────────────────────────────────────
    policy = NavigationPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    print(f"[CP5-play] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    print(f"[CP5-play] Checkpoint keys: {list(checkpoint.keys())}")
    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    policy.to(device)
    policy.eval()

    # ── Reset + record start position ────────────────────────────────────────
    obs, _ = env.reset()
    robot_pos_np = nav_env.scene["robot"].data.root_pos_w[0].cpu().numpy()
    start_xy = (float(robot_pos_np[0]), float(robot_pos_np[1]))
    goal_xy  = (start_xy[0] + args_cli.goal_x, start_xy[1] + args_cli.goal_y)
    print(f"[CP5-play] Start: {start_xy}  Goal: {goal_xy}")

    # ── Build global occupancy grid + A* path ────────────────────────────────
    grid, origin, obstacle_info = build_global_grid(nav_env)
    planner   = AStarPlanner(grid, origin, resolution=0.2)
    astar_path = planner.plan(start_xy, goal_xy)
    if not astar_path:
        print("[CP5-play] Warning: A* found no path — goal may be blocked")

    # ── Spawn viewport markers ────────────────────────────────────────────────
    _spawn_marker("/World/marker_start",
                  xyz=(start_xy[0], start_xy[1], 0.9), color_rgb=(0.05, 0.95, 0.1))
    _spawn_marker("/World/marker_goal",
                  xyz=(goal_xy[0],  goal_xy[1],  0.9), color_rgb=(0.95, 0.05, 0.1))
    print(f"[CP5-play] Markers spawned — green={start_xy}  red={goal_xy}")

    # ── Inference loop ────────────────────────────────────────────────────────
    import math
    trajectory   = []
    goal_reached = False
    _save_path   = os.path.join(_MAPS_DIR, "cp5_trajectory.png")
    os.makedirs(_MAPS_DIR, exist_ok=True)

    # Periodic save every N steps so the file exists on disk even if Isaac Sim
    # hard-exits via os._exit() (which bypasses Python finally blocks entirely).
    _SAVE_INTERVAL = 50

    try:
        for step in range(args_cli.max_steps):
            if not simulation_app.is_running():
                break

            with torch.no_grad():
                result  = policy.act({"observations": obs}, role="policy")
                actions = result[0]

            obs, rewards, terminated, truncated, _ = env.step(actions)

            robot = nav_env.scene["robot"]
            pos   = robot.data.root_pos_w[0, :2].cpu().numpy()
            trajectory.append((float(pos[0]), float(pos[1])))

            dist_to_goal = math.sqrt((pos[0] - goal_xy[0]) ** 2 + (pos[1] - goal_xy[1]) ** 2)
            if dist_to_goal < 0.5:
                print(f"[CP5-play] Goal REACHED at step {step} (dist={dist_to_goal:.2f} m)")
                goal_reached = True

            if step % 10 == 0:
                lin_vel = robot.data.root_lin_vel_b[0].cpu()
                cmd     = actions[0].cpu()
                print(
                    f"Step {step:4d} | "
                    f"CMD vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} hdg={cmd[2]:+.2f} | "
                    f"vel vx={lin_vel[0]:+.2f} vy={lin_vel[1]:+.2f} | "
                    f"POS ({pos[0]:.1f}, {pos[1]:.1f}) | dist_goal={dist_to_goal:.1f} m"
                )

            # Periodic map flush — survives os._exit() on window close
            if step > 0 and step % _SAVE_INTERVAL == 0:
                _save_trajectory_plot(
                    grid=grid, origin=origin, resolution=0.2,
                    astar_path=astar_path, trajectory=trajectory,
                    start_xy=start_xy, goal_xy=goal_xy,
                    reached=goal_reached, save_path=_save_path,
                )

            if terminated.any() or truncated.any():
                print(f"[CP5-play] Episode ended at step {step}")
                obs, _ = env.reset()

    finally:
        # Final save — runs on normal finish, KeyboardInterrupt, or exception.
        # CSV is written first (fast) so data is safe even if matplotlib fails.
        if trajectory:
            csv_path = _save_path.replace(".png", ".csv")
            _save_trajectory_csv(trajectory, csv_path)
            _save_trajectory_plot(
                grid=grid, origin=origin, resolution=0.2,
                astar_path=astar_path, trajectory=trajectory,
                start_xy=start_xy, goal_xy=goal_xy,
                reached=goal_reached, save_path=_save_path,
            )
        else:
            print("[CP5-play] No trajectory recorded — map not saved")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
