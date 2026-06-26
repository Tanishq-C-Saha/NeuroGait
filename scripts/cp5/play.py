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


def _save_trajectory_plot(grid, origin, resolution, astar_path, trajectory, start_xy, goal_xy):
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

        # Occupancy grid background
        ax.imshow(grid, origin="lower", cmap="gray_r", extent=extent, vmin=0, vmax=1)

        # A* planned path (blue)
        if astar_path:
            ax.plot([p[0] for p in astar_path], [p[1] for p in astar_path],
                    color="#1565C0", linewidth=2.0, label="A* planned path",
                    solid_capstyle="round")

        # RL actual trajectory (red)
        if trajectory:
            ax.plot([p[0] for p in trajectory], [p[1] for p in trajectory],
                    color="#D32F2F", linewidth=1.5, label="RL actual trajectory",
                    solid_capstyle="round", alpha=0.8)

        # Start / goal markers
        ax.scatter(*start_xy, s=180, color="#00C853", edgecolors="black",
                   linewidths=1.2, zorder=5, label="Start")
        ax.scatter(*goal_xy,  s=180, color="#D32F2F", edgecolors="black",
                   linewidths=1.2, zorder=5, label="Goal")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.set_title("NeuroGait CP5 — A* Plan vs RL Trajectory")
        ax.grid(True, color="#cccccc", linewidth=0.4, linestyle="--")

        plt.tight_layout()
        os.makedirs(_MAPS_DIR, exist_ok=True)
        save_path = os.path.join(_MAPS_DIR, "cp5_trajectory.png")
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[CP5-play] Trajectory map saved → {save_path}")
    except ImportError:
        print("[CP5-play] matplotlib not available — skipping trajectory plot")


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
    trajectory = []

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

        if step % 10 == 0:
            lin_vel = robot.data.root_lin_vel_b[0].cpu()
            cmd     = actions[0].cpu()
            print(
                f"Step {step:4d} | "
                f"CMD vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} hdg={cmd[2]:+.2f} | "
                f"vel vx={lin_vel[0]:+.2f} vy={lin_vel[1]:+.2f} | "
                f"POS ({pos[0]:.1f}, {pos[1]:.1f})"
            )

        if terminated.any() or truncated.any():
            print(f"[CP5-play] Episode ended at step {step}")
            obs, _ = env.reset()

    # ── Save comparison plot ──────────────────────────────────────────────────
    _save_trajectory_plot(grid, origin, 0.2, astar_path, trajectory, start_xy, goal_xy)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
