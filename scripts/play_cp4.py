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
import math
import sys
import os

from isaaclab.app import AppLauncher

# cli_args lives in scripts/rsl_rl/ — add it to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

# ── argument parsing (must happen before AppLauncher) ────────────────────────

parser = argparse.ArgumentParser(description="CP4: A→B navigation demo")
parser.add_argument("--task",       type=str, required=True,  help="Gym task id")
parser.add_argument("--num_envs",   type=int, default=1,      help="Number of envs")
parser.add_argument("--checkpoint", type=str, required=True,  help="Path to .pt checkpoint")
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

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401

from neurogait.tasks.manager_based.navigation.planning.global_grid import build_global_grid
from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner
from neurogait.tasks.manager_based.navigation.control.waypoint_controller import WaypointController


# ── helpers ───────────────────────────────────────────────────────────────────

def quat_to_yaw(quat_wxyz):
    """Extract yaw from Isaac Lab quaternion (w, x, y, z)."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ── main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """CP4 end-to-end play loop."""

    # ── 1. configure env ─────────────────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
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
    grid, origin = build_global_grid(env.unwrapped)

    # ── 6. plan path ─────────────────────────────────────────────────────────
    goal_xy = (args_cli.goal_x, args_cli.goal_y)
    planner = AStarPlanner(grid, origin, resolution=0.2)
    waypoints = planner.plan(start_xy, goal_xy)

    if not waypoints:
        print("[CP4] No path found — check obstacle layout or goal position. Exiting.")
        env.close()
        return

    # ── 7. create controller ─────────────────────────────────────────────────
    controller = WaypointController(planner)

    # ── 8. get reference to velocity command term for injection ───────────────
    vel_term = env.unwrapped.command_manager.get_term("base_velocity")

    # ── 9. main loop ─────────────────────────────────────────────────────────
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

        # inject velocity command into the command term before obs compute
        vel_term.vel_command_b[0, 0] = vx
        vel_term.vel_command_b[0, 1] = vy
        vel_term.vel_command_b[0, 2] = yaw_rate

        # compute observations with overridden command, forward through policy
        with torch.no_grad():
            obs = env.get_observations()
            actions = policy(obs)

        # step env (will randomise command internally, but we override each iter)
        obs, _, dones, _ = env.step(actions)

        # status print
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
