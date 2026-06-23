"""CP4 play loop — all simulator-dependent logic.

Import only after AppLauncher has started.
"""

import math

import torch
import gymnasium as gym
import importlib.metadata as metadata
from rsl_rl.runners import OnPolicyRunner, DistillationRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import neurogait.tasks  # noqa: F401

from neurogait.tasks.manager_based.navigation.planning.global_grid import build_global_grid
from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner
from neurogait.tasks.manager_based.navigation.control.waypoint_controller import WaypointController

from map_io import save_map
from markers import spawn_marker


def quat_to_yaw(quat_wxyz) -> float:
    """Extract yaw from Isaac Lab quaternion (w, x, y, z)."""
    w, x, y, z = (float(quat_wxyz[i]) for i in range(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def make_main(args_cli, simulation_app):
    """Return a hydra-decorated main() with args_cli bound via closure."""

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        # ── 1. configure env ─────────────────────────────────────────────────
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = _update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # ── 2. create env ────────────────────────────────────────────────────
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # ── 3. load frozen locomotion checkpoint ─────────────────────────────
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[CP4] Loading checkpoint: {resume_path}")

        if agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # ── 4. reset env + record start position ─────────────────────────────
        obs = env.get_observations()
        robot = env.unwrapped.scene["robot"]
        robot_pos_np = robot.data.root_pos_w[0].cpu().numpy()
        start_xy = (float(robot_pos_np[0]), float(robot_pos_np[1]))
        print(f"[CP4] Robot start position: ({start_xy[0]:.2f}, {start_xy[1]:.2f})")

        # ── 5. build global grid ─────────────────────────────────────────────
        grid, origin, obstacle_info = build_global_grid(env.unwrapped)

        # ── 6. plan path ─────────────────────────────────────────────────────
        goal_xy = (args_cli.goal_x, args_cli.goal_y)
        planner = AStarPlanner(grid, origin, resolution=0.2)
        waypoints = planner.plan(start_xy, goal_xy)

        if not waypoints:
            print("[CP4] No path found — check obstacle layout or goal position. Exiting.")
            env.close()
            return

        # ── 7. save map + path to maps/ ──────────────────────────────────────
        save_map(grid, origin, 0.2, waypoints, start_xy, goal_xy, obstacle_info)

        # ── 8. spawn visual markers ───────────────────────────────────────────
        spawn_marker("/World/marker_start",
                     xyz=(start_xy[0], start_xy[1], 0.9),
                     color_rgb=(0.05, 0.95, 0.1))
        spawn_marker("/World/marker_goal",
                     xyz=(goal_xy[0], goal_xy[1], 0.9),
                     color_rgb=(0.95, 0.05, 0.1))
        print(f"[CP4] Markers spawned — green={start_xy}, red={goal_xy}")

        # ── 9. create controller + grab velocity command term ─────────────────
        controller = WaypointController(planner)
        vel_term = env.unwrapped.command_manager.get_term("base_velocity")

        # ── 10. main loop ─────────────────────────────────────────────────────
        goal_reached_count = 0

        for step in range(args_cli.max_steps):
            if not simulation_app.is_running():
                break

            robot_pos_np  = robot.data.root_pos_w[0].cpu().numpy()
            robot_quat_np = robot.data.root_quat_w[0].cpu().numpy()
            robot_xy  = (float(robot_pos_np[0]), float(robot_pos_np[1]))
            robot_yaw = quat_to_yaw(robot_quat_np)

            vx, vy, yaw_rate = controller.step(robot_xy, robot_yaw)

            if vx == 0.0 and vy == 0.0 and yaw_rate == 0.0:
                goal_reached_count += 1
                if goal_reached_count >= 3:
                    print(f"[CP4] Goal reached in {step} steps!")
                    break
            else:
                goal_reached_count = 0

            vel_term.vel_command_b[0, 0] = vx
            vel_term.vel_command_b[0, 1] = vy
            vel_term.vel_command_b[0, 2] = yaw_rate

            with torch.no_grad():
                obs = env.get_observations()
                actions = policy(obs)

            obs, _, dones, _ = env.step(actions)

            if step % 50 == 0:
                _print_status(step, controller, planner, robot_xy, vx, yaw_rate)
        else:
            print(f"[CP4] Max steps ({args_cli.max_steps}) exceeded — did not reach goal")

        env.close()

    return main


# ── internal helpers ──────────────────────────────────────────────────────────

def _update_rsl_rl_cfg(agent_cfg, args_cli):
    """Thin wrapper kept here to avoid importing cli_args inside loop."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
    import cli_args
    return cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)


def _print_status(step, controller, planner, robot_xy, vx, yaw_rate):
    wp_idx = controller.current_waypoint_idx
    total  = len(planner.path_world)
    target = planner.get_waypoint_world(wp_idx)
    if target is not None:
        dist = math.sqrt((robot_xy[0] - target[0])**2 + (robot_xy[1] - target[1])**2)
        print(f"Step {step:4d} | waypoint {wp_idx:2d}/{total} | dist {dist:.2f}m "
              f"| vx={vx:.2f} yaw={yaw_rate:.2f}")
    else:
        print(f"Step {step:4d} | goal reached")
