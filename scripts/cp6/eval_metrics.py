"""CP6 — Quantitative evaluation over N episodes.

Runs a trained CP6 policy for `--num_episodes` episodes and reports:
  - success rate (goal reached within threshold)
  - mean path efficiency  (A* length / actual distance travelled)
  - mean collision count  per episode
  - mean steps to goal    (successful episodes only)
  - comparison table vs CP5 baseline (if --cp5_log provided)

Usage:
    ~/isaac-sim/kit/python/bin/python3 scripts/cp6/eval_metrics.py \
      --task NeuroGait-Navigation-CP6-Play-v0 \
      --checkpoint logs/skrl/neurogait_cp6/<run>/checkpoints/best_agent.pt \
      --num_episodes 50 \
      [--goal_x 8.0] [--goal_y 0.0] [--cp5_log path/to/cp5_metrics.json]

Saves results to: logs/eval/cp6_metrics_<timestamp>.json
"""

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CP6 N-episode evaluation")
parser.add_argument("--task",          type=str, default="NeuroGait-Navigation-CP6-Play-v0")
parser.add_argument("--num_envs",      type=int, default=1)
parser.add_argument("--checkpoint",    type=str, required=True)
parser.add_argument("--num_episodes",  type=int, default=50)
parser.add_argument("--max_steps",     type=int, default=2000,
                    help="Max steps per episode before counting as failure")
parser.add_argument("--goal_x",        type=float, default=8.0)
parser.add_argument("--goal_y",        type=float, default=0.0)
parser.add_argument("--goal_threshold",type=float, default=0.5,
                    help="Distance (m) to count as goal reached")
parser.add_argument("--cp5_log",       type=str, default=None,
                    help="Path to a cp5_metrics.json for comparison table")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from neurogait.tasks.manager_based.navigation.planning.global_grid import build_global_grid
from neurogait.tasks.manager_based.navigation.planning.planner import AStarPlanner
import neurogait.tasks  # noqa: F401
from neurogait.tasks.manager_based.navigation.models import NavigationPolicy

_LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "eval")


def _path_length(waypoints):
    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device  = env_cfg.sim.device
    nav_env = env.env.unwrapped

    policy = NavigationPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    print(f"[CP6-eval] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    policy.to(device)
    policy.eval()

    # ── Metric accumulators ────────────────────────────────────────────────────
    successes          = []
    path_efficiencies  = []
    collision_counts   = []
    steps_to_goal_list = []

    print(f"\n[CP6-eval] Running {args_cli.num_episodes} episodes ...\n")

    for ep in range(args_cli.num_episodes):
        obs, _ = env.reset()

        robot_pos_np = nav_env.scene["robot"].data.root_pos_w[0].cpu().numpy()
        start_xy = (float(robot_pos_np[0]), float(robot_pos_np[1]))
        goal_xy  = (start_xy[0] + args_cli.goal_x, start_xy[1] + args_cli.goal_y)

        # Build A* plan AFTER reset so the randomised obstacle layout is current
        grid, origin, _ = build_global_grid(nav_env)
        planner    = AStarPlanner(grid, origin, resolution=0.2)
        astar_path = planner.plan(start_xy, goal_xy) or []
        astar_len  = _path_length(astar_path) if len(astar_path) >= 2 else None

        trajectory      = [start_xy]
        n_collisions    = 0
        goal_reached    = False
        steps_to_goal   = None

        for step in range(args_cli.max_steps):
            if not simulation_app.is_running():
                break

            with torch.no_grad():
                result  = policy.act({"observations": obs}, role="policy")
                actions = result[0]

            obs, rewards, terminated, truncated, info = env.step(actions)

            pos = nav_env.scene["robot"].data.root_pos_w[0, :2].cpu().numpy()
            trajectory.append((float(pos[0]), float(pos[1])))

            # Collision: positive contact reward was negative → detect via env signal
            # Use terminated (base_contact) as collision proxy when not timeout
            if terminated.any() and not goal_reached:
                n_collisions += 1

            dist = math.sqrt((pos[0] - goal_xy[0]) ** 2 + (pos[1] - goal_xy[1]) ** 2)
            if not goal_reached and dist < args_cli.goal_threshold:
                goal_reached  = True
                steps_to_goal = step + 1

            if terminated.any() or truncated.any():
                break

        actual_len   = _path_length(trajectory)
        efficiency   = (astar_len / actual_len) if (astar_len and actual_len > 0.1) else 0.0

        successes.append(goal_reached)
        path_efficiencies.append(efficiency)
        collision_counts.append(n_collisions)
        if steps_to_goal is not None:
            steps_to_goal_list.append(steps_to_goal)

        print(
            f"  Ep {ep + 1:3d}/{args_cli.num_episodes} | "
            f"{'SUCCESS' if goal_reached else 'FAIL   '} | "
            f"eff={efficiency:.2f} | coll={n_collisions} | "
            f"steps={steps_to_goal if steps_to_goal else '-':>6}"
        )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n = len(successes)
    success_rate     = sum(successes) / n
    mean_efficiency  = sum(path_efficiencies) / n
    mean_collisions  = sum(collision_counts) / n
    mean_steps_goal  = (
        sum(steps_to_goal_list) / len(steps_to_goal_list)
        if steps_to_goal_list else float("nan")
    )

    results = {
        "checkpoint":       args_cli.checkpoint,
        "task":             args_cli.task,
        "num_episodes":     n,
        "goal_xy_local":    [args_cli.goal_x, args_cli.goal_y],
        "goal_threshold_m": args_cli.goal_threshold,
        "success_rate":     round(success_rate, 4),
        "mean_path_efficiency": round(mean_efficiency, 4),
        "mean_collisions_per_ep": round(mean_collisions, 4),
        "mean_steps_to_goal": round(mean_steps_goal, 1),
    }

    # ── Print summary ──────────────────────────────────────────────────────────
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  CP6 Evaluation  ({n} episodes)")
    print(sep)
    print(f"  Success rate          : {success_rate * 100:.1f}%")
    print(f"  Mean path efficiency  : {mean_efficiency:.3f}  (1.0 = optimal)")
    print(f"  Mean collisions / ep  : {mean_collisions:.2f}")
    print(f"  Mean steps to goal    : {mean_steps_goal:.0f}")

    if args_cli.cp5_log and os.path.isfile(args_cli.cp5_log):
        with open(args_cli.cp5_log) as f:
            cp5 = json.load(f)
        print(f"\n  Metric              CP5        CP6 (Δ)")
        print(f"  {sep}")

        def _row(label, key, fmt=".1%"):
            v5 = cp5.get(key, float("nan"))
            v6 = results[key]
            delta = v6 - v5 if not math.isnan(v5) else float("nan")
            sign  = "+" if delta >= 0 else ""
            print(f"  {label:<20}  {v5:{fmt}}    {v6:{fmt}}  ({sign}{delta:{fmt}})")

        _row("Success rate",       "success_rate")
        _row("Path efficiency",    "mean_path_efficiency")
        _row("Collisions/ep",      "mean_collisions_per_ep", fmt=".2f")
    print(sep)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    os.makedirs(_LOGS_DIR, exist_ok=True)
    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_LOGS_DIR, f"cp6_metrics_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[CP6-eval] Results saved → {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
