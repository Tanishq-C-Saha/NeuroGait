"""Check reward term balance for ANY registered NeuroGait task.

Runs --steps steps with random actions (--num_envs envs, headless) and prints
a table of weighted reward magnitudes.  All terms should be within 10× of each
other.  If any term dominates or reads zero, adjust its weight before training.

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/test_reward_scales.py \
      --task NeuroGait-Navigation-CP5-v0 --num_envs 256 --headless

    ~/isaac-sim/kit/python/bin/python3 scripts/test_reward_scales.py \
      --task NeuroGait-Navigation-CP6-v0 --num_envs 64 --headless
"""

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reward scale sanity check")
parser.add_argument("--task",     type=str, required=True,
                    help="Registered gym task id, e.g. NeuroGait-Navigation-CP6-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps",    type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from gymnasium.envs.registration import registry as gym_registry

import neurogait.tasks  # noqa: F401  — registers all NeuroGait envs


def _load_env_cfg(task_id: str):
    """Instantiate the env config registered for task_id."""
    spec = gym_registry.get(task_id)
    if spec is None:
        registered = [k for k in gym_registry if k.startswith("NeuroGait")]
        raise ValueError(
            f"Task '{task_id}' not found.\nRegistered NeuroGait tasks:\n"
            + "\n".join(f"  {k}" for k in sorted(registered))
        )
    entry = spec.kwargs.get("env_cfg_entry_point", "")
    if not entry or ":" not in entry:
        raise ValueError(f"No env_cfg_entry_point registered for '{task_id}'")
    module_path, class_name = entry.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def main():
    device = args_cli.device or "cuda:0"

    env_cfg = _load_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    action_dim = env.action_space.shape[-1]
    rm         = env.unwrapped.reward_manager
    term_names = rm._term_names
    accum      = {name: 0.0 for name in term_names}

    print(f"\n[reward-scales] Task: {args_cli.task}")
    print(f"[reward-scales] {args_cli.num_envs} envs × {args_cli.steps} steps  "
          f"| action_dim={action_dim}\n")

    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break

        actions = torch.rand(args_cli.num_envs, action_dim, device=device) * 2 - 1
        env.step(actions)

        for i, name in enumerate(term_names):
            accum[name] += float(rm._step_reward[:, i].mean())

        if step % 20 == 0:
            pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2].cpu()
            print(f"  step {step:3d}  pos=({pos[0]:.2f}, {pos[1]:.2f})")

    # ── summary table ──────────────────────────────────────────────────────────
    means      = {n: accum[n] / args_cli.steps for n in term_names}
    abs_means  = [abs(v) for v in means.values() if v != 0.0]
    ratio      = max(abs_means) / min(abs_means) if len(abs_means) >= 2 else float("nan")

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Reward Scale Check — {args_cli.task}")
    print(sep)
    print(f"  {'Term':<35} {'Mean/step':>10}  {'Total':>10}")
    print("-" * 62)
    grand_total = 0.0
    for name in term_names:
        mean_val  = means[name]
        total_val = accum[name]
        flag = "  ← ZERO" if mean_val == 0.0 else ""
        print(f"  {name:<35} {mean_val:>10.4f}  {total_val:>10.4f}{flag}")
        grand_total += mean_val
    print("-" * 62)
    print(f"  {'TOTAL':<35} {grand_total:>10.4f}")
    print(sep)
    print(f"\n  Max/min ratio of non-zero terms: {ratio:.1f}×")
    print("  Target: all terms within 10× of each other.")
    if ratio > 10:
        print("  WARNING: ratio > 10 — review weights before training.")
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
