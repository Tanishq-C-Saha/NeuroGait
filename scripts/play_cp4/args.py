"""Argument parser for the CP4 play script.

No simulator imports here — this module is safe to import before AppLauncher.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CP4: A→B navigation demo")
    parser.add_argument("--task",      type=str,   required=True, help="Gym task id")
    parser.add_argument("--num_envs",  type=int,   default=1,     help="Number of envs")
    parser.add_argument("--goal_x",    type=float, default=8.0,   help="Goal X (world m)")
    parser.add_argument("--goal_y",    type=float, default=0.0,   help="Goal Y (world m)")
    parser.add_argument("--agent",     type=str,
                        default="rsl_rl_cfg_entry_point",
                        help="Agent config entry-point key")
    parser.add_argument("--max_steps", type=int,   default=2000,  help="Step budget")
    return parser
