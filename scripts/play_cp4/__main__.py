"""CP4: Rule-based A→B navigation through obstacle field.

Frozen locomotion checkpoint drives low-level locomotion.
A* planner + proportional heading controller drive velocity commands.

Run:
    ~/isaac-sim/kit/python/bin/python3 scripts/play_cp4 \
      --task NeuroGait-Navigation-Unitree-Go2-Play-v1 \
      --num_envs 1 \
      --checkpoint logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt \
      --enable_cameras \
      --goal_x 8.0 --goal_y 0.0

Module layout:
    args.py     — argument parser (no sim deps, safe before AppLauncher)
    map_io.py   — save occupancy grid + A* path to maps/
    markers.py  — spawn visual pillars in the sim viewport
    loop.py     — quat_to_yaw + main play loop
"""

import sys
import os

# Ensure sibling modules (args, loop, …) are importable when this file is run
# directly as `python3 scripts/play_cp4` (Python sets __package__ = None there).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# cli_args lives in scripts/rsl_rl/ — add it to path before arg parsing
sys.path.insert(0, os.path.join(_HERE, "..", "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

from isaaclab.app import AppLauncher
from args import build_parser

# ── argument parsing (must happen before AppLauncher) ────────────────────────

parser = build_parser()
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# ── launch simulator (must be before any omni/isaac imports) ─────────────────

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── now safe to import sim-dependent modules ──────────────────────────────────

from loop import make_main  # noqa: E402

if __name__ == "__main__":
    main = make_main(args_cli, simulation_app)
    main()
    simulation_app.close()
