"""MDP components for the NeuroGait navigation task family.

Exports the full set of functions needed by navigation_base_env_cfg.py:
  - everything from isaaclab.envs.mdp (observations, events, base rewards …)
  - locomotion-style rewards, curriculums, terminations (owned by navigation)
  - CP3: occupancy grid observation terms
  - CP3: NullCommandCfg stub
"""

# base isaaclab mdp (observations, actions, events, commands …)
from isaaclab.envs.mdp import *  # noqa: F401, F403

# navigation-owned locomotion-style functions
from .curriculums import *   # noqa: F401, F403
from .rewards import *       # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

# CP3: depth-camera → occupancy grid observation terms
from .observations import occupancy_grid_obs, occupancy_grid_obs_gpu

# CP3: command config stub (future CPs will replace with planner-backed term)
from .commands import NullCommandCfg
