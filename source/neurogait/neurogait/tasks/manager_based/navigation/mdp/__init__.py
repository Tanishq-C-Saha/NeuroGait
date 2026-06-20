"""Navigation MDP functions.

This package contains ONLY Python functions (the actual computation logic).
Config dataclasses (@configclass) live in navigation/managers/ instead.

Exports:
  - everything from isaaclab.envs.mdp (base obs, events, action funcs …)
  - navigation-owned locomotion functions: feet_air_time, terrain_levels_vel …
  - CP3 depth-camera functions: occupancy_grid_obs, occupancy_grid_obs_gpu
  - CP5 observation functions: goal_vector_obs, robot_velocity_obs
  - CP5 reward functions: reward_progress, reward_heading, penalty_collision, penalty_smoothness
  - CP5 action term: PreTrainedPolicyAction, PreTrainedPolicyActionCfg
  - CP5 event function: init_waypoints
"""

# base isaaclab mdp functions (observations, events, actions, commands …)
from isaaclab.envs.mdp import *  # noqa: F401, F403

# navigation-owned locomotion-style functions
from .curriculums import *   # noqa: F401, F403
from .rewards import *       # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

# CP3: depth-camera → occupancy grid
from .observations import occupancy_grid_obs, occupancy_grid_obs_gpu

# CP5: navigation policy observations
from .observations import goal_vector_obs, robot_velocity_obs

# CP5: hierarchical action term (frozen locomotion backbone)
from .pre_trained_policy_action import PreTrainedPolicyAction, PreTrainedPolicyActionCfg  # noqa: F401

# CP5: waypoint management event
from .waypoint_manager import init_waypoints  # noqa: F401
