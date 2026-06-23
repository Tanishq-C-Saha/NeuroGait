"""Navigation MDP functions.

This package contains ONLY Python functions (the actual computation logic).
Config dataclasses (@configclass) live in navigation/managers/ instead.

Exports:
  - everything from isaaclab.envs.mdp (base obs, events, action funcs …)
  - navigation-owned locomotion functions: feet_air_time, terrain_levels_vel …
  - CP3 depth-camera functions: occupancy_grid_obs, occupancy_grid_obs_gpu
"""

# base isaaclab mdp functions (observations, events, actions, commands …)
from isaaclab.envs.mdp import *  # noqa: F401, F403

# navigation-owned locomotion-style functions
from .curriculums import *   # noqa: F401, F403
from .rewards import *       # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

# CP3: depth-camera → occupancy grid (symmetric)
from .observations import occupancy_grid_obs, occupancy_grid_obs_gpu

# CP5: asymmetric grid + scalar obs terms
from .observations import (
    occupancy_grid_obs_cp5,
    future_waypoints_obs,
    robot_velocity_obs,
    quat_to_yaw_batch,
    _cp5_init_waypoint_state,
    _cp5_reset_waypoint_state,
)

# CP5: navigation reward terms
from .rewards import (
    cp5_reward_velocity_toward_goal,
    cp5_reward_goal_proximity,
    cp5_reward_goal_reached,
    cp5_penalty_collision_velocity_scaled,
    cp5_penalty_stuck,
    cp5_reward_heading,
    cp5_penalty_smoothness,
)

# navigation mdp funcs for PreTrainedPolicyAction
from isaaclab_tasks.manager_based.navigation.mdp import (
    PreTrainedPolicyAction,
    PreTrainedPolicyActionCfg,
)
