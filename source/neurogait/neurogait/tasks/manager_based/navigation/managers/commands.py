"""Command configurations for the NeuroGait navigation task."""

import math
from dataclasses import dataclass

from isaaclab.envs import mdp as env_mdp
from isaaclab.utils import configclass


@configclass
class CommandsCfg:
    """Velocity command sampled uniformly from a range."""

    base_velocity = env_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=env_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@dataclass
class NullCommandCfg:
    """
    Stub for future CPs that will drive commands from the A* planner.

    CP3 uses the velocity command above; this exists as a hook for later.
    """
    name: str = "null_command"
