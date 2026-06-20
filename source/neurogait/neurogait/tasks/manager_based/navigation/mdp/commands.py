"""
Command configurations for the NeuroGait navigation task.

CP3 still uses the fixed-velocity command from the locomotion base class
(set in navigation_env_cfg.py).  This file exists so that the mdp package
is complete and importable.  NullCommandCfg is a stub for future CPs that
will need goal-directed command generation from the A* planner.
"""

from dataclasses import dataclass, field


@dataclass
class NullCommandCfg:
    """
    Stub command config for CP3.

    CP3 velocity commands come directly from the base class
    (LocomotionVelocityRoughEnvCfg) — no navigation-level command manager
    is needed yet.  Future CPs will replace this with a planner-backed term
    that reads the A* path and emits velocity waypoints.
    """
    name: str = "null_command"
