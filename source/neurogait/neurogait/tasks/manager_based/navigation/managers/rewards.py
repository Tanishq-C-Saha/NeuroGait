"""Reward configuration for the NeuroGait navigation task."""

import math

from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# navigation-owned reward functions (feet_air_time, etc.)
from neurogait.tasks.manager_based.navigation import mdp as nav_mdp


@configclass
class RewardsCfg:
    """Reward terms for velocity-tracking locomotion."""

    # ── task rewards ──────────────────────────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=env_mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=env_mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ── penalties ─────────────────────────────────────────────────────────────
    lin_vel_z_l2   = RewTerm(func=env_mdp.lin_vel_z_l2,      weight=-2.0)
    ang_vel_xy_l2  = RewTerm(func=env_mdp.ang_vel_xy_l2,     weight=-0.05)
    dof_torques_l2 = RewTerm(func=env_mdp.joint_torques_l2,  weight=-1.0e-5)
    dof_acc_l2     = RewTerm(func=env_mdp.joint_acc_l2,      weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=env_mdp.action_rate_l2,    weight=-0.01)

    feet_air_time = RewTerm(
        func=nav_mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg":   SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold":    0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=env_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"),
            "threshold":  1.0,
        },
    )

    # ── optional (disabled) ───────────────────────────────────────────────────
    flat_orientation_l2 = RewTerm(func=env_mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits      = RewTerm(func=env_mdp.joint_pos_limits,     weight=0.0)
