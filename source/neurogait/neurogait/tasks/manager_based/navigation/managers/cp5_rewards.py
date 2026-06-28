"""CP5 reward function configuration."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from neurogait.tasks.manager_based.navigation import mdp as nav_mdp


@configclass
class CP5RewardsCfg:
    """7 research-backed navigation reward terms for CP5."""

    velocity_toward_goal = RewTerm(
        func=nav_mdp.cp5_reward_velocity_toward_goal,
        weight=10.0,
    )
    goal_proximity = RewTerm(
        func=nav_mdp.cp5_reward_goal_proximity,
        weight=0.1,   # was 3.0 — reduced to prevent 10× dominance over velocity_toward_goal
    )
    goal_reached = RewTerm(
        func=nav_mdp.cp5_reward_goal_reached,
        weight=20.0,
    )
    collision = RewTerm(
        func=nav_mdp.cp5_penalty_collision_velocity_scaled,
        weight=-5.0,
        params={
            # "base" body only — avoids false-zero from masking all bodies with ".*"
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
        },
    )
    stuck = RewTerm(
        func=nav_mdp.cp5_penalty_stuck,
        weight=-0.3,
    )
    heading = RewTerm(
        func=nav_mdp.cp5_reward_heading,
        weight=0.1,
    )
    smoothness = RewTerm(
        func=nav_mdp.cp5_penalty_smoothness,
        weight=-0.01,
    )
