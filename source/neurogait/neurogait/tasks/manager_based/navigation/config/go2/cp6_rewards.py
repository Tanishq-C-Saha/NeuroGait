"""CP6 reward function configuration.

Sources:
  multiplicative core  — Miki et al. 2022 (Science Robotics)
  path following       — pure-pursuit concept + NavRL++
  graduated clearance  — DWA-3D (2024)
  goal proximity       — Li et al. (2025)
  goal reached         — X-Nav (2025)
  collision            — SEA-Nav (Huang et al., 2026)
  stuck                — SEA-Nav (2026) with near-goal patch
  2nd-order smoothness — Go2 task (2025)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from neurogait.tasks.manager_based.navigation import mdp as nav_mdp


@configclass
class CP6RewardsCfg:
    """9 reward terms for CP6 — multiplicative core + path following + graduated clearance."""

    navigation_core = RewTerm(
        func=nav_mdp.cp6_reward_navigation_core,
        weight=10.0,
    )
    path_following = RewTerm(
        func=nav_mdp.cp6_reward_path_following,
        weight=5.0,
    )
    goal_proximity = RewTerm(
        func=nav_mdp.cp5_reward_goal_proximity,
        weight=0.1,
    )
    goal_reached = RewTerm(
        func=nav_mdp.cp5_reward_goal_reached,
        weight=50.0,
    )
    slow_near_goal = RewTerm(
        func=nav_mdp.cp6_reward_slow_near_goal,
        weight=3.0,
    )
    graduated_clearance = RewTerm(
        func=nav_mdp.cp6_penalty_graduated_clearance,
        weight=-1.0,
    )
    collision = RewTerm(
        func=nav_mdp.cp5_penalty_collision_velocity_scaled,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base")},
    )
    stuck = RewTerm(
        func=nav_mdp.cp6_penalty_stuck_v2,
        weight=-0.3,
    )
    smoothness = RewTerm(
        func=nav_mdp.cp6_penalty_smoothness_2nd_order,
        weight=-1.0,
    )
