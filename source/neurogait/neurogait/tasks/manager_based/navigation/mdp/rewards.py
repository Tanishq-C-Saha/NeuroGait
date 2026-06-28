"""Reward functions for the NeuroGait navigation task.

Copied from isaaclab_tasks locomotion velocity mdp, then extended for
navigation-specific needs. The locomotion rewards stay here so we can
tune weights and add new navigation rewards without touching the frozen
locomotion package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding on the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy) in the yaw-aligned robot frame."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from default joint positions when command is near zero."""
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# CP5 navigation reward terms
# Literature: SEA-Nav (Huang et al., 2026), Li et al. (2025), X-Nav (2025)
# ─────────────────────────────────────────────────────────────────────────────

def _cp5_current_wp_and_dist(env):
    """Return (curr_wp_world (E,2), dist (E,), robot_xy (E,2))."""
    from neurogait.tasks.manager_based.navigation.mdp.observations import (
        _cp5_init_waypoint_state, quat_to_yaw_batch,
    )
    _cp5_init_waypoint_state(env)
    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]
    # _cp5_waypoints is now (E, W, 2) — one world-space path per env
    W        = env._cp5_waypoints.shape[1]
    E_range  = torch.arange(env.num_envs, device=env.device)
    curr_wp  = env._cp5_waypoints[E_range, env._cp5_wp_idx.clamp(max=W - 1)]  # (E, 2)
    dist     = torch.norm(curr_wp - robot_xy, dim=-1)                           # (E,)
    return curr_wp, dist, robot_xy


def cp5_reward_velocity_toward_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Speed-weighted progress reward.

    r = cos(heading_error) × vx × (1 + 1/(1 + 2d²))

    Adapted from SEA-Nav (Huang et al., 2026), Table III.
    Rewards approach SPEED rather than just distance reduction.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import quat_to_yaw_batch

    robot    = env.scene["robot"]
    curr_wp, dist, robot_xy = _cp5_current_wp_and_dist(env)
    robot_yaw = quat_to_yaw_batch(robot.data.root_quat_w)    # (E,)

    goal_angle = torch.atan2(
        curr_wp[:, 1] - robot_xy[:, 1],
        curr_wp[:, 0] - robot_xy[:, 0],
    )
    heading_err = goal_angle - robot_yaw
    # Wrap to [-π, π]
    heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

    vx          = robot.data.root_lin_vel_b[:, 0]
    proximity   = 1.0 + 1.0 / (1.0 + 2.0 * dist ** 2)
    reward      = torch.cos(heading_err) * vx.clamp(min=0.0) * proximity

    # Update previous distance for optional use in other terms
    env._cp5_prev_dist[:] = dist
    return reward


def cp5_reward_goal_proximity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dual-scale tanh proximity shaping toward the FINAL goal.

    r = (1 - tanh(d/5)) + (1 - tanh(d/1))

    Adapted from Li et al. (2025), Eq. 1.
    Uses the FINAL waypoint (destination), NOT the current waypoint.
    This gives a persistent gradient that never resets when the robot
    advances through intermediate waypoints.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)
    robot_xy   = env.scene["robot"].data.root_pos_w[:, :2]          # (E, 2)
    final_goal = env._cp5_waypoints[:, -1, :]                       # (E, 2) per-env destination
    dist = torch.norm(robot_xy - final_goal, dim=-1)                # (E,)
    return (1.0 - torch.tanh(dist / 5.0)) + (1.0 - torch.tanh(dist / 1.0))


def cp5_reward_goal_reached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse bonus when the final waypoint is within 0.3 m.

    From X-Nav (2025): large one-time signal for episode success.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)
    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]
    W        = env._cp5_waypoints.shape[1]
    final_wp = env._cp5_waypoints[:, W - 1, :]               # (E, 2) per-env final waypoint
    dist     = torch.norm(final_wp - robot_xy, dim=-1)
    return (dist < 0.3).float()


def cp5_penalty_collision_velocity_scaled(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="base"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Velocity-scaled collision penalty for trunk (base) body contact.

    r = (1 + 4(‖v‖² + ωz²)) × 𝟙[base-body contact force > threshold]

    sensor_cfg should specify body_names="base" so only the trunk is checked.
    Using ".*" causes all bodies (including feet) to be in body_ids, which
    makes the internal mask empty → contact always zero.

    Adapted from SEA-Nav (Huang et al., 2026), Table III.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]

    # body_ids resolved by Isaac Lab's manager framework against sensor_cfg.body_names
    forces_w  = contact_sensor.data.net_forces_w                      # (E, B_total, 3)
    body_ids  = sensor_cfg.body_ids                                    # list of selected body indices
    force_mag = forces_w[:, body_ids, :].norm(dim=-1)                 # (E, B_selected)
    contact   = (force_mag.max(dim=-1).values > force_threshold).float()  # (E,)

    vx       = robot.data.root_lin_vel_b[:, 0]
    vy       = robot.data.root_lin_vel_b[:, 1]
    yaw_rate = robot.data.root_ang_vel_b[:, 2]
    vel_scale = 1.0 + 4.0 * (vx ** 2 + vy ** 2 + yaw_rate ** 2)

    return vel_scale * contact   # positive magnitude; weight=-5.0 makes this a penalty


def cp5_penalty_stuck(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise robot that commands forward motion but doesn't move.

    Adapted from SEA-Nav (Huang et al., 2026), Table III.
    Sliding window of 20 steps; fires when max position change < 0.1 m.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)

    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2].detach()   # (E, 2)

    # Update ring buffer
    idx = env._cp5_pos_hist_idx % 20
    env._cp5_pos_history[:, idx, :] = robot_xy
    env._cp5_pos_hist_idx += 1

    # Max displacement over the history window
    pos_spread = (env._cp5_pos_history - robot_xy.unsqueeze(1)).norm(dim=-1)  # (E, 20)
    max_disp   = pos_spread.max(dim=-1).values                                 # (E,)

    # Fire penalty when stuck AND commanding forward motion
    nav_actions = env.action_manager.action                           # (E, 3) from PreTrainedPolicyAction
    commanding_forward = (nav_actions[:, 0] > 0.1) if nav_actions.shape[-1] >= 1 else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    stuck  = (max_disp < 0.1) & commanding_forward
    return stuck.float()   # positive magnitude; weight=-3.0 makes this a penalty


def cp5_reward_heading(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gentle heading alignment reward: cos(heading_error).

    Standard formulation. Small weight (0.5) — hint only.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import quat_to_yaw_batch
    robot    = env.scene["robot"]
    curr_wp, _, robot_xy = _cp5_current_wp_and_dist(env)
    robot_yaw = quat_to_yaw_batch(robot.data.root_quat_w)
    goal_angle = torch.atan2(curr_wp[:, 1] - robot_xy[:, 1],
                              curr_wp[:, 0] - robot_xy[:, 0])
    heading_err = torch.atan2(torch.sin(goal_angle - robot_yaw),
                               torch.cos(goal_angle - robot_yaw))
    return torch.cos(heading_err)


def cp5_penalty_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise jerky velocity commands: -‖action_t - action_{t-1}‖.

    From X-Nav (2025). Tiny weight (0.01) — regularizer only.
    Tracks the 3D navigation action, NOT the 12D joint targets.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)

    curr_action = env.action_manager.action   # (E, 3) — nav policy output
    diff        = (curr_action - env._cp5_prev_action).norm(dim=-1)
    env._cp5_prev_action[:] = curr_action.detach()
    return diff   # positive magnitude; weight=-0.01 makes this a penalty


# ─────────────────────────────────────────────────────────────────────────────
# CP6 navigation reward terms
# Literature: Miki et al. 2022 (Science Robotics), DWA-3D 2024, Go2 task 2025
# ─────────────────────────────────────────────────────────────────────────────


def cp6_reward_navigation_core(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Multiplicative reward: r_forward × r_lateral × r_heading.

    Each component is a Gaussian centred on the desired behaviour.
    If ANY component is near zero the entire reward collapses to zero —
    the robot must face forward, suppress lateral drift, AND move toward
    the waypoint simultaneously to earn any reward.

    Sources: Miki et al. 2022 (Science Robotics) orthogonal velocity
    concept; multiplicative structure from biomechanics nav 2025.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import quat_to_yaw_batch

    robot              = env.scene["robot"]
    curr_wp, dist, robot_xy = _cp5_current_wp_and_dist(env)
    robot_yaw          = quat_to_yaw_batch(robot.data.root_quat_w)   # (E,)

    dx = curr_wp[:, 0] - robot_xy[:, 0]
    dy = curr_wp[:, 1] - robot_xy[:, 1]
    target_yaw    = torch.atan2(dy, dx)
    heading_err   = torch.atan2(
        torch.sin(target_yaw - robot_yaw),
        torch.cos(target_yaw - robot_yaw),
    )   # wrapped to [-π, π]

    vx = robot.data.root_lin_vel_b[:, 0]   # forward body velocity
    vy = robot.data.root_lin_vel_b[:, 1]   # lateral body velocity

    target_speed = 0.8   # desired approach speed (m/s)

    # Forward: Gaussian around target projected speed
    r_forward = torch.exp(
        -((vx - torch.cos(heading_err) * target_speed) ** 2) / 0.25
    )
    # Lateral: penalise sideways drift (σ² = 1/3)
    r_lateral = torch.exp(-3.0 * vy ** 2)
    # Heading: penalise facing away from waypoint (σ² = 0.25)
    r_heading = torch.exp(-(heading_err ** 2) / 0.25)

    env._cp5_prev_dist[:] = dist
    return r_forward * r_lateral * r_heading   # all three must be satisfied


def cp6_reward_path_following(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gaussian proximity to the A* planned path (σ² = 1.0 m²).

    The A* path routes through gaps in obstacles; rewarding proximity to
    the path teaches the robot to use those gaps rather than taking long
    detours around the entire obstacle field.

    Concept: pure-pursuit path adherence, adapted for RL.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)

    robot_xy  = env.scene["robot"].data.root_pos_w[:, :2]   # (E, 2)
    waypoints = env._cp5_waypoints                           # (E, W, 2)

    diff      = robot_xy.unsqueeze(1) - waypoints            # (E, W, 2)
    dists     = diff.norm(dim=-1)                            # (E, W)
    min_dist  = dists.min(dim=-1).values                     # (E,)

    return torch.exp(-min_dist ** 2 / 1.0)


def cp6_penalty_graduated_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gaussian obstacle proximity penalty from the nearest depth return.

    Output is always in [0, 1]:
      d = 0.0 m  →  1.00   (contact)
      d = 0.5 m  →  0.61   (very close)
      d = 1.0 m  →  0.14   (approaching)
      d = 1.5 m  →  0.011  (effectively free)

    Formula: exp(-d² / σ²),  σ² = 0.5 m²
    With weight=-0.05, max contribution per step = -0.05.

    Previous version multiplied by robot speed with no upper clamp, producing
    raw values up to 75 during random-action tests (speed ~15 m/s from physics
    instability) which drowned all positive rewards at weight=-1.0.

    Source: DWA-3D (2024) — proximity shaping.
    """
    camera = env.scene["front_cam"]
    depth  = camera.data.output["distance_to_image_plane"]   # (E, H, W)

    depth_clean = depth.clone()
    depth_clean[torch.isnan(depth_clean)] = 10.0   # rays that miss = far
    depth_clean[depth_clean <= 0]         = 10.0

    min_depth = depth_clean.reshape(env.num_envs, -1).min(dim=-1).values  # (E,)

    return torch.exp(-(min_depth ** 2) / 0.5)   # (E,) in [0, 1]; weight=-0.05 applies penalty


def cp6_reward_slow_near_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward low speed when within 1.5 m of the final goal.

    Teaches the robot to decelerate on approach.  Combined with
    cp6_goal_reached (termination) the learned behaviour is:
      fast approach → slow near goal → stop → terminal reward → done.

    Source: X-Nav (2025), deceleration curriculum.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)

    robot_xy   = env.scene["robot"].data.root_pos_w[:, :2]
    final_goal = env._cp5_waypoints[:, -1, :]                      # (E, 2)
    dist       = torch.norm(robot_xy - final_goal, dim=-1)          # (E,)
    speed      = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)

    near_goal = (dist < 1.5).float()
    return near_goal * (1.0 - speed.clamp(max=1.0))


def cp6_penalty_stuck_v2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stuck penalty with near-goal exemption.

    Same sliding-window logic as cp5_penalty_stuck but suppressed when
    the robot is within 1.5 m of the goal (where low speed is desired).

    Source: SEA-Nav (Huang et al., 2026) with near-goal patch.
    """
    from neurogait.tasks.manager_based.navigation.mdp.observations import _cp5_init_waypoint_state
    _cp5_init_waypoint_state(env)

    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2].detach()

    final_goal   = env._cp5_waypoints[:, -1, :]
    dist_to_goal = torch.norm(robot_xy - final_goal, dim=-1)
    near_goal    = dist_to_goal < 1.5

    idx                              = env._cp5_pos_hist_idx % 20
    env._cp5_pos_history[:, idx, :] = robot_xy
    env._cp5_pos_hist_idx           += 1

    pos_spread = (env._cp5_pos_history - robot_xy.unsqueeze(1)).norm(dim=-1)
    max_disp   = pos_spread.max(dim=-1).values

    nav_actions       = env.action_manager.action
    commanding_fwd    = (
        (nav_actions[:, 0] > 0.1)
        if nav_actions.shape[-1] >= 1
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )

    stuck = (max_disp < 0.1) & commanding_fwd & ~near_goal
    return stuck.float()   # positive magnitude; weight=-0.3 makes this a penalty


def cp6_penalty_smoothness_2nd_order(env: ManagerBasedRLEnv) -> torch.Tensor:
    """First + second order action smoothness penalty.

    First-order catches large step changes; second-order catches oscillations
    that look smooth step-to-step but alternate direction every step.

    Source: Go2 task (2025), "First Order Model-Based RL".
    """
    curr = env.action_manager.action   # (E, 3)

    if not hasattr(env, "_cp6_prev_action_1"):
        env._cp6_prev_action_1 = curr.clone()
        env._cp6_prev_action_2 = curr.clone()

    d1 = curr             - env._cp6_prev_action_1
    d2 = env._cp6_prev_action_1 - env._cp6_prev_action_2

    first_order  = d1.norm(dim=-1)
    second_order = (d1 - d2).norm(dim=-1)

    env._cp6_prev_action_2 = env._cp6_prev_action_1.clone()
    env._cp6_prev_action_1 = curr.detach().clone()

    return 0.01 * first_order + 0.005 * second_order  # positive; weight=-1.0
