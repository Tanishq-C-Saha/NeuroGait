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
    """Return (curr_wp_world (E,2), dist (E,)) to the robot's current waypoint."""
    from neurogait.tasks.manager_based.navigation.mdp.observations import (
        _cp5_init_waypoint_state, quat_to_yaw_batch,
    )
    _cp5_init_waypoint_state(env)
    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]
    W        = len(env._cp5_waypoints)
    curr_wp  = env._cp5_waypoints[env._cp5_wp_idx.clamp(max=W - 1)]   # (E, 2)
    dist     = torch.norm(curr_wp - robot_xy, dim=-1)                  # (E,)
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
    """Dual-scale tanh proximity shaping.

    r = (1 - tanh(d/5)) + (1 - tanh(d/1))

    Adapted from Li et al. (2025), Eq. 1.
    Smooth gradient at both long (σ=5 m) and short (σ=1 m) ranges.
    """
    _, dist, _ = _cp5_current_wp_and_dist(env)
    return (1.0 - torch.tanh(dist / 5.0)) + (1.0 - torch.tanh(dist / 1.0))


def cp5_reward_goal_reached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse bonus when the final waypoint is within 0.3 m.

    From X-Nav (2025): large one-time signal for episode success.
    """
    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]
    W        = len(env._cp5_waypoints)
    final_wp = env._cp5_waypoints[W - 1]                     # (2,)
    dist     = torch.norm(final_wp.unsqueeze(0) - robot_xy, dim=-1)
    return (dist < 0.3).float()


def cp5_penalty_collision_velocity_scaled(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Velocity-scaled collision penalty.

    r = -(1 + 4(‖v‖² + ωz²)) × 𝟙[non-foot contact]

    Adapted from SEA-Nav (Huang et al., 2026), Table III.
    High-speed collisions cost up to 5× more than low-speed ones.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]

    # Net force on all non-foot body links (E, num_bodies, 3) → scalar per env
    forces_w = contact_sensor.data.net_forces_w          # (E, B, 3)
    force_mag = forces_w.norm(dim=-1)                     # (E, B)

    # Identify foot bodies (names matching ".*foot" — Go2 has FL/FR/RL/RR_foot)
    try:
        foot_ids = sensor_cfg.body_ids  # set via SceneEntityCfg(body_names=".*foot")
        non_foot_mask = torch.ones(force_mag.shape[1], dtype=torch.bool, device=env.device)
        non_foot_mask[foot_ids] = False
        contact = (force_mag[:, non_foot_mask] > 1.0).any(dim=-1).float()  # (E,)
    except Exception:
        # Fallback: any body with force > 1 N counts as collision
        contact = (force_mag > 1.0).any(dim=-1).float()

    vx       = robot.data.root_lin_vel_b[:, 0]
    vy       = robot.data.root_lin_vel_b[:, 1]
    yaw_rate = robot.data.root_ang_vel_b[:, 2]
    vel_scale = 1.0 + 4.0 * (vx ** 2 + vy ** 2 + yaw_rate ** 2)

    return -(vel_scale * contact)


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
    return -stuck.float()


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
    return -diff
