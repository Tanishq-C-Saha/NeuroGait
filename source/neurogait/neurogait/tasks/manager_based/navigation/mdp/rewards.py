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


# ── CP5: navigation reward / penalty functions ────────────────────────────────


def reward_progress(env) -> torch.Tensor:
    """Distance reduction to current A* waypoint. Also advances waypoints.

    Returns (num_envs,) float — positive when robot moves toward waypoint.
    Waypoints are advanced when the robot comes within 0.3 m.
    """
    if not hasattr(env, "_curr_waypoint_pos") or env._curr_waypoint_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    curr_dist = torch.norm(robot_xy - env._curr_waypoint_pos, dim=-1)

    progress = env._prev_waypoint_dist - curr_dist
    env._prev_waypoint_dist = curr_dist.clone()

    # vectorised waypoint advancement
    n_wps = len(env._waypoints_tensor)
    can_advance = env._curr_waypoint_idx < (n_wps - 1)
    should_advance = (curr_dist < 0.3) & can_advance

    env._curr_waypoint_idx = torch.clamp(
        env._curr_waypoint_idx + should_advance.long(), max=n_wps - 1
    )
    env._curr_waypoint_pos = env._waypoints_tensor[env._curr_waypoint_idx]

    return progress


def reward_heading(env) -> torch.Tensor:
    """cos(heading error) toward current waypoint. Range [-1, 1].

    Returns (num_envs,) float — 1.0 when perfectly facing waypoint.
    """
    if not hasattr(env, "_curr_waypoint_pos"):
        return torch.zeros(env.num_envs, device=env.device)

    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    robot_quat = env.scene["robot"].data.root_quat_w

    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    dx = env._curr_waypoint_pos[:, 0] - robot_xy[:, 0]
    dy = env._curr_waypoint_pos[:, 1] - robot_xy[:, 1]
    target_yaw = torch.atan2(dy, dx)

    heading_error = torch.atan2(
        torch.sin(target_yaw - robot_yaw),
        torch.cos(target_yaw - robot_yaw),
    )
    return torch.cos(heading_error)


def penalty_collision(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """-1 when base contact force exceeds threshold, 0 otherwise.

    Uses net_forces_w_history (shape E, H, B, 3); sensor_cfg selects body_ids.
    Returns (num_envs,) float — negative or zero.
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history  # (E, H, B, 3)
    # max over history → (E, num_selected_bodies)
    force_mag = torch.max(
        torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
    )[0]
    # collision if ANY selected body exceeds threshold
    is_collision = (force_mag > threshold).any(dim=-1)  # (E,)
    return -is_collision.float()


def penalty_smoothness(env) -> torch.Tensor:
    """-||action_t - action_{t-1}||_2 per environment.

    Penalises jerky velocity commands; encourages smooth trajectories.
    Returns (num_envs,) float — negative or zero.
    """
    if not hasattr(env, "_prev_nav_action"):
        return torch.zeros(env.num_envs, device=env.device)

    curr_action = env.action_manager.action  # (E, 3)
    diff = curr_action - env._prev_nav_action
    env._prev_nav_action = curr_action.clone()
    return -torch.norm(diff, dim=-1)
