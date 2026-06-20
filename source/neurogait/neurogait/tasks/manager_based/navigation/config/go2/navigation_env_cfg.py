"""
NeuroGait Navigation — Go2-specific environment configurations.

No longer inherits from the frozen locomotion package. Instead, inherits
from NavigationBaseEnvCfg (owned by the navigation task) via
NeuroGaitNavigationGo2BaseEnvCfg which bakes in all Go2 robot + terrain
+ sensor tuning.

Inheritance chain:
    ManagerBasedRLEnvCfg                         (Isaac Lab)
        └── NavigationBaseEnvCfg                 (navigation_base_env_cfg.py)
                └── NeuroGaitNavigationGo2BaseEnvCfg   (this file)
                        └── NeuroGaitNavigationCP1EnvCfg   (this file, adds camera)
                                └── NeuroGaitNavigationCP1EnvCfg_PLAY
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from neurogait.tasks.manager_based.navigation.navigation_base_env_cfg import (
    NavigationBaseEnvCfg,
)
from neurogait.tasks.manager_based.navigation.mdp import occupancy_grid_obs

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


# ── Go2 rough-terrain base ────────────────────────────────────────────────────


@configclass
class NeuroGaitNavigationGo2BaseEnvCfg(NavigationBaseEnvCfg):
    """
    Go2 robot + terrain tuning, no camera yet.

    All overrides that used to be spread across locomotion's
    UnitreeGo2RoughEnvCfg and this file are consolidated here so every
    future navigation CP can inherit from a single clean base.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── robot ────────────────────────────────────────────────────────────
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # ── terrain: scaled for the Go2's small footprint ────────────────────
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step  = 0.01

        # ── actions ──────────────────────────────────────────────────────────
        self.actions.joint_pos.scale = 0.25

        # ── events ───────────────────────────────────────────────────────────
        self.events.push_robot = None
        self.events.base_com   = None
        self.events.add_base_mass.params["mass_distribution_params"]      = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names          = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x":     (0.0, 0.0), "y":     (0.0, 0.0), "z":   (0.0, 0.0),
                "roll":  (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # ── rewards ──────────────────────────────────────────────────────────
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight         = 0.01
        self.rewards.undesired_contacts           = None
        self.rewards.dof_torques_l2.weight        = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight  = 1.5
        self.rewards.track_ang_vel_z_exp.weight   = 0.75
        self.rewards.dof_acc_l2.weight            = -2.5e-7

        # ── terminations ─────────────────────────────────────────────────────
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


# ── CP3: depth camera + occupancy grid observation ────────────────────────────


@configclass
class NeuroGaitNavigationCP1EnvCfg(NeuroGaitNavigationGo2BaseEnvCfg):
    """
    CP3 env: Go2 + RealSense D455-style depth camera + occupancy grid obs.

    Only adds what's new vs. the Go2 base:
      - depth camera on Robot/base/front_cam
      - occupancy_grid obs term (→ 1600-dim flattened 40×40 grid)
      - velocity command pinned for testing (fixed forward, no heading)
    """

    def __post_init__(self):
        super().__post_init__()

        # ── depth camera ─────────────────────────────────────────────────────
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0.0, 0.1),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )

        # ── occupancy grid observation term (CP3) ────────────────────────────
        self.observations.policy.occupancy_grid = ObsTerm(func=occupancy_grid_obs)

        # ── pin velocity command for deterministic CP3 testing ───────────────
        self.commands.base_velocity.heading_command      = False
        self.commands.base_velocity.ranges.lin_vel_x    = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y    = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z    = (0.0, 0.0)


@configclass
class NeuroGaitNavigationCP1EnvCfg_PLAY(NeuroGaitNavigationCP1EnvCfg):
    """Single-env play config for manual inspection."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs    = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows   = 5
            self.scene.terrain.terrain_generator.num_cols   = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque     = None
        self.events.push_robot                     = None
