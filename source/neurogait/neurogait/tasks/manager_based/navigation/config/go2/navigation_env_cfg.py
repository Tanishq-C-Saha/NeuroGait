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
                        │       └── NeuroGaitNavigationCP1EnvCfg_PLAY
                        └── NeuroGaitNavigationCP5EnvCfg   (CP5: trained RL nav policy)
                                └── NeuroGaitNavigationCP5EnvCfg_PLAY
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass

from neurogait.tasks.manager_based.navigation.navigation_base_env_cfg import (
    NavigationBaseEnvCfg,
)
from isaaclab.envs import mdp as env_mdp

from neurogait.tasks.manager_based.navigation import mdp as nav_mdp
from neurogait.tasks.manager_based.navigation.mdp import (
    occupancy_grid_obs,
    occupancy_grid_obs_gpu,
)

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# ── frozen locomotion env config (needed for PreTrainedPolicyActionCfg) ───────
# We import our own Go2 rough env cfg (same config the checkpoint was trained with).
# Only .actions.joint_pos and .observations.policy are used by the action term.
from neurogait.tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

_LOW_LEVEL_ENV_CFG = UnitreeGo2RoughEnvCfg()


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
        # navigation policy observations — separate group, locomotion actor never reads this
        @configclass
        class NavigationPolicyCfg(ObsGroup):
            occupancy_grid = ObsTerm(func=occupancy_grid_obs)

        self.observations.navigation_policy = NavigationPolicyCfg()


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

        # CP4 navigation can take 30–60 sim-seconds to reach an 8 m goal.
        # The base config sets 20 s which causes mid-run respawns; 120 s
        # is generous enough for any planned path we generate.
        self.episode_length_s = 120.0

        if getattr(self.scene.terrain, "terrain_generator", None) is not None:
            self.scene.terrain.terrain_generator.num_rows   = 5
            self.scene.terrain.terrain_generator.num_cols   = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque     = None
        self.events.push_robot                     = None


# ── CP5: First Trained Navigation Policy ──────────────────────────────────────
# Architecture:
#   Navigation policy (skrl PPO, 3-dim action) → PreTrainedPolicyAction
#   → frozen locomotion policy (TorchScript) → 12 joint targets → physics
#
# The navigation policy only sees navigation observations (1612-dim).
# The locomotion policy's 235-dim obs are handled internally by PreTrainedPolicyAction.


@configclass
class _CP5ActionsCfg:
    """Actions: velocity command → frozen locomotion policy via PreTrainedPolicyAction."""

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = (
        nav_mdp.PreTrainedPolicyActionCfg(
            asset_name="robot",
            # TorchScript export of the frozen Go2 rough locomotion policy
            policy_path=(
                "logs/rsl_rl/unitree_go2_rough/"
                "2026-06-13_19-33-23/exported/policy.pt"
            ),
            low_level_decimation=4,   # loco runs at 50 Hz; nav at 5 Hz
            low_level_actions=_LOW_LEVEL_ENV_CFG.actions.joint_pos,
            low_level_observations=_LOW_LEVEL_ENV_CFG.observations.policy,
        )
    )


@configclass
class _CP5ObservationsCfg:
    """Observations for the navigation RL policy (NOT the locomotion policy)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """What the navigation policy sees — ~1612-dim."""

        # robot kinematics (6 dims)
        base_lin_vel      = ObsTerm(func=env_mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=env_mdp.projected_gravity)

        # goal (3 dims): direction + normalised distance to current A* waypoint
        goal_vector = ObsTerm(func=nav_mdp.goal_vector_obs)

        # velocity (3 dims): vx, vy, yaw_rate in base frame
        robot_velocity = ObsTerm(func=nav_mdp.robot_velocity_obs)

        # local occupancy grid from depth camera (1600 dims: 40×40)
        occupancy_grid = ObsTerm(func=occupancy_grid_obs_gpu)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # Total: 3 + 3 + 3 + 3 + 1600 = 1612 dims


@configclass
class _CP5RewardsCfg:
    """Navigation reward terms — all 4 logged separately in TensorBoard."""

    termination_penalty = RewTerm(func=env_mdp.is_terminated, weight=-200.0)

    progress = RewTerm(
        func=nav_mdp.reward_progress,
        weight=2.0,
    )

    heading = RewTerm(
        func=nav_mdp.reward_heading,
        weight=0.3,
    )

    collision = RewTerm(
        func=nav_mdp.penalty_collision,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold":  1.0,
        },
    )

    smoothness = RewTerm(
        func=nav_mdp.penalty_smoothness,
        weight=0.01,
    )


@configclass
class NeuroGaitNavigationCP5EnvCfg(NeuroGaitNavigationGo2BaseEnvCfg):
    """CP5: Navigation env with frozen locomotion backbone + RL navigation policy.

    Inherits scene (obstacles, height scanner, contact forces) and robot config
    from NeuroGaitNavigationGo2BaseEnvCfg. Adds depth camera, then replaces:
      - actions   → PreTrainedPolicyActionCfg (frozen loco policy)
      - observations.policy → navigation-only obs (1612-dim)
      - rewards   → navigation rewards (progress, heading, collision, smoothness)
    """

    def __post_init__(self):
        super().__post_init__()

        # ── depth camera: RayCasterCamera (Warp raycasting, no RTX renderer) ──
        # Replaces CameraCfg — same depth output, ~100x lower GPU cost.
        # Enables training with 1024+ envs (was limited to ~12 with CameraCfg).
        # 80×60 is sufficient for a 40×40 occupancy grid at 0.2 m/cell.
        self.scene.camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.2,  # once per nav step (decimation=40, sim.dt=0.005)
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.3, 0.0, 0.1),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
            data_types=["distance_to_image_plane"],
            mesh_prim_paths=[
                "/World/ground",                      # flat terrain plane (TerrainImporterCfg prim_path)
                "{ENV_REGEX_NS}/obstacle_cube_.*",    # box obstacles
                "{ENV_REGEX_NS}/obstacle_cyl_.*",     # cylinder obstacles
            ],
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                height=60,
                width=80,
            ),
        )

        # ── replace actions with hierarchical action term ─────────────────────
        self.actions = _CP5ActionsCfg()

        # ── replace observations with navigation-only obs ─────────────────────
        self.observations = _CP5ObservationsCfg()

        # ── replace rewards with navigation rewards ───────────────────────────
        self.rewards = _CP5RewardsCfg()

        # ── disable the locomotion velocity command (not used in CP5) ─────────
        self.commands.base_velocity = None

        # ── add waypoint initialisation event ─────────────────────────────────
        self.events.init_waypoints = EventTerm(
            func=nav_mdp.init_waypoints,
            mode="reset",
        )

        # ── timing: nav policy at 5 Hz, loco policy at 50 Hz ─────────────────
        # sim.dt = 0.005 s (set by base)
        # low_level_decimation = 4 → loco every 4 apply_actions calls
        # nav decimation = 40 → nav policy every 40 sim steps = 0.2 s = 5 Hz
        self.decimation = 40
        self.sim.render_interval = 40     # match high-level decimation (no warning)
        self.episode_length_s = 30.0      # 30 s per episode = 150 nav steps

        # ── sensor update periods ─────────────────────────────────────────────
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation
                * self.sim.dt
            )


@configclass
class NeuroGaitNavigationCP5EnvCfg_PLAY(NeuroGaitNavigationCP5EnvCfg):
    """Single-env play config for evaluating a trained CP5 nav policy."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs    = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        # Longer episodes for evaluation
        self.episode_length_s = 60.0

        if getattr(self.scene.terrain, "terrain_generator", None) is not None:
            self.scene.terrain.terrain_generator.num_rows   = 5
            self.scene.terrain.terrain_generator.num_cols   = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque     = None
        self.events.push_robot                     = None
