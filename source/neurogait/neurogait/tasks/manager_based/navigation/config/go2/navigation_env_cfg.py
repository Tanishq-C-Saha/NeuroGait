"""
NeuroGait Navigation — Go2-specific environment configurations.

Inheritance chain:
    ManagerBasedRLEnvCfg                                  (Isaac Lab)
        └── NavigationBaseEnvCfg                          (navigation_base_env_cfg.py)
                └── NeuroGaitNavigationGo2BaseEnvCfg      (this file)
                        ├── NeuroGaitNavigationCP1EnvCfg  (CP3/CP4: rule-based, CameraCfg)
                        │       └── NeuroGaitNavigationCP1EnvCfg_PLAY
                        └── NeuroGaitNavigationCP5EnvCfg  (CP5: trained policy, RayCaster)
                                ├── NeuroGaitNavigationCP5EnvCfg_PLAY
                                └── NeuroGaitNavigationCP6EnvCfg  (CP6: randomized A* obstacles)
                                        ├── NeuroGaitNavigationCP6EnvCfg_PLAY
                                        └── NeuroGaitNavigationCP65EnvCfg (CP6.5: path-first + curriculum)
                                                └── NeuroGaitNavigationCP65EnvCfg_PLAY
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from neurogait.tasks.manager_based.navigation.navigation_base_env_cfg import (
    NavigationBaseEnvCfg,
)
from neurogait.tasks.manager_based.navigation import mdp as nav_mdp
from neurogait.tasks.manager_based.navigation.mdp import occupancy_grid_obs  # CP1

from .cp5_rewards import CP5RewardsCfg
from .cp6_rewards import CP6RewardsCfg
from neurogait.tasks.manager_based.navigation.managers import CurriculumCfg as _BaseCurriculumCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# ── CP5: Frozen locomotion policy (TorchScript, exported by rsl_rl play.py) ──
_LOCO_PT = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..", "..", "..", "..",  # → project root
    "logs", "rsl_rl", "unitree_go2_rough",
    "2026-06-13_19-33-23", "exported", "policy.pt",
)
_LOCO_PT = os.path.normpath(_LOCO_PT)

# LOW_LEVEL_ENV_CFG: must use the rough variant to match 235-dim trained model
# (see concept/cp5_concepts/research_notes.md — Research C)
from neurogait.tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (  # noqa: E402
    UnitreeGo2RoughEnvCfg,
)
LOW_LEVEL_ENV_CFG = UnitreeGo2RoughEnvCfg()


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


# ── CP5: Trained navigation policy ────────────────────────────────────────────

#
# Key design decisions (see concept/cp5_concepts/ for full rationale):
#   - Actions:  PreTrainedPolicyActionCfg wraps frozen locomotion TorchScript
#   - Camera:   MultiMeshRayCasterCameraCfg (Warp GPU, not RTX CameraCfg)
#   - Obs:      1615 dims — grid(1600) + waypoints(9) + vel(3) + gravity(3)
#   - Timing:   decimation=40 → nav step = 0.2 s (5 Hz); loco runs at 50 Hz
#   - Rewards:  7 terms from SEA-Nav, Li et al. 2025, X-Nav 2025
#   - heading_command=True: action[2] = heading angle NOT yaw rate


@configclass
class NeuroGaitNavigationCP5EnvCfg(NeuroGaitNavigationGo2BaseEnvCfg):
    """CP5: Trained navigation policy with PreTrainedPolicyAction + RayCaster camera.

    Inherits the scene (obstacles, terrain, robot, sensors) from the Go2 base,
    then replaces actions, observations.policy, rewards, and timing.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── 1. GPS-level timing: nav at 5 Hz, loco at 50 Hz ─────────────────
        # low_level_decimation=4 → locomotion step = 4 × 0.005 = 0.02 s = 50 Hz
        # navigation decimation = 40 → nav step    = 40 × 0.005 = 0.20 s = 5 Hz
        self.sim.dt             = 0.005
        self.decimation         = 40
        self.scene.env_spacing  = 10
        self.sim.render_interval = self.decimation   # CRITICAL: must equal decimation
        self.episode_length_s   = 120

        # ── 2. Replace actions with PreTrainedPolicyActionCfg ─────────────────
        self.actions.joint_pos = None                              # disable loco joint action
        self.actions.pre_trained_policy_action = (
            nav_mdp.PreTrainedPolicyActionCfg(
                asset_name="robot",
                policy_path=_LOCO_PT,
                low_level_decimation=4,
                low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
                low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
            )
        )

        # ── 3. MultiMeshRayCasterCamera (GPU-efficient, replaces CameraCfg) ────
        # scene["front_cam"] is what occupancy_grid_obs_cp5 reads
        self.scene.front_cam = MultiMeshRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            mesh_prim_paths=["/World/ground", "{ENV_REGEX_NS}/obstacle_.*"],
            update_period=0.2,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=(0.3, 0.0, 0.1),
                rot=(0.4305, -0.5610, 0.5610, -0.4305),
                convention="ros",
            ),
            data_types=["distance_to_image_plane"],
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=1.88,
                width=80,
                height=60,
            ),
        )

        # ── 4. Replace observations.policy with CP5 navigation obs (1615 dims) ─
        # ORDER IS CRITICAL: grid [0:1600] FIRST so CNN splits at index 1600.
        @configclass
        class CP5PolicyCfg(ObsGroup):
            # Grid must be first — CNN splits obs at index 1600
            occupancy_grid    = ObsTerm(func=nav_mdp.occupancy_grid_obs_cp5)   # 1600
            future_waypoints  = ObsTerm(func=nav_mdp.future_waypoints_obs)     # 9
            robot_velocity    = ObsTerm(func=nav_mdp.robot_velocity_obs)       # 3
            projected_gravity = ObsTerm(func=env_mdp.projected_gravity)        # 3
            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        self.observations.policy = CP5PolicyCfg()

        # ── 5. Replace rewards with 7 navigation terms (null locomotion ones) ──
        self.rewards = CP5RewardsCfg()  # type: ignore[assignment]

        # ── 6. Keep commands alive (harmless, base_velocity command exists) ───
        self.commands.base_velocity.heading_command = True

        # ── 7. Height scanner update period (required by locomotion internal obs)
        self.scene.height_scanner.update_period = (
            4 * self.sim.dt   # low_level_decimation × sim.dt = 0.02 s
        )


@configclass
class NeuroGaitNavigationCP5EnvCfg_PLAY(NeuroGaitNavigationCP5EnvCfg):
    """Single-env CP5 play config for evaluation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs     = 1
        self.scene.env_spacing  = 10
        self.episode_length_s   = 120.0   # generous for long paths

        if getattr(self.scene.terrain, "terrain_generator", None) is not None:
            self.scene.terrain.terrain_generator.num_rows   = 5
            self.scene.terrain.terrain_generator.num_cols   = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque     = None
        self.events.push_robot                     = None


# ── CP6: Generalized navigation + upgraded rewards ────────────────────────────
#
# What changes from CP5 (see concept/cp6_concepts/ for full rationale):
#   Rewards:  multiplicative core (Miki 2022), path-following, graduated
#             clearance (DWA-3D 2024), goal termination, 2nd-order smoothness
#   Events:   obstacles randomized ±1.5 m on every episode reset; A* replans
#   Termination: episode ends when robot is within 0.5 m of goal
#   Fixes:    crab-walking, long detours, no goal-stop from CP5


@configclass
class NeuroGaitNavigationCP6EnvCfg(NeuroGaitNavigationCP5EnvCfg):
    """CP6: generalized navigation — randomized obstacles + upgraded rewards.

    Inherits the full CP5 setup (PreTrainedPolicyAction, RayCaster camera,
    1615-dim obs, 5 Hz nav / 50 Hz loco timing) and overrides:
      - rewards      → CP6RewardsCfg (9 terms, multiplicative core)
      - events       → adds obstacle randomization EventTerm
      - terminations → adds goal-reached DoneTerm
    """

    def __post_init__(self):
        super().__post_init__()

        # ── 1. Upgraded reward function ───────────────────────────────────────
        self.rewards = CP6RewardsCfg()  # type: ignore[assignment]

        # ── 2. Obstacle randomization on every episode reset ──────────────────
        # Dynamic field addition — Isaac Lab managers discover terms via dir(),
        # so new fields set on @configclass instances are picked up at runtime.
        # (Same pattern as CP5's self.actions.pre_trained_policy_action = ...)
        self.events.randomize_obstacles = EventTerm(  # type: ignore[attr-defined]
            func=nav_mdp.cp6_randomize_obstacles_and_replan,
            mode="reset",
            params={"position_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
        )

        # ── 3. Goal-reached episode termination ───────────────────────────────
        self.terminations.goal_reached = DoneTerm(  # type: ignore[attr-defined]
            func=nav_mdp.cp6_goal_reached,
            params={"threshold": 0.1},
        )


@configclass
class NeuroGaitNavigationCP6EnvCfg_PLAY(NeuroGaitNavigationCP6EnvCfg):
    """Single-env CP6 play config for evaluation and trajectory visualisation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs    = 1
        self.scene.env_spacing = 10
        self.episode_length_s  = 120.0

        if getattr(self.scene.terrain, "terrain_generator", None) is not None:
            self.scene.terrain.terrain_generator.num_rows   = 5
            self.scene.terrain.terrain_generator.num_cols   = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque     = None
        self.events.push_robot                     = None


@configclass
class CP65CurriculumCfg(_BaseCurriculumCfg):
    """5-axis difficulty curriculum for CP6.5.

    Isaac Lab's CurriculumManager calls obstacle_difficulty automatically
    inside _reset_idx() — before the reset EventTerm fires — so
    cp65_reset_with_generated_scene always reads fresh env._curr_* values.
    """

    obstacle_difficulty = CurrTerm(
        func=nav_mdp.curriculum_obstacle_difficulty,
        params={
            "ramp_iterations": 2_000,   # training iterations to full difficulty
            "gap_start": 2.0,  "gap_end": 0.51,
            "obs_start": 3,    "obs_end": 12,
            "pad_start": 3.0,  "pad_end": 1.5,
            "angle_start": 0.3, "angle_end": 0.8,
        },
    )


@configclass
class NeuroGaitNavigationCP65EnvCfg(NeuroGaitNavigationCP6EnvCfg):
    """CP6.5 — Obstacles-first scene generation + Isaac Lab curriculum.

    Uses CurriculumTermCfg to ramp difficulty automatically — no manual
    step() calls, no off-by-one bugs.  Difficulty is read by the reset
    event from env._curr_* attributes the curriculum term writes first.
    """

    def __post_init__(self):
        super().__post_init__()

        # Wire in the built-in curriculum manager
        self.curriculum = CP65CurriculumCfg()

        # Swap reset event: obstacles-first generation + random goals
        # (no ramp_iterations param — curriculum is handled by CurriculumTerm above)
        self.events.randomize_obstacles = EventTerm(  # type: ignore[attr-defined]
            func=nav_mdp.cp65_reset_with_generated_scene,
            mode="reset",
        )


@configclass
class NeuroGaitNavigationCP65EnvCfg_PLAY(NeuroGaitNavigationCP65EnvCfg):
    """Single-env CP6.5 play config for evaluation and trajectory visualisation."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs    = 1
        self.scene.env_spacing = 10
        self.episode_length_s  = 120.0

        if getattr(self.scene.terrain, "terrain_generator", None) is not None:
            self.scene.terrain.terrain_generator.num_rows   = 5
            self.scene.terrain.terrain_generator.num_cols   = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque     = None
        self.events.push_robot                     = None
