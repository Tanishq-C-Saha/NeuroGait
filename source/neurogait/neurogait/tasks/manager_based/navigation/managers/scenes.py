"""Scene configuration for the NeuroGait navigation task."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Terrain scene with a legged robot and optional depth camera."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned"
                "/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven"
                "/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

    # camera is None by default; set in CP3+ env cfgs via scene.camera = CameraCfg(...)
    camera: None = None

    # ── obstacles (CP3.5) ─────────────────────────────────────────────────────
    # All prims under {ENV_REGEX_NS}/Obstacles/ so CP4's build_global_grid can
    # filter them by path prefix without touching robot or terrain prims.
    # Active physics (kinematic_enabled=False), mass 40-65 kg so they react
    # visibly on contact but don't fly away.
    #
    # Layout (top-down, robot starts near origin, goal at ~(8,0)):
    #
    #  Y
    #  ↑
    #  3 |                         [cyl_03]
    #  2 |    [cube_01]                        [cube_05]
    #  1 |             [cyl_01]          [cube_04]
    #  0 | start ─────────────────────────────────── goal(8,0)
    # -1 |        [cube_02]       [cyl_02]
    # -2 |              [cube_03]            [cube_06]
    # -3 |
    #    └──────────────────────────────────────────────────→ X
    #    0     1     2     3     4     5     6     7     8

    obstacle_cube_01: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cube_01",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.8, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.35, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 1.8, 0.25)),
    )

    obstacle_cube_02: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cube_02",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.0, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, -1.0, 0.3)),
    )

    obstacle_cube_03: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cube_03",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 0.7, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=55.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, -1.8, 0.25)),
    )

    obstacle_cube_04: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cube_04",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.5, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=65.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.25, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.5, 1.0, 0.3)),
    )

    obstacle_cube_05: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cube_05",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.8, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.3, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.0, 2.0, 0.25)),
    )

    obstacle_cube_06: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cube_06",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=55.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.5, -2.0, 0.3)),
    )

    obstacle_cyl_01: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cyl_01",
        spawn=sim_utils.CylinderCfg(
            radius=0.5,
            height=0.6,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.45)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, 0.5, 0.3)),
    )

    obstacle_cyl_02: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cyl_02",
        spawn=sim_utils.CylinderCfg(
            radius=0.4,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.5, -1.0, 0.25)),
    )

    obstacle_cyl_03: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cyl_03",
        spawn=sim_utils.CylinderCfg(
            radius=0.45,
            height=0.6,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=42.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.42, 0.48)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 2.8, 0.3)),
    )
