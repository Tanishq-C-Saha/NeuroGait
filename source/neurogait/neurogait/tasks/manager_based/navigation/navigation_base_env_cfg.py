"""
NeuroGait Navigation — base environment configuration.

This file only assembles the pieces; all @configclass blocks live in
navigation/managers/ (one per manager). The mdp/ folder holds the
Python functions those managers call.

Inheritance chain for Go2:
    ManagerBasedRLEnvCfg          (Isaac Lab)
        └── NavigationBaseEnvCfg  (this file)
                └── NeuroGaitNavigationGo2BaseEnvCfg   (config/go2/navigation_env_cfg.py)
                        └── NeuroGaitNavigationCP1EnvCfg
                                └── ...PLAY
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from neurogait.tasks.manager_based.navigation.managers import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class NavigationBaseEnvCfg(ManagerBasedRLEnvCfg):
    """
    Base environment for all NeuroGait navigation tasks.

    Owns its own class hierarchy — does not inherit from the frozen
    locomotion package.  Robot-specific tuning goes in the Go2 subclass.
    """

    scene:        MySceneCfg      = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    commands:     CommandsCfg     = CommandsCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:       EventCfg        = EventCfg()
    curriculum:   CurriculumCfg   = CurriculumCfg()

    def __post_init__(self):
        self.decimation        = 4
        self.episode_length_s  = 20.0
        self.sim.dt             = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
