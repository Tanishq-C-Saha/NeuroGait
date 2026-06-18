"""Refrencing Locomotion Network Config"""

from neurogait.tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from isaaclab.utils import configclass


@configclass
class NeuroGaitNavigationCP1PPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """CP1: reuses the locomotion checkpoint's network shape.
    Not a real navigation policy config — replace once CP5 trains an actual nav policy."""
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "neurogait_navigation_cp1"


