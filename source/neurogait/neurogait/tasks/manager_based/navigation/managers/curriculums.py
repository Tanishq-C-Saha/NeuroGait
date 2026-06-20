"""Curriculum configuration for the NeuroGait navigation task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from neurogait.tasks.manager_based.navigation import mdp as nav_mdp


@configclass
class CurriculumCfg:
    """Terrain difficulty curriculum."""

    terrain_levels = CurrTerm(func=nav_mdp.terrain_levels_vel)
