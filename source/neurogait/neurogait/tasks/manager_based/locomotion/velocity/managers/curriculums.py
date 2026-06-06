"""Curriculums for locomotion velocity manager-based task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from neurogait.tasks.manager_based.locomotion.velocity import mdp

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
