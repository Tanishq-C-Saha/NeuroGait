"""Curriculum configuration for the NeuroGait navigation task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """Terrain difficulty curriculum — disabled on flat plane terrain."""

    terrain_levels: CurrTerm | None = None
