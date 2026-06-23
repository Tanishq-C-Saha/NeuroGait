"""Manager config classes for the NeuroGait navigation task.

Each file in this package contains exactly one @configclass.
Import from here to assemble navigation_base_env_cfg.py.
"""

from .actions import ActionsCfg
from .commands import CommandsCfg, NullCommandCfg
from .curriculums import CurriculumCfg
from .events import EventCfg
from .observations import ObservationsCfg
from .rewards import RewardsCfg
from .scenes import MySceneCfg
from .terminations import TerminationsCfg
