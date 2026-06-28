from .scene_generator import (
    generate_scene,
    apply_scene_to_env,
    random_goal,
    GO2_WIDTH,
    GO2_LENGTH,
    MIN_GAP,
    MIN_CORRIDOR,   # backwards-compat alias
    SAFETY_MARGIN,
)
from .curriculum import difficulty_at
