"""
Curriculum for progressive navigation difficulty during training.

Difficulty is controlled by 4 axes:
  corridor_width        — 2.0m (easy) → 0.51m (just squeezable for Go2)
  num_obstacles         — 3 (sparse)  → 12 (dense)
  num_control_points    — 2 (gentle)  → 5 (complex winding)
  max_lateral_deviation — 1.0m (mild) → 3.0m (extreme turns)

Progress is tracked via env.common_step_counter (Isaac Lab's built-in
counter, incremented once per env.step() call at nav frequency).

Cite: curriculum learning following DreamerNav (2025) and ANYmal
Parkour (Hoeller et al., 2024).

Progression:
  0%:   corridor=2.00m, 3 obstacles, 2 ctrl-pts — learns to move forward
  25%:  corridor=1.63m, 5 obstacles, 2 ctrl-pts — avoids sparse obstacles
  50%:  corridor=1.26m, 7 obstacles, 3 ctrl-pts — follows curved paths
  75%:  corridor=0.88m, 10 obstacles, 4 ctrl-pts — tight winding gaps
  100%: corridor=0.51m, 12 obstacles, 5 ctrl-pts — just squeezable
"""

from __future__ import annotations

from .scene_generator import MIN_CORRIDOR   # 0.51 m


class NavigationCurriculum:
    """Tracks training progress and returns current difficulty parameters.

    Usage (inside env event function):
        if not hasattr(env, "_curriculum"):
            env._curriculum = NavigationCurriculum()
        difficulty = env._curriculum.get_difficulty(env.common_step_counter)
    """

    # Easy → Hard limits for each difficulty axis
    _PARAMS: dict[str, dict] = {
        "corridor_width":         {"start": 2.00, "end": MIN_CORRIDOR},
        "num_obstacles":          {"start": 3,    "end": 12},
        "num_control_points":     {"start": 2,    "end": 5},
        "max_lateral_deviation":  {"start": 1.0,  "end": 3.0},
    }

    def __init__(self, ramp_steps: int = 24_576_000):
        """
        Args:
            ramp_steps: total env.common_step_counter value at which full
                        difficulty is reached.  Default = 2000 iterations
                        × 24 rollouts × 512 envs = 24,576,000 steps.
                        Scale proportionally for different batch sizes.
        """
        self.ramp_steps = ramp_steps

    def get_difficulty(self, current_steps: int) -> dict:
        """Return current difficulty parameters for the given step count."""
        t = min(1.0, current_steps / max(1, self.ramp_steps))

        result: dict = {}
        for key, limits in self._PARAMS.items():
            val = limits["start"] + t * (limits["end"] - limits["start"])
            if key in ("num_obstacles", "num_control_points"):
                val = int(round(val))
            result[key] = val

        return result

    def progress_str(self, current_steps: int) -> str:
        """One-line progress summary for rate-limited logging."""
        t   = min(1.0, current_steps / max(1, self.ramp_steps))
        d   = self.get_difficulty(current_steps)
        return (
            f"Curriculum {t * 100:.0f}%: "
            f"corridor={d['corridor_width']:.2f}m, "
            f"obstacles={d['num_obstacles']}, "
            f"ctrl_pts={d['num_control_points']}, "
            f"deviation={d['max_lateral_deviation']:.1f}m"
        )
