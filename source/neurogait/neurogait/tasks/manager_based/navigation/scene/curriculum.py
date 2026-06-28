"""
Curriculum for progressive navigation difficulty.

5 axes ramp from easy to hard over ramp_iterations:
  min_gap_width  : 2.0m → 0.51m  (corridor tightness — Go2 just-squeezable)
  num_obstacles  : 3    → 12     (scene density)
  arena_padding  : 3.0m → 1.5m  (obstacle-to-path proximity — smaller = harder)
  goal_dist      : (6,7)→ (5,10) (goal distance spread)
  goal_angle     : (±0.3)→(±0.8) (goal angle spread)

Usage during training:
    env._curriculum = NavigationCurriculum()
    env._curriculum.step()            # call once per reset batch
    diff = env._curriculum.get_difficulty()

Usage for visualization:
    curriculum = NavigationCurriculum(ramp_iterations=100)
    curriculum.current_iteration = 40   # set to 40% progress
    diff = curriculum.get_difficulty()
"""

from __future__ import annotations

from .scene_generator import MIN_GAP


class NavigationCurriculum:
    """Counter-based 5-axis curriculum for navigation scene generation."""

    def __init__(self, ramp_iterations: int = 40_000):
        """
        Args:
            ramp_iterations: number of `step()` calls before full difficulty.
                             Training default = 40_000 (~2000 training iterations
                             at ~20 reset-event calls per iteration for 512 envs).
                             For visualization, pass ramp_iterations=100 and
                             set current_iteration directly.
        """
        self.ramp_iterations = ramp_iterations
        self.current_iteration = 0

    def step(self) -> None:
        """Increment internal counter by one. Call once per episode reset batch."""
        self.current_iteration = min(self.current_iteration + 1, self.ramp_iterations)

    def get_difficulty(self) -> dict:
        """Return current difficulty params as a dict.

        Keys: min_gap_width, num_obstacles, arena_padding,
              goal_dist (tuple), goal_angle (tuple)
        """
        t = self.current_iteration / max(1, self.ramp_iterations)
        t = min(1.0, t)

        def lerp(a, b):
            return a + t * (b - a)

        def lerp_tuple(a, b):
            return (lerp(a[0], b[0]), lerp(a[1], b[1]))

        return {
            "min_gap_width":  lerp(2.00, MIN_GAP),     # 2.0m → 0.51m
            "num_obstacles":  int(round(lerp(3, 12))),  # 3 → 12
            "arena_padding":  lerp(3.0, 1.5),           # 3.0m → 1.5m
            "goal_dist":      lerp_tuple((6.0, 7.0), (5.0, 10.0)),
            "goal_angle":     lerp_tuple((-0.3, 0.3), (-0.8, 0.8)),
        }

    def progress_str(self) -> str:
        """One-line summary for rate-limited training log."""
        t = min(1.0, self.current_iteration / max(1, self.ramp_iterations))
        d = self.get_difficulty()
        return (
            f"Curriculum {t * 100:.0f}%: "
            f"gap={d['min_gap_width']:.2f}m, "
            f"obs={d['num_obstacles']}, "
            f"pad={d['arena_padding']:.1f}m, "
            f"angle=±{d['goal_angle'][1]:.1f}rad"
        )
