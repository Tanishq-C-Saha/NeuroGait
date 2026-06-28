"""Difficulty interpolation helper for CP6.5 navigation curriculum.

During training, difficulty is driven by the Isaac Lab CurriculumTermCfg
``curriculum_obstacle_difficulty`` (in mdp/curriculums.py).  Isaac Lab calls
that term automatically before every reset event — no manual step() needed.

This module keeps ``difficulty_at(t)`` as a pure function for standalone
visualization scripts that have no live env object.
"""

from __future__ import annotations

from .scene_generator import MIN_GAP


def difficulty_at(t: float) -> dict:
    """Return difficulty params at curriculum progress t ∈ [0, 1].

    Keys: min_gap_width, num_obstacles, arena_padding,
          goal_dist (tuple), goal_angle (tuple).
    """
    t = max(0.0, min(1.0, t))

    def lerp(a: float, b: float) -> float:
        return a + t * (b - a)

    def lerp_t(a: tuple, b: tuple) -> tuple:
        return (lerp(a[0], b[0]), lerp(a[1], b[1]))

    return {
        "min_gap_width":  lerp(2.00, MIN_GAP),
        "num_obstacles":  int(round(lerp(3, 12))),
        "arena_padding":  lerp(3.0, 1.5),
        "goal_dist":      lerp_t((6.0, 7.0), (5.0, 10.0)),
        "goal_angle":     lerp_t((-0.3, 0.3), (-0.8, 0.8)),
    }
