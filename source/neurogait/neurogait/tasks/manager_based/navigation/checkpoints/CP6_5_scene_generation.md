# CP6.5 — Path-First Scene Generation + Curriculum

## Core Idea

CP6 used A* to plan around randomly placed obstacles. The key insight for CP6.5:

> **Generate the path first, then place obstacles outside it.**

This guarantees every generated scene is traversable by construction, eliminating
the need for A* during training entirely. Replanning cost drops to zero.

---

## Architecture

```
episode reset
    └── cp65_reset_with_generated_scene()
            ├── NavigationCurriculum.get_difficulty(common_step_counter)
            ├── generate_scene(start, goal, difficulty params)
            │       ├── _generate_random_path()   ← cubic spline, scipy or numpy
            │       └── _place_obstacles()        ← rejection sampling outside corridor
            ├── apply_scene_to_env()              ← broadcast local→world, all envs
            └── env._cp5_waypoints = (E, W, 2)   ← same tensor rewards use (CP6 rewards unchanged)
```

No A* is called. `_cp5_init_waypoint_state` is bypassed via `hasattr` guard
(the generator sets `env._cp5_waypoints` before obs functions check it).

---

## Go2 C-Space Constants

| Parameter         | Value   | Notes                            |
|-------------------|---------|----------------------------------|
| `GO2_BODY_WIDTH`  | 0.31 m  | Actual spec (not estimate)       |
| `GO2_BODY_LENGTH` | 0.70 m  | Actual spec                      |
| `SAFETY_MARGIN`   | 0.10 m  | Per-side clearance beyond body   |
| `MIN_CORRIDOR`    | 0.51 m  | `0.31 + 2×0.10` — just squeezable |

Corridor half-width = 0.255 m (robot centre is always ≥ 0.255 m from every
obstacle corner when `corridor_width = MIN_CORRIDOR`).

---

## 4-Axis Curriculum

Progress tracked via `env.common_step_counter` (Isaac Lab built-in, nav frequency).
Ramp over 24,576,000 steps = 2000 iters × 24 rollouts × 512 envs.

| Progress | Corridor | Obstacles | Ctrl pts | Max deviation |
|----------|----------|-----------|----------|---------------|
| 0%       | 2.00 m   | 3         | 2        | 1.0 m         |
| 25%      | 1.63 m   | 5         | 2        | 1.5 m         |
| 50%      | 1.26 m   | 7         | 3        | 2.0 m         |
| 75%      | 0.88 m   | 10        | 4        | 2.5 m         |
| 100%     | 0.51 m   | 12        | 5        | 3.0 m         |

Axis semantics:
- **corridor_width** — how tightly the robot must follow the path
- **num_obstacles** — scene density
- **num_control_points** — path winding complexity
- **max_lateral_deviation** — how far the path wanders from the straight line

---

## Reward Weights (inherited from CP6)

| Term                | Weight | Role                                    |
|---------------------|--------|-----------------------------------------|
| `navigation_core`   | +10.0  | Product of forward × lateral × heading  |
| `path_following`    | +0.5   | Closeness to A*-free spline waypoints   |
| `goal_proximity`    | +0.1   | Continuous pull toward goal             |
| `goal_reached`      | +50.0  | Sparse bonus at ≤0.1 m from goal        |
| `slow_near_goal`    | +3.0   | Decelerate to dock, not rush            |
| `graduated_clearance`| −0.05 | Light penalty for close-to-obstacle nav |
| `collision`         | −1.5   | Contact force on base                   |
| `stuck`             | −0.3   | Velocity < 0.1 m/s for > 3 s           |
| `smoothness`        | −1.0   | 2nd-order jerk penalty                  |

Obs: 1615-dim = **40×40 occupancy grid** (1600 cells) + 9 waypoint scalars +
3 velocity + 3 gravity.

---

## Files Created / Modified

| File | Change |
|------|--------|
| `scene/scene_generator.py` | `generate_scene()`, `apply_scene_to_env()`, Go2 C-space constants |
| `scene/curriculum.py`      | `NavigationCurriculum` — 4-axis linear ramp |
| `scene/__init__.py`         | Package exports |
| `mdp/events.py`             | `cp65_reset_with_generated_scene()` + rate-limited log |
| `mdp/__init__.py`           | Exports `cp65_reset_with_generated_scene` |
| `config/go2/navigation_env_cfg.py` | `NeuroGaitNavigationCP65EnvCfg`, `_PLAY` |
| `config/go2/__init__.py`    | Registers `NeuroGait-Navigation-CP65-v0`, `CP65-Play-v0` |
| `planning/planner.py`       | Removed 3 A* print statements |
| `planning/global_grid.py`   | Removed 2 build/completion prints |
| `mdp/observations.py`       | Removed waypoint-state-init and fallback prints |
| `scripts/cp6/visualize_generated_scene.py` | 6-panel difficulty plot |

---

## Training Command

```bash
python scripts/train.py --task NeuroGait-Navigation-CP65-v0 --num_envs 512
```

The task name auto-detects the log directory:
`logs/skrl/neurogait_navigation_cp65_v0/YYYY-MM-DD_HH-MM-SS/`

---

## What Was Fixed vs CP6

1. **Print spam** — all 6 A*/planner/obs prints silenced; training terminal is clean
2. **No A* call at reset** — replanning was O(n²) per reset × 512 envs; now O(n) CPU geometry
3. **Guaranteed traversability** — A* could fail and fall back to straight-line; generator never fails
4. **Progressive difficulty** — CP6 had fixed obstacle count and layout; CP6.5 scales with training progress
