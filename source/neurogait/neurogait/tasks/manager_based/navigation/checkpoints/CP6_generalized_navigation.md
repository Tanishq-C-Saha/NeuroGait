# CP6 — Generalized Navigation: Learning Receipt

**Date:** 2026-06-28
**Status:** Implementation complete, awaiting first training run
**Builds on:** CP5 (trained navigation policy, single fixed obstacle layout)

---

## What changed from CP5

| Component | CP5 | CP6 |
|-----------|-----|-----|
| Obstacles | Fixed positions | Randomized ±1.5 m per reset |
| A* path | Built once at env launch | Replanned every episode reset |
| Reward core | Scalar velocity-toward-goal | Multiplicative (Miki 2022) |
| Path following | None | exp(−min_dist²/1.0) to waypoints |
| Collision avoidance | Binary contact penalty | Graduated clearance via depth cam |
| Goal termination | None (episode runs out) | DoneTerm at 0.5 m from goal |
| Stuck detection | All envs | Near-goal exempt (< 1.5 m) |
| Action smoothness | 1st-order jerk | 2nd-order: jerk + jerk-delta |

---

## Architecture

CP6 inherits the full CP5 stack unchanged:
- **Policy network**: `NavigationPolicy` (CNN on 100×100 occupancy grid + MLP on 15-dim scalars)
- **Locomotion backbone**: Frozen pre-trained PPO weights (Go2 locomotion, `low_level_decimation=4`)
- **Observation**: 1615-dim (100×100 grid + 15 scalars) at 5 Hz navigation rate
- **Action**: `[vx, vy, heading]` command to the locomotion policy

The pre-trained locomotion policy is NOT fine-tuned during CP6.

---

## Reward function (CP6RewardsCfg)

| Term | Weight | Purpose | Source |
|------|--------|---------|--------|
| `navigation_core` | +10.0 | r_forward × r_lateral × r_heading | Miki et al. 2022 |
| `path_following` | +5.0 | exp(−min_dist²) to nearest waypoint | NavRL++ |
| `goal_proximity` | +0.1 | Shaping toward final waypoint | Li et al. 2025 |
| `goal_reached` | +50.0 | Sparse bonus at 0.5 m radius | X-Nav 2025 |
| `slow_near_goal` | +3.0 | Reward deceleration < 1.5 m from goal | Custom |
| `graduated_clearance` | −1.0 | Depth-based danger/caution zones | DWA-3D 2024 |
| `collision` | −1.5 | Contact velocity-scaled penalty | SEA-Nav 2026 |
| `stuck` | −0.3 | Stationary detection (near-goal exempt) | SEA-Nav 2026 |
| `smoothness` | −1.0 | 2nd-order action jerk | Go2 task 2025 |

**Multiplicative core design** (key CP6 insight):
```
r_core = r_forward × r_lateral × r_heading
```
Each factor is `exp(-err²/σ²)` so ALL three must be satisfied simultaneously.
Prevents crab-walking (forward+lateral) and pure yaw-spinning (heading only).

---

## Events

### `cp6_randomize_obstacles_and_replan` (mode="reset")
- Applies a **shared** (dx, dy) offset to ALL 9 obstacles across ALL parallel envs
- Shared offset means env-0's occupancy grid is representative of every env
- Builds the grid **locally** from computed positions (avoids PhysX read-back timing)
- Retries A* up to 3 times with different random offsets
- Falls back to straight-line path if all retries fail

**Why shared, not per-env?**
Per-env randomization would require per-env A* and per-env occupancy grids (a 40×40m grid per env × 4096 envs = too much GPU memory). Shared offsets give generalization across episodes rather than across envs. Per-env diversity is deferred to CP7.

---

## Key implementation notes

### `_cp5_init_waypoint_state(env)` call ordering
`cp6_randomize_obstacles_and_replan` calls `_cp5_init_waypoint_state(env)` first.
This ensures `env._cp5_waypoints` exists before the event handler writes into it.
On the first ever reset, `_cp5_init_waypoint_state` builds the initial A* path — the
randomization event immediately overwrites it with the randomized plan. This is correct.

### Waypoint tensor size after replan
A* on different obstacle configurations returns paths of different lengths (W varies).
After replan:
```python
env._cp5_waypoints = new_wps          # (E, W_new, 2)
env._cp5_wp_idx.clamp_(max=W_new - 1) # running envs: clamp index to new length
env._cp5_wp_idx[env_ids] = 0          # resetting envs: restart from waypoint 0
```

### Graduated clearance sensor
Uses `env.scene["front_cam"]` (MultiMeshRayCasterCamera) depth buffer.
NaN values (no intersection) are replaced with 10.0 m before comparison.
Does NOT re-trigger the full observation pipeline — reads directly from `cam.data`.

### 2nd-order smoothness state
Uses `env._cp6_prev_action_1` and `env._cp6_prev_action_2` (separate from CP5's
`env._cp5_prev_action`) to avoid interfering with CP5 reward computation if CP5
rewards are ever used alongside CP6 rewards.

---

## Files added / modified

| File | Change |
|------|--------|
| `mdp/events.py` | NEW — `cp6_randomize_obstacles_and_replan` |
| `mdp/rewards.py` | Added 6 CP6 reward/penalty functions |
| `mdp/terminations.py` | Added `cp6_goal_reached` |
| `mdp/__init__.py` | Exported all new CP6 symbols |
| `config/go2/navigation_env_cfg.py` | Added `CP6RewardsCfg`, `NeuroGaitNavigationCP6EnvCfg`, `NeuroGaitNavigationCP6EnvCfg_PLAY` |
| `config/go2/__init__.py` | Registered `NeuroGait-Navigation-CP6-v0` and `NeuroGait-Navigation-CP6-Play-v0` |
| `scripts/cp6/eval_metrics.py` | NEW — N-episode evaluation with comparison table |

---

## Recommended training hyperparameters

Start from CP5's `skrl_nav_ppo_cfg.yaml` unchanged.
If `navigation_core` reward collapses in first 500k steps, reduce its weight from 10.0 → 5.0.

Expected training time: ~8–12 h on 4096 envs (same as CP5).

---

## Evaluation

```bash
~/isaac-sim/kit/python/bin/python3 scripts/cp6/eval_metrics.py \
  --task NeuroGait-Navigation-CP6-Play-v0 \
  --checkpoint logs/skrl/neurogait_cp6/<run>/checkpoints/best_agent.pt \
  --num_episodes 50
```

Target metrics after training:
- Success rate   ≥ 80%
- Path efficiency ≥ 0.70 (A* length / actual distance)
- Collisions/ep  ≤ 0.5

---

## Known limitations / CP7 scope

- Obstacle randomization is **shared** across envs — all envs see same layout per episode
- Per-env independent randomization deferred to CP7 (requires per-env A* + grid)
- Depth-camera clearance reward assumes `front_cam` is always present in scene
- No dynamic obstacles (moving humans, other robots) — CP8 scope
