# CP4: Rule-Based Pipeline Plumbing

## Status
- [x] planning/global_grid.py — env.scene.rigid_objects → 2D occupancy grid
- [x] planning/astar.py — A* (4-dir, Manhattan heuristic)
- [x] planning/planner.py — AStarPlanner world↔grid wrapper
- [x] control/waypoint_controller.py — proportional heading controller
- [x] scripts/play_cp4.py — end-to-end A→B navigation demo

## Passing bar
- [ ] Robot navigates A→B through obstacle scene without falling
- [ ] Waypoint index printed to console, advances correctly
- [ ] Robot stops when goal reached (within 0.3 m)
- [ ] No shape mismatch errors (policy obs group stays 235-dim)
- [ ] Screen recording captured

## Run command
```bash
~/isaac-sim/kit/python/bin/python3 scripts/play_cp4.py \
  --task NeuroGait-Navigation-Unitree-Go2-Play-v1 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt \
  --enable_cameras \
  --goal_x 8.0 --goal_y 0.0
```

## What was built

**planning/global_grid.py** — Iterates `env.scene.rigid_objects` and rasterises every
entry whose name contains "obstacle", "cube", or "cyl" into a 200×200 grid (0.2 m/cell
→ 40 m × 40 m). Footprint radius comes from the spawn config: `CuboidCfg.size` gives
`max(sx, sy)/2`, `CylinderCfg.radius` is used directly. An inflation ring of 0.2 m
(one cell) gives the 0.3 m wide Go2 body clearance. World coord convention: col ↔ X,
row ↔ Y (row 0 = min Y).

**planning/astar.py** — Fresh A* (does NOT import from concept/). 4-directional
movement, Manhattan heuristic, heapq priority queue. Returns None if start/goal is
inside an obstacle or out of bounds.

**planning/planner.py** — `AStarPlanner` wraps world↔grid coordinate conversion and
A* into a clean interface. Downsample the raw A* path: keep every 5th cell plus the
final goal to reduce waypoint count without losing path shape.

**control/waypoint_controller.py** — `WaypointController` runs a proportional heading
error controller. Algorithm:
1. Compute direction vector to current waypoint
2. `heading_error = wrap_to_pi(atan2(dy, dx) - robot_yaw)`
3. `yaw_rate = clip(1.2 × heading_error, ±0.9)`
4. `vx = 0.8 × max(0, 1 - |heading_error| / (π/2))` — slows when turning
5. Advance waypoint index when within 0.3 m

**scripts/play_cp4.py** — End-to-end script. Follows the exact AppLauncher →
`@hydra_task_config` → `RslRlVecEnvWrapper` → `runner.load` pattern from
`scripts/rsl_rl/play.py`. Key CP4 addition: before each policy forward pass, the
command term's internal tensor is overridden:
```python
vel_term = env.unwrapped.command_manager.get_term("base_velocity")
vel_term.vel_command_b[0, 0] = vx
vel_term.vel_command_b[0, 1] = vy
vel_term.vel_command_b[0, 2] = yaw_rate
obs = env.get_observations()   # recomputes with overridden command
actions = policy(obs)
```
`env.step()` will randomise the command again internally, but the override at the
start of each iteration ensures the policy always sees the planned command.

## Why privileged global map (not SLAM) at this stage
The supervisor gave us a known map. Reading obstacle positions from sim state is
correct for a known-map problem. In deployment the source changes to RTAB-Map
(validated Go2 pipeline, same RealSense D455 + IMU already onboard) but the
downstream code — A*, RL policy, locomotion — is unchanged. The RL navigation
policy never sees the global map directly (only local grid + waypoint direction),
so adding RTAB-Map requires zero retraining.

## Why A* (not RL) for global planning
A* is provably optimal for shortest-path on a known grid. RL would converge
slowly toward what A* gives for free. RL belongs on the LOCAL reactive layer
(CP5+) where dynamic obstacles and real-time perception make hand-coded rules
insufficient.

## What is deliberately NOT done here
- Local camera grid (CP3) is computed in the obs group but not consumed.
  The waypoint controller drives everything. CP5 replaces the controller
  with a learned policy that reads the local grid.
- No new gym task registration. Uses `NeuroGait-Navigation-Unitree-Go2-Play-v1`.
- Frozen locomotion checkpoint untouched. 235-dim policy obs group unchanged.

## Concepts used
| Concept | One-line explanation |
|---------|---------------------|
| Global occupancy grid | 2D bitmap of obstacle footprints, world-scale, used by A* |
| A* on 2D grid | Optimal shortest-path search; f = g + h, h = Manhattan distance |
| Path downsampling | Keep every Nth waypoint to reduce command chatter |
| Proportional heading control | `yaw_rate = Kp × heading_error`; speed tapers when turning |
| Command injection | Write to `vel_term.vel_command_b` before `observation_manager.compute()` so policy obs include the planned command |
| `@hydra_task_config` | Isaac Lab decorator that loads env_cfg + agent_cfg from gym registry and passes them to main() |
| `RslRlVecEnvWrapper` | Wraps the Isaac Lab env for rsl_rl; `get_observations()` returns TensorDict; `policy(obs)` internally normalises and runs the actor |
