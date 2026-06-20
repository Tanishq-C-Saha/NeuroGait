# CP3 — Depth Camera + Occupancy Grid

## Status
- [x] verify_cp3.py passes standalone
- [ ] observations.py loads in Isaac Lab without import error
- [ ] occupancy_grid_obs returns correct shape (num_envs, 1600)
- [ ] no NaN values on first 100 steps
- [ ] visual check: grid shows occupied cells where obstacles are

## What Was Built

**concept/verify_cp3.py** — Standalone math proof (zero Isaac Lab dependency). Creates a fake 2 m depth image, unprojets it via the pinhole formula, rotates to robot frame, builds a 40×40 occupancy grid, and asserts the expected cells are occupied. All 4 assertions pass.

**concept/occupancy_grid.py** — Fixed a `return grid` indentation bug (it was inside the for-loop, returning after the first point). Replaced the Python for-loop with vectorised NumPy ops (height filter → cell-index computation → bounds mask → fancy indexing). Result: **~4.5 ms per call at 307 K points** (full 640×480 frame), ~100× faster than the for-loop version.

**mdp/observations.py** — Isaac Lab observation term. Implements `occupancy_grid_obs` (numpy path, safe for single-env play) and `occupancy_grid_obs_gpu` (pure-torch GPU path, suitable for multi-env training). Both include a NaN guard for the known first-step camera pose bug.

**mdp/commands.py** — Minimal `NullCommandCfg` dataclass stub. CP3 still uses fixed velocity commands from the locomotion base class; this file makes the `mdp` package importable and provides a hook for future CP4+ planner-backed commands.

**mdp/__init__.py** — Exports `occupancy_grid_obs`, `occupancy_grid_obs_gpu`, `NullCommandCfg`.

## Files Changed
| File | What changed |
|------|-------------|
| `concept/occupancy_grid.py` | Fixed return-inside-loop bug; replaced for-loop with vectorised NumPy |
| `concept/verify_cp3.py` | **New** — standalone math verification script |
| `mdp/__init__.py` | **New** — package exports |
| `mdp/observations.py` | **New** — `occupancy_grid_obs` + `occupancy_grid_obs_gpu` terms |
| `mdp/commands.py` | **New** — `NullCommandCfg` stub |

## Mini-Checkpoints Achieved
1. `occupancy_grid.py` bug fixed and vectorised — existing `__main__` tests pass
2. `verify_cp3.py` passes all 4 assertions standalone (no Isaac Lab needed)
3. `commands.py` and `occupancy_grid.py` import cleanly outside Isaac Lab
4. `observations.py` written with NaN guard, both numpy and GPU torch paths

## Concepts Used
| Concept | One-line explanation |
|---------|---------------------|
| Camera intrinsic matrix K | 3×3 matrix encoding focal lengths (fx, fy) and principal point (cx, cy); K⁻¹ maps pixel + depth to a 3D ray |
| unproject_depth | Isaac Lab utility that applies K⁻¹ · [u,v,1]ᵀ · d to every pixel, returning camera-frame 3D points |
| transform_points | Isaac Lab utility that applies a rigid transform (position + quaternion) to a batch of 3D points |
| Robot frame transform | Subtract robot base position (translation), then rotate by quat_inv(robot_quat) (orientation) to go from world axes to body axes |
| Occupancy grid binning | Map (x,y) → (col,row) via division by resolution and offset by center_cell; height filter removes ground and sky returns |

## Concepts to Understand Better (study these yourself)
- [ ] Why K⁻¹ × [u,v,1]ᵀ × d gives a 3D point (pinhole camera model)
- [ ] What quat_inv does and why we need it for world→robot transform
- [ ] Why we filter by height (min/max) in the occupancy grid function
- [ ] What NaN in pos_w / quat_w_ros means and why it happens on step 0

## References
- Isaac Lab unproject_depth API: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html
- Isaac Lab camera how-to: https://isaac-sim.github.io/IsaacLab/main/source/how-to/save_camera_output.html
- Known NaN bug: https://github.com/isaac-sim/IsaacLab/issues/3004
- Pinhole camera model math: `concept/verify_cp3.py` (inline comments, step 2)

## How to Reproduce This Yourself
1. Read `concept/occupancy_grid.py` top to bottom — understand what `center_cell`, `col_offset`, `row_offset` do
2. Draw a 40×40 grid on paper; place the robot at cell (20,20); mark where (x=2m, y=0) lands by hand
3. Run `verify_cp3.py` and match its output against your paper calculation
4. Read `mdp/observations.py` step by step: trace depth → camera pts → world pts → robot pts → grid
5. Run the full Isaac Lab play script below and check the printed shape + NaN count

## Test Commands
```bash
# standalone verification (no Isaac Lab needed)
ipy source/neurogait/neurogait/tasks/manager_based/navigation/concept/verify_cp3.py

# full Isaac Lab test (check shape + NaN)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task NeuroGait-Navigation-Rubble-Play-v0 \
  --checkpoint logs/rsl_rl/unitree_go2_rough/2026-06-13_19-33-23/model_1499.pt \
  --enable_cameras
```
