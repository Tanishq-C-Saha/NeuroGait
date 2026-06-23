# 06: Asymmetric Grid (Robot at Row 10, Not Row 20)

## The confusion
Why place the robot at row 10 in the 40×40 grid instead of the center (row 20)?
Isn't the center more balanced?

## The answer
The D455 camera has useful range 0.4–6 m forward. Placing the robot at row 10
gives 6 m forward view and 2 m behind — matching the camera's useful range.
The center placement wastes 6 m behind the robot where the camera cannot see.

## How it works

### Symmetric (robot at row 20, OLD)

```
Forward view:  (40 - 20) × 0.2 = 4.0 m   ← only 4 m forward
Backward view: 20 × 0.2         = 4.0 m   ← 4 m behind (useless, camera looks forward)
```

### Asymmetric (robot at row 10, CP5)

```
Forward view:  (40 - 10) × 0.2 = 6.0 m   ← 6 m forward (matches D455 range)
Backward view: 10 × 0.2         = 2.0 m   ← 2 m behind (useful for recovery)
Lateral:       20 × 0.2         = 4.0 m   ← 4 m each side (unchanged)
```

### Grid dimensions remain 40×40

Same obs size (1600 dims). Same CNN architecture. The only change is one integer:
`robot_row = 10` instead of `robot_row = grid_size // 2`.

### Why 2 m behind is still useful

During obstacle avoidance the robot sometimes reverses briefly. A 2 m backward view
lets the grid show nearby objects behind the robot, preventing it from backing into walls.
Zero backward view would leave `vy < 0` maneuvers completely blind.

## What breaks if you get this wrong

Using the symmetric grid wastes ~33% of the grid's forward capacity.
At 1 m/s, the 4 m forward view covers only 4 seconds of lookahead — too short for
the robot to plan around large obstacles at typical navigation speeds.

With the asymmetric grid, 6 m forward at 1 m/s = 6 seconds of lookahead, which is
sufficient to see all obstacles in the CP4 course from the robot's initial position.

## Code reference

- `mdp/observations.py` — `occupancy_grid_obs_cp5()` line: `robot_row = 10`
- Grid constant: `_N_CELLS = 40`, `_RESOLUTION_M = 0.2`, `_GRID_SIZE_M = 8.0`
- Old symmetric version: `occupancy_grid_obs()` still uses `robot_row = grid_size // 2 = 20`
