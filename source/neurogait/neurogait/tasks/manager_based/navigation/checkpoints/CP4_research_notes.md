# CP4 Research Notes: Navigation Safety, C-Space Inflation & 8-Connected A*

## The Point-Robot Problem

Every basic A* tutorial treats the robot as a dimensionless point. Real robots are rigid bodies. If A* routes the robot centre 0.05 m from an obstacle edge, but the robot body is 0.31 m wide, the robot physically collides — even when A* reports the path as collision-free.

This is the single most common mistake in textbook-to-robot transfers, and it is exactly what was happening in CP4 before this fix.

---

## 1  Configuration Space (C-Space) Obstacle Inflation

### Concept (LaValle, *Planning Algorithms*, 2006, Ch. 4)

The idea: instead of planning with a rigid-body robot in *workspace*, inflate every obstacle by the robot's size so you can plan with a **point robot** in **configuration space (C-space)**.

Mathematically, the inflated obstacle set C\_obs is the **Minkowski sum** of the original obstacle set W\_obs and the robot's body shape B (negated):

```
C_obs = W_obs ⊕ (-B)
```

For a convex robot body approximated as a circle of radius R\_robot:

```
C_obs = { p | dist(p, obstacle_edge) < R_robot }
```

After this inflation:
- Plan for a point at position p\_robot
- If the point's path avoids C\_obs, the physical robot body avoids W\_obs

### Why this matters for Go2

| Quantity | Value |
|---|---|
| Go2 body length | ~0.67 m |
| Go2 body width | ~0.31 m |
| Half-diagonal (worst case) | sqrt(0.335² + 0.155²) ≈ **0.37 m** |
| Grid resolution | 0.20 m / cell |

**Before fix** — `_INFLATION_M = 0.20 m`:
```
clearance from obstacle edge = 0.20 m
robot half-width             = 0.37 m (diagonal worst case)
body clearance               = 0.20 - 0.37 = -0.17 m  ← COLLISION
```
The robot's body was *guaranteed* to collide even on the "safe" planned path.

**After fix** — `_INFLATION_M = 0.37 + 0.13 = 0.50 m`:
```
clearance from obstacle edge = 0.50 m
robot half-width             = 0.37 m
safety margin                = 0.13 m  ← robot body clears by 0.13 m
```

---

## 2  ROS Navigation Stack (move\_base / costmap2D) — Industry Reference

The ROS navigation stack (Marder-Eppstein et al., ICRA 2010) implements exactly this concept through `costmap2d`:

```yaml
# costmap_common_params.yaml  (standard robot deployment config)
robot_radius: 0.30           # physical body radius (or footprint polygon)
inflation_radius: 0.55       # additional safety buffer — total clearance = 0.85 m
cost_scaling_factor: 10      # exponential cost decay away from obstacles
```

Key distinction:
- **`robot_radius`** — physical body extent (mandatory safety zone)
- **`inflation_radius`** — extra buffer beyond body (adjustable for speed vs safety)

Cells within `robot_radius` of an obstacle → hard-occupied (never plan through).  
Cells within `inflation_radius` → elevated cost (planner avoids but can enter if necessary).

Our implementation simplifies to binary (0/1) with a single inflation value covering both concepts. This is correct and sufficient for a known-map global planner.

### Typical values for real deployments

| Robot | Body width | Inflation used | Grid resolution |
|---|---|---|---|
| TurtleBot3 (wheeled) | 0.28 m | 0.30 m | 0.05 m/cell |
| Clearpath Husky | 0.67 m | 0.60 m | 0.05 m/cell |
| ANYmal C (ETH Zurich) | 0.58 m | 0.50 m | 0.10 m/cell |
| **Unitree Go2 (ours)** | **0.31 m** | **0.50 m** | **0.20 m/cell** |

---

## 3  Legged Robot Navigation Papers

### ANYmal — ETH Zurich (Miki et al., *Science Robotics* 2022)

"Learning robust perceptive locomotion for quadrupedal robots in the wild" uses a **two-layer hierarchy**:
1. **Global planner** — A\* on elevation-map occupancy grid with body-clearance inflation (their term: "traversability map"). Resolution: 0.1 m/cell, inflation ≥ 0.3 m (ANYmal body half-width).
2. **Local planner / RL controller** — handles dynamic obstacles and uneven terrain in real time. The global path is used as a *reference direction*, not a rigid trajectory.

Key insight: the global A\* path tells the robot WHERE to go; the local RL policy decides HOW to step. This is exactly the CP4→CP5 separation in NeuroGait.

### Go2 ROS2 navigation (Unitree official stack)

Unitree's ROS2 nav stack for Go2 uses:
- `nav2` with `inflation_layer` plugin
- `inflation_radius = 0.55 m` (generous buffer; Go2 is 0.31 m wide)
- `robot_radius = 0.35 m` (slightly larger than physical to account for foot swing)
- A\* global planner (`GridBased` plugin) with 8-connected movement

This confirms our 0.50 m inflation is in the right range.

### RSL-RL Locomotion Baseline (Rudin et al., CoRL 2022)

"Learning to Walk in Minutes" — the locomotion policy we are using as the frozen backbone. The policy was trained on rough terrain with velocity commands as interface. Navigation-level planners are explicitly **out of scope** for this policy. The correct decomposition (which we implement) is:

```
Global A* planner → velocity commands → RL locomotion policy → joint torques
```

---

## 4  8-Connected A\* with Proper Diagonal Costs

### Why 8-connected over 4-connected

| Property | 4-connected | 8-connected |
|---|---|---|
| Moves per cell | 4 (N/S/E/W) | 8 (N/S/E/W + NE/NW/SE/SW) |
| Path style | Manhattan "staircase" | Smooth diagonal |
| Physical robot preference | Waypoints are axis-aligned only | More natural heading transitions |
| Heuristic | Manhattan distance | Octile distance |

For a physical robot tracked by a proportional heading controller, 8-connected A\* produces fewer heading changes and shorter total path length.

### Correct costs (CRITICAL — common mistake)

```python
# WRONG: treating diagonal as cost 1 (same as cardinal)
# → diagonals are "too cheap", path oscillates at 45°

# CORRECT:
cardinal_cost  = 1.0
diagonal_cost  = sqrt(2) ≈ 1.414   # actual Euclidean distance in cell units
```

If diagonal cost is set to 1 instead of √2, A\* produces paths that prefer diagonal moves even when orthogonal moves would give a shorter real-world distance. The path looks "diagonal-happy" and does not reflect true shortest-path behaviour.

### Admissible heuristic for 8-connected A\*

For 8-connected movement with diagonal cost √2, the tightest admissible and consistent heuristic is the **octile distance**:

```
h(cell) = max(|dr|, |dc|) + (sqrt(2) - 1) * min(|dr|, |dc|)
```

This is tighter than pure Euclidean distance and avoids over-estimating, so A\* remains optimal.

### Our implementation

```python
# astar.py
_MOVES = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),       # cardinal: cost 1
    (-1,-1, √2), (-1, 1, √2), (1,-1, √2), (1, 1, √2),            # diagonal: cost √2
]

def heuristic(r, c):
    dr, dc = abs(r - gr), abs(c - gc)
    return max(dr, dc) + (√2 - 1) * min(dr, dc)   # octile distance
```

---

## 5  What We Changed in CP4 and Why

| Component | Before | After | Reason |
|---|---|---|---|
| `_INFLATION_M` | 0.20 m | **0.50 m** | Robot body 0.37 m half-extent + 0.13 m safety |
| A\* connectivity | 4-connected | **8-connected** | Smoother paths, fewer heading changes |
| A\* costs | 1.0 for all moves | **1.0 cardinal / √2 diagonal** | Correct Euclidean path length |
| Heuristic | Manhattan distance | **Octile distance** | Tightest admissible h for 8-connected |
| Map PNG | Dark funky theme | **B&W occupancy grid** | Standard robotics convention (white=free, black=occupied) |

---

## 6  Path Downsampling and Smoothing

Raw A\* returns a cell-by-cell path — potentially hundreds of points for an 8 m run. Feeding every cell as a waypoint to the heading controller causes:
- Rapid (choppy) waypoint advances every 0.2 m
- No benefit since controller already does proportional approach

Our fix: keep every **5th** cell + always include the final goal. This gives ~10–15 waypoints for a typical run, enough to define the shape of the path without chattiness.

Further smoothing (not implemented here, belongs in CP5+):
- **Shortcut pruning**: remove waypoints where the robot can see the next-next waypoint in a straight line without obstacle intersection
- **Bezier smoothing**: fit a smooth curve through downsampled waypoints
- **Path re-planning on disturbance**: re-run A\* if robot deviates > 0.5 m from path

---

## 7  Concepts Learned

| Concept | One-line definition |
|---|---|
| C-space obstacle | Obstacle inflated by robot body radius; robot centre plans as point |
| Minkowski sum | Formal operation defining C-space from workspace obstacles + robot shape |
| Inflation radius | Buffer zone in C-space beyond robot physical body; tunable safety margin |
| 8-connected A\* | A\* allowing diagonal moves with cost √2; produces shorter smoother paths |
| Octile distance | Tightest admissible heuristic for 8-connected grids |
| Costmap2d | ROS layer implementing C-space inflation with binary + gradient cost zones |
| Path downsampling | Keep every Nth waypoint to reduce waypoint chatter without losing shape |
| Two-layer hierarchy | Global planner (A\*) provides direction; local RL policy handles execution |

---

## 8  References

1. Hart, Nilsson, Raphael (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*. — Original A\* paper.
2. LaValle (2006). *Planning Algorithms*. Cambridge University Press. Ch. 4 (C-space, Minkowski sum). — http://planning.cs.uiuc.edu/
3. Thrun, Burgard, Fox (2005). *Probabilistic Robotics*. MIT Press. Ch. 5, 9. — Occupancy grids, robot motion.
4. Marder-Eppstein et al. (2010). "The Office Marathon: Robust Navigation in an Indoor Office Environment." *ICRA*. — Origin of ROS move\_base / costmap2d.
5. Miki et al. (2022). "Learning robust perceptive locomotion for quadrupedal robots in the wild." *Science Robotics*. — ANYmal two-layer navigation (global A\* + local RL).
6. Rudin et al. (2022). "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning." *CoRL*. — RSL-RL locomotion baseline (our frozen policy backbone).
7. Kumar, Michael, Hauskrecht (2022). *Unitree Go2 Navigation Stack* (GitHub: unitreerobotics/unitree\_ros2). — Go2-specific nav2 config with inflation values.
