# 07: Pure Pursuit — 3 Future Waypoints

## The confusion
Why give the navigation policy 3 future waypoints (9 dims) instead of just the
current target waypoint (3 dims)?

## The answer
With only the current waypoint, the policy sees a turn only when it arrives at
the waypoint — too late to start turning. With 3 lookahead waypoints, the policy
sees an upcoming right turn while still approaching the current waypoint and can
begin curving early. This is the pure pursuit principle.

## How it works

### Single waypoint (naive)

```
Path: A → B → C (sharp right turn at B)

Step 1: robot at A, waypoint=B → vx=1.0, vy=0.0   (go straight)
Step 2: robot arrives at B, waypoint=C → sudden turn right
         → jerky maneuver, possibly overshoots B
```

### 3 Waypoints (pure pursuit)

```
Path: A → B → C

Step 1: robot at A
   obs: [wp_B direction, wp_C direction, wp_D direction]   (3 lookahead waypoints)
   → policy sees the upcoming right turn at B/C
   → starts curving slightly right early
   → smooth arc through B instead of sharp corner
```

### Observation encoding

Each of the 3 waypoints is encoded as `[dir_x, dir_y, norm_distance]` in the robot frame:

```python
wp_obs = []
for i in range(3):
    wp_idx = min(curr_wp_idx + i, len(waypoints) - 1)
    wp = waypoints[wp_idx]
    delta = wp - robot_pos  # world frame
    # rotate to robot frame
    cos_yaw, sin_yaw = cos(-robot_yaw), sin(-robot_yaw)
    dx_robot =  cos_yaw * delta_x + sin_yaw * delta_y
    dy_robot = -sin_yaw * delta_x + cos_yaw * delta_y
    dist = sqrt(dx_robot² + dy_robot²)
    wp_obs += [dx_robot / (dist + eps), dy_robot / (dist + eps), min(dist / 10, 1.0)]
# shape: (9,)
```

Total: 3 waypoints × 3 values = 9 dims

### Why 3 waypoints (not 15 or more)

NavRL++ (Xu et al., 2026) uses 15 waypoints for UAV navigation at 5 m/s.
For Go2 at ~1 m/s with 1 m waypoint spacing, 3 waypoints gives 3 m of lookahead
= 3 seconds. That is sufficient to anticipate all turns in the CP4 obstacle course.
Adding more waypoints increases obs size without benefit for this specific scenario.

## What breaks if you get this wrong

With a single waypoint, the policy must make all turns reactively at the waypoint
location. This creates zigzag paths and poor performance in dense obstacle fields
where waypoints are close together. The policy may learn to slow down at every
waypoint to have time to turn — reducing average navigation speed.

## Code reference

- `mdp/observations.py` — `future_waypoints_obs()` — returns (num_envs, 9)
- Waypoint indices: `[curr_wp_idx, curr_wp_idx+1, curr_wp_idx+2]` clamped to len(path)-1
- Obs order: indices [1600:1609] (after occupancy grid) — see `06_asymmetric_grid.md`
