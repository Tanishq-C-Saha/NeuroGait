"""Proportional heading controller that tracks A* waypoints sequentially."""

import math


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class WaypointController:
    """
    Tracks A* waypoints one by one using a proportional heading error.
    Outputs (vx, vy, yaw_rate) clipped to the frozen policy training range:
        vx  ∈ [0, 0.8]   (sub-limit of training ±1.0)
        vy  = 0.0
        yaw ∈ [-0.9, 0.9] (sub-limit of training ±1.0)
    """

    WAYPOINT_RADIUS = 0.3   # advance when within this distance of current waypoint
    KP_YAW         = 1.2    # proportional gain on heading error
    VX_MAX         = 0.8    # m/s forward speed cap
    YAW_MAX         = 0.9   # rad/s yaw cap

    def __init__(self, planner):
        self.planner = planner
        self.current_waypoint_idx = 0

    def reset(self):
        self.current_waypoint_idx = 0

    def step(self, robot_xy, robot_yaw):
        """
        Args:
            robot_xy  : (x, y) world position
            robot_yaw : heading in radians

        Returns:
            (vx, vy, yaw_rate) command tuple.  Returns (0, 0, 0) when goal reached.
        """
        target = self.planner.get_waypoint_world(self.current_waypoint_idx)
        if target is None:
            return (0.0, 0.0, 0.0)

        dx = target[0] - robot_xy[0]
        dy = target[1] - robot_xy[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # advance to next waypoint when close enough
        if dist < self.WAYPOINT_RADIUS:
            total = len(self.planner.path_world)
            print(
                f"[CP4] Advanced to waypoint "
                f"{self.current_waypoint_idx}/{total}, dist={dist:.2f}m"
            )
            self.current_waypoint_idx += 1
            next_target = self.planner.get_waypoint_world(self.current_waypoint_idx)
            if next_target is None:
                return (0.0, 0.0, 0.0)
            # recompute for new target (no infinite recursion)
            dx = next_target[0] - robot_xy[0]
            dy = next_target[1] - robot_xy[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.WAYPOINT_RADIUS:
                return (0.0, 0.0, 0.0)

        heading_to_target = math.atan2(dy, dx)
        heading_error = wrap_to_pi(heading_to_target - robot_yaw)

        yaw_rate = max(-self.YAW_MAX, min(self.YAW_MAX, self.KP_YAW * heading_error))
        vx = self.VX_MAX * max(0.0, 1.0 - abs(heading_error) / (math.pi / 2))

        return (vx, 0.0, yaw_rate)
