"""AStarPlanner: world-coordinate wrapper around the A* grid search."""

from .astar import astar
from .global_grid import world_to_grid, grid_to_world


class AStarPlanner:
    def __init__(self, grid, origin, resolution=0.2):
        self.grid = grid
        self.origin = origin
        self.resolution = resolution
        self.path_world = []

    def plan(self, start_world_xy, goal_world_xy):
        """
        Convert world→grid, run A*, convert path back to world coords.
        Keeps every 5th waypoint + always the final goal.

        Returns:
            list of (x, y) world waypoints, or [] if no path.
        """
        start_rc = world_to_grid(start_world_xy, self.origin, self.resolution)
        goal_rc  = world_to_grid(goal_world_xy,  self.origin, self.resolution)

        print(f"[CP4] Planning path from {start_world_xy} to {goal_world_xy}...")
        path_grid = astar(start_rc, goal_rc, self.grid)

        if path_grid is None:
            print("[CP4] No path found")
            self.path_world = []
            return []

        # downsample: keep every 5th cell + always include the goal
        downsampled = path_grid[::5]
        if downsampled[-1] != path_grid[-1]:
            downsampled = list(downsampled) + [path_grid[-1]]

        self.path_world = [
            grid_to_world(rc, self.origin, self.resolution) for rc in downsampled
        ]
        print(f"[CP4] Path found: {len(self.path_world)} waypoints")
        return self.path_world

    def get_waypoint_world(self, idx):
        """Return self.path_world[idx], or None if out of range."""
        if idx < len(self.path_world):
            return self.path_world[idx]
        return None
