"""A* pathfinding on a 2D numpy occupancy grid."""

import heapq
import numpy as np


def astar(start, goal, grid):
    """
    A* pathfinding on a 2D numpy grid.

    Args:
        start : (row, col) start cell
        goal  : (row, col) goal cell
        grid  : np.ndarray, 0=free, nonzero=obstacle

    Returns:
        list of (row, col) from start to goal inclusive, or None if unreachable.

    Uses 4-directional movement and Manhattan distance heuristic.
    """
    rows, cols = grid.shape

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    sr, sc = start
    gr, gc = goal

    if not in_bounds(sr, sc) or not in_bounds(gr, gc):
        return None
    if grid[sr, sc] != 0 or grid[gr, gc] != 0:
        return None
    if start == goal:
        return [start]

    def heuristic(r, c):
        return abs(r - gr) + abs(c - gc)

    # heap: (f, g, row, col)
    open_heap = [(heuristic(sr, sc), 0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0}

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)

        if (r, c) == (gr, gc):
            # reconstruct path
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sr, sc))
            path.reverse()
            return path

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or grid[nr, nc] != 0:
                continue
            new_g = g + 1
            if new_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = new_g
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (new_g + heuristic(nr, nc), new_g, nr, nc))

    return None
