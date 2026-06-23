"""A* pathfinding on a 2D numpy occupancy grid.

Uses 8-connected movement (cardinal + diagonal) with proper costs:
    cardinal step  : cost 1
    diagonal step  : cost sqrt(2) ≈ 1.414

Heuristic: octile distance (tightest admissible h for 8-connectivity).
    h = max(|dr|, |dc|) + (sqrt(2) - 1) * min(|dr|, |dc|)

8-connected A* is preferred over 4-connected for physical robots because:
  - diagonal moves produce shorter, smoother paths
  - paths look more natural when visualised and tracked by a heading controller
  - avoids the "staircase" artefact of 4-connected Manhattan paths
"""

import heapq
import math

_SQRT2 = math.sqrt(2)

# (dr, dc, cost)
_MOVES = [
    (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0, 1, 1.0),   # cardinal
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),  # diagonal
]


def astar(start, goal, grid):
    """
    A* pathfinding on a 2D numpy grid.

    Args:
        start : (row, col) start cell
        goal  : (row, col) goal cell
        grid  : np.ndarray, 0=free, nonzero=obstacle

    Returns:
        list of (row, col) from start to goal inclusive, or None if unreachable.
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
        # octile distance — tight admissible heuristic for 8-connectivity
        dr = abs(r - gr)
        dc = abs(c - gc)
        return max(dr, dc) + (_SQRT2 - 1) * min(dr, dc)

    # heap: (f, g, row, col)
    open_heap = [(heuristic(sr, sc), 0.0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)

        if (r, c) == (gr, gc):
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sr, sc))
            path.reverse()
            return path

        for dr, dc, move_cost in _MOVES:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or grid[nr, nc] != 0:
                continue
            new_g = g + move_cost
            if new_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = new_g
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (new_g + heuristic(nr, nc), new_g, nr, nc))

    return None
