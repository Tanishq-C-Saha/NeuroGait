"""
CHECKPOINT 2 — Full A* Planner

This builds directly on YOUR get_neighbors-style function from before.
Read this top to bottom. Run it. Then go back and reread the parts that
don't make sense yet — they will, once you see the output.
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

rows = 20
cols = 40

# obstacle coordinates, just for building the demo grid below —
# the planner itself never sees this list, only the finished grid array
obstacle_coords = np.array([
    (1, 1),
    (1, 2),
    (2, 1),
    (3, 3),
])

# this IS what the planner actually reads: a 2D array of values.
# 0 = free, 1 = obstacle. This is the same kind of array your real
# occupancy grid (from the depth camera, eventually) will be.
demo_grid = np.zeros((rows, cols), dtype=int)
demo_grid[obstacle_coords[:, 0], obstacle_coords[:, 1]] = 1

occupancy_grid = np.random.randint(0,2,size = (20,40))


def get_neighbors(pos, grid):
    """
    Build the list of valid neighbors by reading the grid directly.

    `grid` is a 2D array where each cell holds a value: 0 means free,
    1 (or anything non-zero) means obstacle. This is exactly the shape
    a real occupancy grid comes in — no separate obstacle list needed,
    we just look up the value that's already sitting in the array.
    """
    row, col = pos
    
    rows, cols = grid.shape  # read the size directly from the grid itself
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    neighbors = []
    for d_row, d_col in moves:
        new_row, new_col = row + d_row, col + d_col
        inside_grid = (0 <= new_row < rows) and (0 <= new_col < cols)
        if not inside_grid:
            continue
        # the actual change: check the VALUE stored in the grid,
        # instead of checking membership in a separate set
        if grid[new_row, new_col] != 0:
            continue
        neighbors.append((new_row, new_col))
    return neighbors


def heuristic(cell, goal):
    """
    The h-cost: a GUESS of how far this cell is from the goal.
    We use Manhattan distance (sum of row-difference and col-difference)
    because we can only move up/down/left/right, never diagonally —
    Manhattan distance is the exact "as the robot walks" distance on
    this kind of grid, which is why it works well as a heuristic here.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def astar(start, goal, grid):
    """
    Returns the shortest path from start to goal as a list of cells,
    e.g. [(0,0), (0,1), (1,1), ...], or None if no path exists.

    `grid` is the full 2D occupancy array — pass in any size, any
    obstacle layout. Nothing here hard-codes a particular grid.
    """

    #check if the start and goal are valid
    if (grid[start[0]][start[1]])!=0:

        #position not good
        print("Robot at invalid Position")
        return
    
    if (grid[goal[0]][goal[1]])!=0:

        #position not good
        print("Goal at invalid Position")
        return

    # g_cost[cell] = cheapest number of steps found so far to reach `cell`
    # from `start`. This answers the THIRD question from before: when we
    # find a NEW way to reach a cell we've already seen, we only update
    # g_cost if the new way is CHEAPER than what we already recorded.
    # Never overwriting would mean we might keep a worse first-found path.
    # Always overwriting would mean we might throw away a better path we
    # found earlier in favor of a worse one found later. "Only update if
    # cheaper" is the correct rule — it's what keeps A* optimal.
    g_cost = {start: 0}

    # came_from[cell] = "which cell did I step from, to reach `cell`
    # with its current best g_cost?" This is how we reconstruct the
    # final path at the end — by walking these breadcrumbs backward
    # from the goal to the start.
    came_from = {}

    # The heap holds (priority, cell) tuples. Priority = g_cost + h_cost,
    # i.e. "steps so far" + "guessed steps remaining". This is the number
    # A* uses to decide what to explore next — lower is more promising.
    open_heap = [(heuristic(start, goal), start)]

    # A set of cells we've FULLY explored already, so we don't waste time
    # re-exploring them. (Different from g_cost, which tracks cost; this
    # just tracks "have I already expanded this cell's neighbors?")
    visited = set()

    while open_heap:
        # always pull out the most promising cell — this is exactly
        # heapq.heappop from the exercise above
        current_priority, current_cell = heapq.heappop(open_heap)

        if current_cell == goal:
            # we've reached the goal — reconstruct the path by walking
            # came_from backward, then reverse it so it reads start->goal
            path = [current_cell]
            while current_cell in came_from:
                current_cell = came_from[current_cell]
                path.append(current_cell)
            path.reverse()
            return path

        if current_cell in visited:
            # we might push the same cell to the heap more than once
            # (once per neighbor that discovers it) — skip if we've
            # already fully expanded it once.
            continue
        visited.add(current_cell)

        for neighbor in get_neighbors(current_cell, grid):
            # cost to reach `neighbor` if we go through `current_cell`
            tentative_g = g_cost[current_cell] + 1  # each step costs 1

            # the "only update if cheaper" rule from above:
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current_cell
                priority = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (priority, neighbor))

    # if the heap empties out and we never returned, there's no path
    return None


# ---------------------------------------------------------------
# TEST 1 — a simple case you can verify by counting on the grid by hand
# ---------------------------------------------------------------
start = (0, 0)
goal = (4, 4)
path = astar(start, goal, occupancy_grid)
print(f"Path from {start} to {goal}:")
print(path)
if (path is not None):
    print(f"Path length: {len(path)} cells, {len(path)-1} steps")


# ---------------------------------------------------------------
# TEST 2 — visualize it, building on your own matplotlib code
# ---------------------------------------------------------------
cmap = ListedColormap(["white", "black"])
fig, ax = plt.subplots()
ax.imshow(occupancy_grid, cmap=cmap, interpolation="nearest")
ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(which="minor", color="gray", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
ax.xaxis.tick_top()

if path is not None:
    path_rows = [cell[0] for cell in path]
    path_cols = [cell[1] for cell in path]
    ax.plot(path_cols, path_rows, color="blue", linewidth=2, zorder=10)
    ax.scatter(path_cols, path_rows, color="blue", s=40, zorder=11)

ax.scatter(start[1], start[0], color="red", s=150, zorder=12, label="start")
ax.scatter(goal[1], goal[0], color="green", s=150, zorder=12, label="goal")
ax.legend()
ax.set_title("A* path", pad=30)
plt.show()
