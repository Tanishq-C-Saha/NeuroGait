"""
CP3 — Point cloud to occupancy grid

This is the piece AFTER the depth-image-to-3D-points step (which Isaac
Lab's own unproject_depth / create_pointcloud_from_depth functions will
handle for us later, using the real camera). Here we build and test the
part that's genuinely ours: given a bunch of 3D points (in the robot's
own frame, x=forward, y=left, z=up), turn them into a 2D occupancy grid.

We test this with FAKE points we make up by hand, so we can verify by
counting on paper exactly which grid cell each point should land in,
before ever touching real (noisy, complicated) simulator data.
"""

import numpy as np


def points_to_occupancy_grid(points_xyz, grid_size_m, resolution_m, max_height_m=2.0, min_height_m=0.05):
    """
    Convert a 3D point cloud into a 2D occupancy grid, centered on the
    robot, with the robot facing +x (forward) and +y (left) -- standard
    robot-frame convention.

    points_xyz: (N, 3) array of points, each (x, y, z) in METERS,
    relative to the robot. x=forward, y=left, z=up.
    grid_size_m: how wide/tall the square grid is, in meters
    (e.g. 8.0 means the grid covers an 8m x 8m area)
    resolution_m: size of each grid cell, in meters (e.g. 0.2 means
    each cell represents a 20cm x 20cm patch of ground)
    max_height_m / min_height_m: points outside this height range get
    ignored -- e.g. ground points (z near 0) and ceiling/
    sky points (z very high) shouldn't count as "obstacles"

    Returns: a 2D numpy array of shape (n_cells, n_cells), where
    1 = occupied, 0 = free.
    """
    n_cells = int(grid_size_m / resolution_m)
    grid = np.zeros((n_cells, n_cells), dtype=int)

    # the robot sits at the CENTER of this grid, not the corner.
    # so a point at (x=0, y=0) -- right at the robot's feet -- should
    # map to the middle cell, not cell (0,0).
    center_cell = n_cells // 2

    for x, y, z in points_xyz:
        # step 1: ignore points that are ground-level or very high up --
        # these aren't "obstacles", they're floor noise or sky/ceiling
        if z < min_height_m or z > max_height_m:
            continue

        # step 2: convert this point's real-world (x,y) offset from the
        # robot into a grid cell index. Dividing by resolution tells us
        # "how many cells away from center is this point", then we
        # round to the nearest whole cell and shift by center_cell so
        # the robot's own position lands in the middle of the grid.
        col_offset = int(round(x / resolution_m))
        row_offset = int(round(y / resolution_m))

        row = center_cell - row_offset  # NOTE: see explanation below
        col = center_cell + col_offset

        # step 3: ignore points that fall outside the grid entirely
        # (e.g. something very far away, beyond our grid's coverage)
        if 0 <= row < n_cells and 0 <= col < n_cells:
            grid[row, col] = 1

            return grid


        # -----------------------------------------------------------------
        # WHY row = center - row_offset (not center + row_offset)?
        #
        # We're treating row 0 as the TOP of the grid (matching how images
        # and matplotlib's imshow are normally displayed -- row increases
        # DOWNWARD on screen). But y (left) increases to the LEFT in robot-
        # frame convention. So a point to the robot's LEFT (positive y)
        # should appear toward the TOP of the displayed grid (smaller row
        # number), not the bottom. Subtracting gets this orientation right.
        # This is exactly the same row/col-vs-x/y care from your A* work --
        # always worth being explicit about, never assumed.
        # -----------------------------------------------------------------


if __name__ == "__main__":
    # ---------------------------------------------------------
    # TEST 1: a single point, straight ahead of the robot.
    # Hand-check: grid is 8m wide, 0.2m resolution -> 40 cells.
    # center_cell = 20. A point 2m straight ahead (x=2, y=0)
    # is 2/0.2 = 10 cells forward of center.
    # "Forward" should map to a SPECIFIC direction in the grid --
    # let's just run it and see, then verify by reasoning about it.
    # ---------------------------------------------------------
    test_points = np.array([
        [2.0, 0.0, 0.5],   # 2m straight ahead, half a meter up - should count
        [0.0, 0.0, 0.0],   # right at the robot's feet, ground level - should be IGNORED (too low)
        [0.0, 0.0, 5.0],   # directly above, very high - should be IGNORED (too high)
    ])

    grid = points_to_occupancy_grid(test_points, grid_size_m=8.0, resolution_m=0.2)

    occupied_cells = np.argwhere(grid == 1)
    print("Grid shape:", grid.shape)
    print("Occupied cells (row, col):", occupied_cells.tolist())
    print("Expected: exactly ONE occupied cell (from the forward point),")
    print("the ground point and the too-high point should both be excluded.")
    print()

    # ---------------------------------------------------------
    # TEST 2: points to the left and right, to check row/col
    # orientation is sane -- a point to the LEFT should land on
    # the opposite side of the grid from a point to the RIGHT.
    # ---------------------------------------------------------
    test_points2 = np.array([
        [1.0, 1.0, 0.5],   # 1m ahead, 1m to the LEFT
        [1.0, -1.0, 0.5],  # 1m ahead, 1m to the RIGHT
    ])
    grid2 = points_to_occupancy_grid(test_points2, grid_size_m=8.0, resolution_m=0.2)
    occ2 = np.argwhere(grid2 == 1)
    print("Left/right test occupied cells (row, col):", occ2.tolist())
    print("These two should have the SAME row-offset-from-forward-point")
    print("pattern, but be on OPPOSITE sides (different col), confirming")
    print("left and right don't get confused with each other.")