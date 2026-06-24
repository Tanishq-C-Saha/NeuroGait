# d:/CAPSTONE/Navigation/neurogait/planning/planner.py
import heapq
import numpy as np
import torch

class AStarPlanner:
    """
    3. A* Planner (planner.py).
    Performs grid search pathfinding over costmaps to produce reference paths.
    """
    def __init__(self, heuristic_weight=1.0):
        self.heuristic_weight = heuristic_weight

    def plan_batch(self, start_poses: torch.Tensor, goal_poses: torch.Tensor, costmaps: torch.Tensor) -> list:
        """
        Runs planning for multiple environments.
        
        start_poses: Shape (num_envs, 3)
        goal_poses: Shape (num_envs, 3)
        costmaps: Shape (num_envs, W, H)
        """
        paths = []
        num_envs = start_poses.shape[0]
        
        # Move inputs to CPU for processing A* list-logic
        costmaps_cpu = costmaps.cpu().numpy()
        start_cpu = start_poses.cpu().numpy()
        goal_cpu = goal_poses.cpu().numpy()
        
        for i in range(num_envs):
            path = self._plan_single(start_cpu[i], goal_cpu[i], costmaps_cpu[i])
            paths.append(path)
            
        return paths

    def _plan_single(self, start, goal, costmap) -> list:
        """Runs A* on a single cost grid."""
        W, H = costmap.shape
        
        # Convert absolute metric coordinates to grid cell coordinates
        # Center of grid represents robot pose (W/2, H/2)
        start_idx = (int(W / 2), int(H / 2))
        
        # Calculate approximate grid goal index based on relative goal distance
        rel_goal_x = goal[0] - start[0]
        rel_goal_y = goal[1] - start[1]
        
        goal_idx_x = int(W / 2 + rel_goal_x * 5) # Scale factor mapping meters to cells
        goal_idx_y = int(H / 2 + rel_goal_y * 5)
        goal_idx = (max(0, min(W - 1, goal_idx_x)), max(0, min(H - 1, goal_idx_y)))

        # Priority queue: elements are (f_score, (x, y))
        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))
        
        came_from = {}
        g_score = {start_idx: 0.0}
        
        # Neighbors layout (8-way grid connectivity)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        max_iterations = 200
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == goal_idx:
                break
                
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < W and 0 <= neighbor[1] < H:
                    # Traversal cost = step distance + terrain roughness cost
                    step_cost = np.sqrt(dx**2 + dy**2)
                    terrain_cost = costmap[neighbor[0], neighbor[1]] * 10.0
                    
                    # Impassable check
                    if terrain_cost > 8.0:
                        continue
                        
                    tentative_g = g_score[current] + step_cost + terrain_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        # Heuristic: Euclidean distance to goal
                        h = self.heuristic_weight * np.sqrt((neighbor[0] - goal_idx[0])**2 + (neighbor[1] - goal_idx[1])**2)
                        f = tentative_g + h
                        came_from[neighbor] = current
                        heapq.heappush(open_set, (f, neighbor))
                        
        # Reconstruct path
        path = []
        curr = goal_idx
        
        if curr in came_from or curr == start_idx:
            while curr != start_idx:
                # Convert grid cell back to relative coordinates (relative to robot)
                rel_x = (curr[0] - W / 2) / 5.0
                rel_y = (curr[1] - H / 2) / 5.0
                path.append((rel_x + start[0], rel_y + start[1]))
                curr = came_from[curr]
            path.append((start[0], start[1]))
            path.reverse()
        else:
            # Fallback path directly targeting goal
            path = [(start[0], start[1]), (goal[0], goal[1])]
            
        return path
