# d:/CAPSTONE/Navigation/neurogait/communication/memory.py
import torch

class SharedObstacleMemory:
    """
    Phase 3: Shared Obstacle Memory.
    A centralized/distributed key-value store that fuses local occupancy grids 
    from multiple robots (R1, R2) into a unified coordinate map.
    """
    def __init__(self, map_size=(100, 100), cell_resolution=0.1, device="cuda:0"):
        self.device = device
        self.map_size = map_size
        self.resolution = cell_resolution
        
        # Central coordinate grid (0.0 = clear, 1.0 = blocked)
        self.global_grid = torch.zeros(map_size, device=device)
        self.confidence_grid = torch.zeros(map_size, device=device)

    def register_obstacle(self, robot_id: int, relative_obstacle_points: torch.Tensor, robot_position: torch.Tensor):
        """
        Transforms relative point coordinates from a specific robot to the global frame
        and updates global_grid with confidence weights.
        """
        if relative_obstacle_points.shape[0] == 0:
            return
            
        # Convert to global frame coordinates
        global_points = relative_obstacle_points + robot_position[:2].view(1, 2)
        
        # Map to grid indices
        idx_x = (global_points[:, 0] / self.resolution).long() + int(self.map_size[0] / 2)
        idx_y = (global_points[:, 1] / self.resolution).long() + int(self.map_size[1] / 2)
        
        # Filter out of bounds points
        mask = (idx_x >= 0) & (idx_x < self.map_size[0]) & (idx_y >= 0) & (idx_y < self.map_size[1])
        idx_x = idx_x[mask]
        idx_y = idx_y[mask]
        
        # Bayesian update or simple accumulation
        self.global_grid[idx_x, idx_y] = 1.0
        self.confidence_grid[idx_x, idx_y] += 1.0

    def query_local_map(self, robot_position: torch.Tensor, local_width=64, local_height=64) -> torch.Tensor:
        """
        Crops global grid around robot position to return a localized grid.
        """
        grid_pos_x = int(robot_position[0] / self.resolution) + int(self.map_size[0] / 2)
        grid_pos_y = int(robot_position[1] / self.resolution) + int(self.map_size[1] / 2)
        
        # Crop boundaries
        start_x = max(0, grid_pos_x - int(local_width / 2))
        end_x = min(self.map_size[0], grid_pos_x + int(local_width / 2))
        start_y = max(0, grid_pos_y - int(local_height / 2))
        end_y = min(self.map_size[1], grid_pos_y + int(local_height / 2))
        
        local_crop = torch.zeros((local_width, local_height), device=self.device)
        
        # Write cropped segment
        slice_w = end_x - start_x
        slice_h = end_y - start_y
        
        if slice_w > 0 and slice_h > 0:
            local_crop[:slice_w, :slice_h] = self.global_grid[start_x:end_x, start_y:end_y]
            
        return local_crop
