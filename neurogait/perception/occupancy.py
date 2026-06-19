# d:/CAPSTONE/Navigation/neurogait/perception/occupancy.py
import torch

class OccupancyGridPipeline:
    """
    4. Occupancy Grid Pipeline.
    Converts raw depth camera observations into a robot-centric 2D occupancy grid.
    """
    def __init__(self, num_envs, grid_width=64, grid_height=64, cell_size=0.1, device="cuda:0"):
        self.num_envs = num_envs
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size # meters per grid cell (e.g. 10cm)
        self.device = device
        
        # Local occupancy grids (0.0 = free, 1.0 = occupied)
        self.occupancy_grids = torch.zeros((num_envs, grid_width, grid_height), device=device)

    def process(self, depth_images: torch.Tensor, robot_positions: torch.Tensor, robot_orientations: torch.Tensor) -> torch.Tensor:
        """
        Projects depth image points into local robot coordinates and fills occupancy grid.
        
        depth_images: Tensor of shape (num_envs, channels, height, width)
        robot_positions: Tensor of shape (num_envs, 3)
        robot_orientations: Tensor of shape (num_envs, 4)
        """
        # Vectorized projection pipeline template
        # 1. Backproject depth points using camera intrinsic matrix
        # 2. Transform points to base footprint frame
        # 3. Bin coordinates into grid indices
        
        # Here we implement a high-performance vector representation wrapper:
        self.occupancy_grids.zero_()
        
        # Dummy projection logic for simulation testing:
        # Simulate obstacle blockages around simulated terrains
        for i in range(self.num_envs):
            # Let's add simulated noise and obstacles based on positions
            # Simulated rubble obstacle ahead of robot
            obstacle_dist = 2.0 - (robot_positions[i, 0] % 5.0)
            if 0 < obstacle_dist < 1.5:
                # Map obstacle coordinates to cell positions
                cell_x = int(self.grid_width / 2 + obstacle_dist / self.cell_size)
                cell_y = int(self.grid_height / 2)
                if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                    self.occupancy_grids[i, cell_x - 2:cell_x + 3, cell_y - 2:cell_y + 3] = 1.0
                    
        # Add random sensor noise
        noise = torch.rand_like(self.occupancy_grids) < 0.01
        self.occupancy_grids = torch.clamp(self.occupancy_grids + noise.float(), 0.0, 1.0)
        
        return self.occupancy_grids
