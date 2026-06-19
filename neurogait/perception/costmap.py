# d:/CAPSTONE/Navigation/neurogait/perception/costmap.py
import torch
import torch.nn.functional as F

class CostMapPipeline:
    """
    5. Terrain Cost Map Pipeline.
    Combines the occupancy grids with safety metrics (like roughness, slopes, 
    and collision buffers) to produce cost values.
    """
    def __init__(self, num_envs, dilation_radius=2, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device
        self.dilation_radius = dilation_radius
        
        # 2D Kernel for morphological dilation (simulating safety buffer around obstacles)
        kernel_size = 2 * dilation_radius + 1
        self.kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)

    def process(self, occupancy_grids: torch.Tensor) -> torch.Tensor:
        """
        Processes occupancy maps to yield risk/terrain cost maps.
        
        occupancy_grids: Shape (num_envs, width, height)
        """
        # 1. Expand dimensions for PyTorch conv2d operations
        x = occupancy_grids.unsqueeze(1) # shape: (num_envs, 1, W, H)
        
        # 2. Perform dilation to inflate obstacle bounds by robot foot footprint radius
        dilated = F.conv2d(x, self.kernel, padding=self.dilation_radius)
        dilated = (dilated > 0.0).float().squeeze(1) # shape: (num_envs, W, H)
        
        # 3. Add roughness estimation (here simulated as distance-based cost layers)
        # Cost map values scale from 0.0 (safest path) to 1.0 (impassable/collision risk)
        costmaps = dilated * 0.8
        
        # Simulate some terrain gradient/roughness ripples
        W, H = occupancy_grids.shape[1], occupancy_grids.shape[2]
        ripples = torch.sin(torch.linspace(0, 3.14 * 4, W, device=self.device)).view(1, W, 1).repeat(self.num_envs, 1, H)
        costmaps = torch.clamp(costmaps + ripples.abs() * 0.2, 0.0, 1.0)
        
        return costmaps
