# d:/CAPSTONE/Navigation/tests/test_perception.py
import unittest
import torch
from neurogait.perception.occupancy import OccupancyGridPipeline
from neurogait.perception.costmap import CostMapPipeline

class TestPerception(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.num_envs = 4
        self.grid_width = 32
        self.grid_height = 32

        self.occupancy_pipeline = OccupancyGridPipeline(
            self.num_envs, 
            grid_width=self.grid_width, 
            grid_height=self.grid_height,
            device=self.device
        )
        self.costmap_pipeline = CostMapPipeline(
            self.num_envs, 
            dilation_radius=1, 
            device=self.device
        )

    def test_occupancy_shapes(self):
        depth_images = torch.zeros((self.num_envs, 1, 64, 64), device=self.device)
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        orientations = torch.zeros((self.num_envs, 4), device=self.device)
        orientations[:, 3] = 1.0 # identity quaternion

        grids = self.occupancy_pipeline.process(depth_images, positions, orientations)
        self.assertEqual(grids.shape, (self.num_envs, self.grid_width, self.grid_height))

    def test_costmap_shapes(self):
        mock_occupancy = torch.zeros((self.num_envs, self.grid_width, self.grid_height), device=self.device)
        costmaps = self.costmap_pipeline.process(mock_occupancy)
        self.assertEqual(costmaps.shape, (self.num_envs, self.grid_width, self.grid_height))
        self.assertTrue(torch.all(costmaps >= 0.0))
        self.assertTrue(torch.all(costmaps <= 1.0))

if __name__ == "__main__":
    unittest.main()
