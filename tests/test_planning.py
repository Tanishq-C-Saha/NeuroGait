# d:/CAPSTONE/Navigation/tests/test_planning.py
import unittest
import torch
from neurogait.planning.planner import AStarPlanner

class TestPlanning(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.planner = AStarPlanner()

    def test_astar_search(self):
        num_envs = 2
        start_poses = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=self.device)
        goal_poses = torch.tensor([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], device=self.device)
        costmaps = torch.zeros((num_envs, 32, 32), device=self.device)

        paths = self.planner.plan_batch(start_poses, goal_poses, costmaps)
        
        self.assertEqual(len(paths), num_envs)
        for i in range(num_envs):
            self.assertTrue(len(paths[i]) >= 2)
            # Verify last coordinates are close to the target coordinates
            goal_x, goal_y = goal_poses[i, 0].item(), goal_poses[i, 1].item()
            path_end = paths[i][-1]
            self.assertAlmostEqual(path_end[0], goal_x, delta=0.5)
            self.assertAlmostEqual(path_end[1], goal_y, delta=0.5)

if __name__ == "__main__":
    unittest.main()
