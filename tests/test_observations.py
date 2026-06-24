# d:/CAPSTONE/Navigation/tests/test_observations.py
import unittest
import torch
from neurogait.envs.observations import (
    NavigationObservationGroup,
    quat_rotate_inverse,
    get_goal_direction,
    get_goal_distance,
    get_robot_base_velocity,
    get_next_three_waypoints,
    get_nearest_obstacle_positions
)

class MockEnv:
    def __init__(self, num_envs=4, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        
        # Setup mock properties
        self.goal_positions = torch.tensor([[10.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
        self.robot_positions = torch.zeros((num_envs, 3), device=device)
        self.astar_paths = [[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)] for _ in range(num_envs)]
        self.costmaps = torch.zeros((num_envs, 32, 32), device=device)

class TestObservations(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.num_envs = 4
        self.env = MockEnv(self.num_envs, self.device)
        self.group = NavigationObservationGroup()

    def test_quat_rotation_inverse(self):
        # Quaternion rotating vector (1, 0, 0) by 90 degrees around Z axis [w, x, y, z]
        # q = [cos(45), 0, 0, sin(45)] = [0.7071, 0, 0, 0.7071]
        q = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], device=self.device)
        v = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        
        # Rotating by inverse of 90 degrees yaw (which is -90 degrees yaw)
        # Vector (1, 0, 0) should become (0, 1, 0)
        v_rot = quat_rotate_inverse(q, v)
        self.assertAlmostEqual(v_rot[0, 0].item(), 0.0, delta=1e-3)
        self.assertAlmostEqual(v_rot[0, 1].item(), -1.0, delta=1e-3)
        self.assertAlmostEqual(v_rot[0, 2].item(), 0.0, delta=1e-3)

    def test_observation_dimensions(self):
        obs = self.group.get_obs(self.env)
        expected_dim = self.group.get_observation_size()
        
        self.assertEqual(obs.shape, (self.num_envs, expected_dim))
        self.assertFalse(torch.isnan(obs).any())
        self.assertFalse(torch.isinf(obs).any())

if __name__ == "__main__":
    unittest.main()
