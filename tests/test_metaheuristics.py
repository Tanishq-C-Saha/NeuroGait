# d:/CAPSTONE/Navigation/tests/test_metaheuristics.py
import unittest
import numpy as np
import torch
from neurogait.agents.metaheuristics.optimizer_framework import (
    AntColonyPathOptimizer,
    GreyWolfPathOptimizer,
    GeneticAlgorithmPathOptimizer,
    BenchmarkRunner
)
from neurogait.envs.navigation_env import NeuroGaitNavigationRubbleEnv
from neurogait.envs.wrappers import SkrlEnvWrapper
from neurogait.agents.ppo_agent import NavigationPPOAgent

class TestMetaheuristics(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.costmap = np.zeros((32, 32))
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([5.0, 5.0])

    def test_aco_path_shape(self):
        optimizer = AntColonyPathOptimizer()
        path = optimizer.optimize_path(self.costmap, self.start, self.goal)
        self.assertEqual(path.shape[0], 4)
        self.assertEqual(path.shape[1], 2)
        np.testing.assert_almost_equal(path[0], self.start)
        np.testing.assert_almost_equal(path[-1], self.goal)

    def test_gwo_path_shape(self):
        optimizer = GreyWolfPathOptimizer()
        path = optimizer.optimize_path(self.costmap, self.start, self.goal)
        self.assertEqual(path.shape[0], 4)
        np.testing.assert_almost_equal(path[0], self.start)
        np.testing.assert_almost_equal(path[-1], self.goal)

    def test_benchmark_runner(self):
        class EnvConfig:
            def __init__(self):
                self.num_envs = 1
                self.decimation = 10
                self.name = "NeuroGait-Navigation-Rubble-v0"
        
        env = NeuroGaitNavigationRubbleEnv(EnvConfig())
        wrapped_env = SkrlEnvWrapper(env)
        
        class MockConfig:
            def get(self, k, d):
                return d
                
        agent = NavigationPPOAgent(MockConfig(), wrapped_env.observation_space, wrapped_env.action_space, self.device)
        
        runner = BenchmarkRunner(wrapped_env, agent, self.device)
        report = runner.run_all_benchmarks()
        
        self.assertTrue("Success Rate" in report)
        self.assertTrue("PPO+ACO" in report)
        self.assertTrue("PPO+GWO" in report)

if __name__ == "__main__":
    unittest.main()
