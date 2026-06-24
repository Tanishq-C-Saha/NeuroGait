# d:/CAPSTONE/Navigation/tests/test_agents.py
import unittest
import torch
from neurogait.utils.heads import RiskPredictionHead
from neurogait.utils.pareto import ParetoFrontOptimizer
from neurogait.agents.ppo_agent import NavigationPPOAgent

class MockSpace:
    def __init__(self, shape):
        self.shape = shape

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.obs_dim = 128
        self.act_dim = 3
        
        self.risk_head = RiskPredictionHead(input_dim=self.obs_dim, device=self.device)

    def test_risk_head_outputs(self):
        batch_size = 5
        dummy_obs = torch.randn((batch_size, self.obs_dim), device=self.device)
        predictions = self.risk_head(dummy_obs)
        
        self.assertEqual(predictions.shape, (batch_size, 1))
        self.assertTrue(torch.all(predictions >= 0.0))
        self.assertTrue(torch.all(predictions <= 1.0))

    def test_pareto_front(self):
        # 3 points with 2 objectives (to maximize)
        # Point 0: [1.0, 1.0] (Dominates point 2)
        # Point 1: [0.5, 2.0] (Non-dominated)
        # Point 2: [0.8, 0.8] (Dominated by point 0)
        objectives = torch.tensor([
            [1.0, 1.0],
            [0.5, 2.0],
            [0.8, 0.8]
        ]).numpy()
        
        mask = ParetoFrontOptimizer.identify_pareto(objectives)
        self.assertTrue(mask[0]) # Point 0 is Pareto optimal
        self.assertTrue(mask[1]) # Point 1 is Pareto optimal
        self.assertFalse(mask[2]) # Point 2 is Dominated

if __name__ == "__main__":
    unittest.main()
