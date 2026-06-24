# d:/CAPSTONE/Navigation/neurogait/agents/td3_agent.py
import os
import torch
from neurogait.agents.base_agent import BaseAgent

try:
    from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
    SKRL_AVAILABLE = True
except ImportError:
    SKRL_AVAILABLE = False

class TD3Agent(BaseAgent):
    """
    TD3 agent baseline wrapper for comparison.
    """
    def __init__(self, cfg, observation_space, action_space, device="cuda:0"):
        super().__init__(cfg, observation_space, action_space, device)
        
        if SKRL_AVAILABLE:
            # Placeholder for actual skrl TD3 initialization
            pass
        else:
            print("[Warning] skrl is not installed. TD3 agent is initialized in mockup mode.")

    def act(self, observations: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        batch_size = observations.shape[0] if len(observations.shape) > 1 else 1
        return torch.zeros((batch_size, 3), device=self.device)

    def record_transition(self, obs, action, reward, next_obs, done, info) -> None:
        pass

    def train_step(self) -> dict:
        return {}

    def save(self, path: str) -> None:
        torch.save({"mock": True}, path)

    def load(self, path: str) -> None:
        pass
