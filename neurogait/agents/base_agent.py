# d:/CAPSTONE/Navigation/neurogait/agents/base_agent.py
from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    """
    Abstract interface allowing PPO/SAC/TD3 and nature-inspired algorithms
    (ACO, RSA, GA, GWO, GAT-networks) to share a standardized training and step API.
    """
    def __init__(self, cfg, observation_space, action_space, device="cuda:0"):
        self.cfg = cfg
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def act(self, observations: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        """Computes navigation output commands (e.g. target velocities)."""
        pass

    @abstractmethod
    def record_transition(self, obs, action, reward, next_obs, done, info) -> None:
        """Stores experience buffer data (essential for RL and metaheuristics)."""
        pass

    @abstractmethod
    def train_step(self) -> dict:
        """Updates internal weights or population parameters."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serializes agent policy parameters or population models."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Restores policy checkpoints or optimization states."""
        pass
