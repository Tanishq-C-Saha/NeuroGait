# d:/CAPSTONE/Navigation/neurogait/utils/heads.py
import torch
import torch.nn as nn

class RiskPredictionHead(nn.Module):
    """
    AT6: Risk Prediction Head.
    Auxiliary network processing exteroceptive and proprioceptive observations
    to predict the probability of safety violations (falls, high-clearance collisions).
    """
    def __init__(self, input_dim=128, device="cuda:0"):
        super().__init__()
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Bound probability between 0.0 and 1.0
        )
        self.to(device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Computes safety risk probability.
        
        observations: (batch_size, input_dim)
        """
        # Ensure correct tensor type and device
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        return self.network(observations)
