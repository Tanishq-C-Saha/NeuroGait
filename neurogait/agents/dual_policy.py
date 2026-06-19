# d:/CAPSTONE/Navigation/neurogait/agents/dual_policy.py
import torch
from neurogait.agents.base_agent import BaseAgent

class DualPolicyCoordinator(BaseAgent):
    """
    AT7: Dual Navigation Policy Coordinator.
    Blends outputs from the Progress Policy (optimizing speed to target) and 
    the Caution Policy (optimizing collision clearance on rubble) weighted 
    dynamically by the estimated collision risk probability.
    """
    def __init__(self, cfg, observation_space, action_space, progress_policy, caution_policy, device="cuda:0"):
        super().__init__(cfg, observation_space, action_space, device)
        self.progress_policy = progress_policy
        self.caution_policy = caution_policy
        print("Initialized NeuroGait Dual Navigation Policy Coordinator")

    def act(self, observations: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        """
        AT7: Computes soft blended commands:
        cmd = (1.0 - risk) * cmd_progress + risk * cmd_caution
        """
        # 1. Query risk prediction from the risk prediction head of the progress policy (or default)
        if hasattr(self.progress_policy, "predict_risk"):
            risk = self.progress_policy.predict_risk(observations) # shape: (num_envs, 1)
        elif hasattr(self.caution_policy, "predict_risk"):
            risk = self.caution_policy.predict_risk(observations)
        else:
            # Default fallback risk prediction if not trained
            risk = torch.zeros((observations.shape[0], 1), device=self.device)

        # Ensure risk stays within probability bounds
        risk = torch.clamp(risk, min=0.0, max=1.0)

        # 2. Query individual policy actions
        cmd_progress = self.progress_policy.act(observations, evaluation) # shape: (num_envs, 3)
        cmd_caution = self.caution_policy.act(observations, evaluation)   # shape: (num_envs, 3)

        # 3. Soft blending math formulation
        blended_cmd = (1.0 - risk) * cmd_progress + risk * cmd_caution
        return blended_cmd

    def record_transition(self, obs, action, reward, next_obs, done, info) -> None:
        # Log transition data to both policies during training/evaluation
        self.progress_policy.record_transition(obs, action, reward, next_obs, done, info)
        self.caution_policy.record_transition(obs, action, reward, next_obs, done, info)

    def train_step(self) -> dict:
        # Dynamic policy parameters are optimized offline or in independent training stages
        return {}

    def save(self, path: str) -> None:
        self.progress_policy.save(path + "_progress.pt")
        self.caution_policy.save(path + "_caution.pt")

    def load(self, path: str) -> None:
        self.progress_policy.load(path + "_progress.pt")
        self.caution_policy.load(path + "_caution.pt")
