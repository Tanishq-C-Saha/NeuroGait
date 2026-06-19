# d:/CAPSTONE/Navigation/neurogait/agents/ppo_agent.py
import os
import torch
import torch.nn as nn
from neurogait.agents.base_agent import BaseAgent
from neurogait.utils.heads import RiskPredictionHead

# Optional imports from skrl (mocked/wrapped safely if skrl/isaaclab is not in system env)
try:
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    SKRL_AVAILABLE = True
except ImportError:
    SKRL_AVAILABLE = False
    PPO_DEFAULT_CONFIG = {}

# Standard PyTorch Actor-Critic model template for high-level Navigation PPO
if SKRL_AVAILABLE:
    class PolicyModel(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
            
            # High level observation size is flat shape of observations
            obs_dim = observation_space.shape[0] if hasattr(observation_space, "shape") else 128
            act_dim = action_space.shape[0] if hasattr(action_space, "shape") else 3
            
            # Shared trunk
            self.shared_trunk = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU()
            )
            
            # Action mean head
            self.action_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, act_dim)
            )
            
            # Collision probability head
            self.risk_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 1),
                nn.Sigmoid() # Bounded collision probability
            )
            self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

        def act(self, inputs, role):
            # inputs["states"] has the observation buffer
            states = inputs["states"]
            shared_features = self.shared_trunk(states)
            action_mean = self.action_head(shared_features)
            collision_prob = self.risk_head(shared_features)
            
            # Return action mean, log standard deviation, and dictionary of auxiliary head values
            return action_mean, self.log_std_parameter, {"collision_probability": collision_prob}


    class ValueModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self)
                
            obs_dim = observation_space.shape[0] if hasattr(observation_space, "shape") else 128
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 1)
            )

        def act(self, inputs, role):
            states = inputs["states"]
            return self.net(states), {}
else:
    class PolicyModel:
        pass
    class ValueModel:
        pass


class NavigationPPOAgent(BaseAgent):
    """
    AT1, AT4, AT6: PPO agent wrapper for skrl engine that includes
    the auxiliary Risk Prediction Head and support for training/evaluation flags.
    """
    def __init__(self, cfg, observation_space, action_space, device="cuda:0"):
        super().__init__(cfg, observation_space, action_space, device)
        
        self.loss_fn_risk = nn.BCELoss()
        self.risk_weight = cfg.get("risk_head", {}).get("loss_weight", 0.5)

        if SKRL_AVAILABLE:
            # Custom PPO setup
            self.policy_model = PolicyModel(observation_space, action_space, device)
            self.value_model = ValueModel(observation_space, action_space, device)
            
            # Setup joint optimizer for shared trunk and auxiliary risk head parameters
            self.optimizer_risk = torch.optim.Adam(
                list(self.policy_model.shared_trunk.parameters()) + 
                list(self.policy_model.risk_head.parameters()), 
                lr=1e-3
            )
            
            # Configure PPO parameters from cfg
            skrl_cfg = PPO_DEFAULT_CONFIG.copy()
            skrl_cfg["learning_rate"] = cfg.get("learning_rate", 3e-4)
            skrl_cfg["mini_batches"] = cfg.get("mini_batches", 4)
            skrl_cfg["discount_factor"] = cfg.get("discount_factor", 0.99)
            skrl_cfg["lambda"] = cfg.get("lambda_gae", 0.95)
            skrl_cfg["entropy_loss_scale"] = cfg.get("entropy_coefficient", 0.01)
            skrl_cfg["ratio_clip_range"] = cfg.get("clip_range", 0.2)
            skrl_cfg["grad_norm_clip"] = 0.5
            skrl_cfg["state_preprocessor"] = None
            
            self.skrl_agent = PPO(
                models={"policy": self.policy_model, "value": self.value_model},
                memory=None, # will be populated by skrl environment loader
                cfg=skrl_cfg,
                observation_space=observation_space,
                action_space=action_space,
                device=device
            )
        else:
            self.skrl_agent = None
            self.risk_head = RiskPredictionHead(
                input_dim=observation_space.shape[0] if hasattr(observation_space, "shape") else 128,
                device=device
            )
            self.optimizer_risk = torch.optim.Adam(self.risk_head.parameters(), lr=1e-3)
            print("[Warning] skrl is not installed. PPO agent is initialized in mockup mode.")


    def act(self, observations: torch.Tensor, evaluation: bool = False) -> torch.Tensor:
        if self.skrl_agent is not None:
            # Wrap standard forward pass
            with torch.no_grad():
                inputs = {"states": observations}
                action, _, _ = self.policy_model.act(inputs, role="policy")
            return action
        else:
            # Mock actions: random velocity targets: (v_x, v_y, omega_z)
            batch_size = observations.shape[0] if len(observations.shape) > 1 else 1
            return torch.zeros((batch_size, 3), device=self.device)

    def predict_risk(self, observations: torch.Tensor) -> torch.Tensor:
        """
        AT6: Predict probability of fall/collision.
        """
        if self.skrl_agent is not None:
            with torch.no_grad():
                shared = self.policy_model.shared_trunk(observations)
                return self.policy_model.risk_head(shared)
        else:
            return self.risk_head(observations)

    def record_transition(self, obs, action, reward, next_obs, done, info) -> None:
        # Pass to skrl memory if needed (skrl handles this inside its training loop automatically)
        pass

    def train_step(self) -> dict:
        # skrl manages the learning loop, but we can update our risk head here
        return {}

    def update_risk_head(self, observations: torch.Tensor, risk_targets: torch.Tensor) -> float:
        """
        Trains risk head using cross-entropy or binary classification loss.
        """
        self.optimizer_risk.zero_grad()
        if self.skrl_agent is not None:
            shared = self.policy_model.shared_trunk(observations)
            predictions = self.policy_model.risk_head(shared)
        else:
            predictions = self.risk_head(observations)
            
        loss = self.loss_fn_risk(predictions, risk_targets.float().view(-1, 1))
        loss.backward()
        self.optimizer_risk.step()
        return loss.item()

    def init_wandb(self, logger_cfg):
        if SKRL_AVAILABLE:
            # Code to bind wandb logger to skrl agent
            pass

    def save(self, path: str) -> None:
        if self.skrl_agent is not None:
            torch.save({
                'policy_state_dict': self.policy_model.state_dict(),
                'value_state_dict': self.value_model.state_dict(),
                'risk_head_state_dict': self.risk_head.state_dict()
            }, path)
        else:
            torch.save({'risk_head_state_dict': self.risk_head.state_dict()}, path)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if self.skrl_agent is not None and 'policy_state_dict' in checkpoint:
                self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
                self.value_model.load_state_dict(checkpoint['value_state_dict'])
            if 'risk_head_state_dict' in checkpoint:
                self.risk_head.load_state_dict(checkpoint['risk_head_state_dict'])
            print(f"Successfully loaded agent weights from {path}")
        else:
            print(f"Weights file not found at {path}")
class MockSpace:
    def __init__(self, shape):
        self.shape = shape
