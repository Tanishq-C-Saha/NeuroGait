"""CNN+MLP hybrid navigation policy and value network for skrl PPO.

Architecture
------------
Observation (1615 dims) is split at index 1600:
  - [0 : 1600]   40×40 occupancy grid   → CNN branch
  - [1600 : 1615] scalar obs              → MLP branch
    ├── [1600:1609]  future waypoints (3×3)
    ├── [1609:1612]  robot velocity [vx, vy, yaw_rate]
    └── [1612:1615]  projected gravity

Actions (3 dims) after tanh squashing:
  [0]  vx      ∈ [-1.0, 1.0] m/s
  [1]  vy      ∈ [-1.0, 1.0] m/s
  [2]  heading ∈ [-π,  +π]  rad  (NOT yaw rate — see concept/02_heading_vs_yawrate.md)

skrl API note
-------------
GaussianMixin.act() calls compute(inputs, role) and reads outputs["log_std"].
compute() MUST return (mean_actions, {"log_std": log_std_tensor}).
The three-tuple form (mean, log_std, {}) is wrong and will raise a KeyError.
"""

import math

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

GRID_H = 40
GRID_W = 40
GRID_OBS_SIZE = GRID_H * GRID_W   # 1600
SCALAR_OBS_SIZE = 15              # 9 waypoints + 3 velocity + 3 gravity


def _build_cnn() -> tuple[nn.Sequential, int]:
    """Shared CNN builder.  Returns (cnn, flat_size)."""
    cnn = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # 40×40 → 20×20
        nn.ELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 20×20 → 10×10
        nn.ELU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 10×10 → 10×10
        nn.ELU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        flat_size = cnn(torch.zeros(1, 1, GRID_H, GRID_W)).shape[1]
    return cnn, flat_size


class NavigationPolicy(GaussianMixin, Model):
    """Actor: CNN for grid + MLP for scalars → [vx, vy, heading].

    The CNN flat_size is computed dynamically so grid dimensions can be
    changed without rewriting the FC layers.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = -1.0,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.grid_cnn, cnn_flat = _build_cnn()

        self.grid_fc = nn.Sequential(
            nn.Linear(cnn_flat, 128), nn.ELU(),
            nn.Linear(128, 64),      nn.ELU(),
        )

        self.scalar_encoder = nn.Sequential(
            nn.Linear(SCALAR_OBS_SIZE, 64), nn.ELU(),
            nn.Linear(64, 32),              nn.ELU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ELU(),
            nn.Linear(128, 64),      nn.ELU(),
            nn.Linear(64, self.num_actions),  # → 3
        )

        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std)
        )

    def compute(self, inputs: dict, role: str = ""):
        obs = inputs["observations"]

        # CNN branch — spatial grid
        grid_flat = obs[:, :GRID_OBS_SIZE]
        grid_2d   = grid_flat.view(-1, 1, GRID_H, GRID_W)
        grid_feat = self.grid_fc(self.grid_cnn(grid_2d))       # (B, 64)

        # MLP branch — scalar obs
        scalars      = obs[:, GRID_OBS_SIZE:]
        scalar_feat  = self.scalar_encoder(scalars)            # (B, 32)

        # Merged action head
        merged      = torch.cat([grid_feat, scalar_feat], dim=-1)
        raw_actions = self.policy_head(merged)                 # (B, 3)

        # Squash to valid command ranges:
        #   vx, vy  → [-1, 1]   via tanh
        #   heading → [-π, π]   via π × tanh
        vx_vy   = torch.tanh(raw_actions[:, :2])
        heading = math.pi * torch.tanh(raw_actions[:, 2:3])
        mean_actions = torch.cat([vx_vy, heading], dim=-1)

        return mean_actions, {"log_std": self.log_std_parameter}


class NavigationValue(DeterministicMixin, Model):
    """Critic: same CNN+MLP encoder → scalar value."""

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False, **kwargs):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.grid_cnn, cnn_flat = _build_cnn()

        self.grid_fc = nn.Sequential(
            nn.Linear(cnn_flat, 128), nn.ELU(),
            nn.Linear(128, 64),      nn.ELU(),
        )

        self.scalar_encoder = nn.Sequential(
            nn.Linear(SCALAR_OBS_SIZE, 64), nn.ELU(),
            nn.Linear(64, 32),              nn.ELU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ELU(),
            nn.Linear(128, 64),      nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs: dict, role: str = ""):
        obs = inputs["observations"]

        grid_flat = obs[:, :GRID_OBS_SIZE]
        grid_2d   = grid_flat.view(-1, 1, GRID_H, GRID_W)
        grid_feat = self.grid_fc(self.grid_cnn(grid_2d))

        scalars     = obs[:, GRID_OBS_SIZE:]
        scalar_feat = self.scalar_encoder(scalars)

        merged = torch.cat([grid_feat, scalar_feat], dim=-1)
        return self.value_head(merged), {}
