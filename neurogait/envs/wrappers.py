# d:/CAPSTONE/Navigation/neurogait/envs/wrappers.py
import torch
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


try:
    from skrl.envs.wrappers.torch import GymWrapper
    SKRL_AVAILABLE = True
except ImportError:
    SKRL_AVAILABLE = False
    class GymWrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.device = "cuda:0"

class SkrlEnvWrapper(GymWrapper):
    """
    Adapter wrapper to connect the custom NeuroGait navigation environment 
    to skrl's runner engine.
    """
    def __init__(self, env):
        super().__init__(env)
        self.device = torch.device(getattr(env, "device", "cuda:0"))

    def step(self, actions):
        """
        Executes action in simulation, converting outputs to standard PyTorch structures.
        """
        obs, reward, terminated, truncated, info = self.env.step(actions)
        
        # Ensure correct formatting for skrl expectations
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        else:
            reward = reward.view(-1, 1)
            
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device).view(-1, 1)
        else:
            terminated = terminated.view(-1, 1)
            
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device).view(-1, 1)
        else:
            truncated = truncated.view(-1, 1)
            
        return obs, reward, terminated, truncated, info

    def reset(self):
        """Resets the wrapped environment."""
        obs, info = self.env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return obs, info
