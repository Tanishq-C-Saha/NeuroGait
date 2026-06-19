# d:/CAPSTONE/Navigation/neurogait/train.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from neurogait.envs.navigation_env import NeuroGaitNavigationRubbleEnv
from neurogait.envs.wrappers import SkrlEnvWrapper
from neurogait.agents.ppo_agent import NavigationPPOAgent

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("="*60)
    print("         NeuroGait: Deep Legged Navigation Trainer")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("-"*60)

    # Set device and random seeds
    torch.manual_seed(cfg.seed)
    device = cfg.device
    print(f"Executing training on target device: {device}")

    # 1. Instantiate parallel simulation environments
    print(f"Initializing simulator environment: {cfg.env.name}")
    env = NeuroGaitNavigationRubbleEnv(cfg.env)
    wrapped_env = SkrlEnvWrapper(env)

    # 2. Instantiate and configure Navigation Agent
    print(f"Creating high-level agent policy: {cfg.agent.class_name}")
    agent = NavigationPPOAgent(
        cfg=cfg.agent,
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=device
    )

    # 3. Handle checkpoints if resuming
    if cfg.checkpoint_path is not None:
        print(f"Resuming training from checkpoint: {cfg.checkpoint_path}")
        agent.load(cfg.checkpoint_path)

    # 4. Core training loop execution
    total_timesteps = cfg.agent.get("total_timesteps", 100000)
    print(f"Starting training loop for {total_timesteps} timesteps...")
    
    obs, info = wrapped_env.reset()
    last_actions = torch.zeros((wrapped_env.env.num_envs, 3), device=device)
    
    # Track metrics
    steps = 0
    while steps < total_timesteps:
        # Get actions from agent
        actions = agent.act(obs)
        
        # Take step in environment
        next_obs, rewards, terminated, truncated, extras = wrapped_env.step(actions)
        
        # Train risk head based on reset signals (representing collisions/falls)
        risk_targets = (terminated | truncated).float()
        loss_risk = agent.update_risk_head(obs, risk_targets)
        
        # Store transition and record steps (skrl PPO trains internally,
        # but we track training steps for standalone simulation dry-runs)
        agent.record_transition(obs, actions, rewards, next_obs, terminated | truncated, extras)
        
        obs = next_obs
        last_actions = actions.clone()
        steps += wrapped_env.env.num_envs
        
        if steps % (wrapped_env.env.num_envs * 10) == 0:
            print(f"Step {steps}/{total_timesteps} | Risk Loss: {loss_risk:.4f} | Reward Mean: {rewards.mean().item():.3f}")

    # 5. Save final weights
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/neurogait_final.pt"
    agent.save(save_path)
    print(f"Training completed successfully. Policy weights serialized to {save_path}")

if __name__ == "__main__":
    main()
