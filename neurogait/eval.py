# d:/CAPSTONE/Navigation/neurogait/eval.py
import os
import hydra
from omegaconf import DictConfig
import torch
import numpy as np

from neurogait.envs.navigation_env import NeuroGaitNavigationRubbleEnv
from neurogait.envs.wrappers import SkrlEnvWrapper
from neurogait.agents.ppo_agent import NavigationPPOAgent
from neurogait.utils.metrics import MetricsCollector

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("="*60)
    print("         NeuroGait: Deep Legged Navigation Evaluator")
    print("="*60)
    
    device = cfg.device
    torch.manual_seed(cfg.seed)
    
    # 1. Instantiate wrapped environment
    env = NeuroGaitNavigationRubbleEnv(cfg.env)
    wrapped_env = SkrlEnvWrapper(env)
    
    # 2. Instantiate and load PPO agent
    agent = NavigationPPOAgent(
        cfg=cfg.agent,
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=device
    )
    
    # Locate checkpoint
    checkpoint = cfg.checkpoint_path if cfg.checkpoint_path is not None else "checkpoints/neurogait_final.pt"
    if os.path.exists(checkpoint):
        agent.load(checkpoint)
    else:
        print(f"[Warning] No checkpoint found at {checkpoint}. Running evaluation with random policy.")

    # 3. Setup metrics aggregator
    collector = MetricsCollector()
    
    # 4. Evaluation Loop
    num_eval_episodes = 20
    print(f"Executing {num_eval_episodes} evaluation rollout trials...")
    
    for episode in range(num_eval_episodes):
        obs, info = wrapped_env.reset()
        done = False
        
        trajectory = []
        actions = []
        risk_errors = []
        steps = 0
        
        # Track initial position
        trajectory.append(env.robot_positions[0].cpu().numpy().copy())
        
        while not done and steps < 200:
            # Deterministic/eval action selection
            action = agent.act(obs, evaluation=True)
            
            # Predict risk and verify accuracy
            predicted_risk = agent.predict_risk(obs)
            
            next_obs, reward, terminated, truncated, extras = wrapped_env.step(action)
            
            # Calculate classification error of risk head
            actual_risk = float(terminated[0].item() or truncated[0].item())
            risk_errors.append(abs(predicted_risk[0].item() - actual_risk))
            
            obs = next_obs
            steps += 1
            
            trajectory.append(env.robot_positions[0].cpu().numpy().copy())
            actions.append(action[0].cpu().numpy().copy())
            
            done = (terminated[0].item() or truncated[0].item())
            
        success = extras["success"][0].item()
        collision = extras["collisions"][0].item()
        
        collector.update_episode(
            success=success,
            collision=collision,
            steps=steps,
            trajectory=np.array(trajectory),
            actions=np.array(actions),
            risk_errors=risk_errors
        )
        print(f"Trial {episode+1:02d} complete | Steps: {steps:3d} | Success: {success} | Collision: {collision}")

    # 5. Compile and print metrics summary for publication
    summary = collector.get_summary()
    print("="*60)
    print("                  EVALUATION METRICS REPORT")
    print("="*60)
    for k, v in summary.items():
        print(f"{k.replace('_', ' ').title():<30} : {v:.4f}")
    print("="*60)

    # Save metrics report as csv
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/evaluation_metrics.csv"
    with open(report_path, "w") as f:
        f.write("metric,value\n")
        for k, v in summary.items():
            f.write(f"{k},{v}\n")
    print(f"Serialized final evaluation report table to {report_path}")

if __name__ == "__main__":
    main()
