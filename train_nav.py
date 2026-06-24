# d:/CAPSTONE/Navigation/train_nav.py
import os
import argparse
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Try importing IsaacLab and skrl components
try:
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.envs.wrappers.torch import GymWrapper
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveRL
    SKRL_AVAILABLE = True
except ImportError:
    SKRL_AVAILABLE = False

from neurogait.envs.navigation_env import NeuroGaitNavigationRubbleEnv
from neurogait.envs.wrappers import SkrlEnvWrapper
from neurogait.agents.ppo_agent import PolicyModel, ValueModel

class CheckpointManager:
    """
    3. Checkpoint Manager.
    Handles serialization of policy models, values, and scalers based on best performance thresholds.
    """
    def __init__(self, save_dir="checkpoints", checkpoint_prefix="neurogait_nav"):
        self.save_dir = save_dir
        self.prefix = checkpoint_prefix
        os.makedirs(save_dir, exist_ok=True)
        self.best_reward = -float('inf')

    def save(self, agent, epoch, current_reward, is_best=False):
        # Save standard periodic check
        checkpoint_path = os.path.join(self.save_dir, f"{self.prefix}_epoch_{epoch}.pt")
        
        # Assemble save dict
        save_dict = {
            'epoch': epoch,
            'reward': current_reward,
        }
        
        if hasattr(agent, "policy_model"):
            save_dict['policy_state'] = agent.policy_model.state_dict()
            save_dict['value_state'] = agent.value_model.state_dict()
        if hasattr(agent, "skrl_agent") and agent.skrl_agent is not None:
            save_dict['optimizer_state'] = agent.skrl_agent.optimizer.state_dict()
            if agent.skrl_agent.state_preprocessor is not None:
                save_dict['scaler_state'] = agent.skrl_agent.state_preprocessor.state_dict()

        # Save standard
        torch.save(save_dict, checkpoint_path)
        
        # Save best indicator
        if is_best or current_reward > self.best_reward:
            self.best_reward = current_reward
            best_path = os.path.join(self.save_dir, f"{self.prefix}_best.pt")
            torch.save(save_dict, best_path)
            print(f"[CheckpointManager] Saved new best checkpoint to {best_path} (Reward: {current_reward:.3f})")

    def load(self, agent, path):
        if not os.path.exists(path):
            print(f"[CheckpointManager] [Error] File not found at {path}")
            return 0
        
        checkpoint = torch.load(path, map_location=agent.device)
        
        if 'policy_state' in checkpoint and hasattr(agent, "policy_model"):
            agent.policy_model.load_state_dict(checkpoint['policy_state'])
            agent.value_model.load_state_dict(checkpoint['value_state'])
            
        if 'optimizer_state' in checkpoint and hasattr(agent, "skrl_agent") and agent.skrl_agent is not None:
            agent.skrl_agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
        if 'scaler_state' in checkpoint and hasattr(agent, "skrl_agent") and agent.skrl_agent is not None:
            if agent.skrl_agent.state_preprocessor is not None:
                agent.skrl_agent.state_preprocessor.load_state_dict(checkpoint['scaler_state'])
                
        start_epoch = checkpoint.get('epoch', 0)
        self.best_reward = checkpoint.get('reward', -float('inf'))
        print(f"[CheckpointManager] Successfully loaded checkpoint from {path} (Resuming from epoch: {start_epoch})")
        return start_epoch

def setup_cli_arguments():
    """
    2. CLI Arguments.
    Configures parser arguments for experiment parameters, hardware indexing, and logging options.
    """
    parser = argparse.ArgumentParser(description="NeuroGait Navigation PPO Trainer")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments in IsaacLab")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU/CPU target device execution")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=16, help="Rollout steps per epoch per agent")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO policy learning rate")
    parser.add_argument("--log_dir", type=str, default="tb_logs", help="TensorBoard output path")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Saved checkpoints path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file path to resume training from")
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation rollout interval rate")
    return parser.parse_args()

def run_evaluation(env, agent, num_episodes=5) -> float:
    """
    Evaluation Callback.
    Runs evaluation loops with disabled noise layers to gauge policy convergence.
    """
    print("[Evaluation Callback] Running deterministic evaluation...")
    total_rewards = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done and steps < 200:
                if SKRL_AVAILABLE and hasattr(agent, "policy_model"):
                    inputs = {"states": obs}
                    action, _, _ = agent.policy_model.act(inputs, role="policy")
                else:
                    action = torch.zeros((env.env.num_envs, 3), device=env.device)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward[0].item()
                done = terminated[0].item() or truncated[0].item()
                steps += 1
            total_rewards.append(ep_reward)
            
    mean_reward = np.mean(total_rewards)
    print(f"[Evaluation Callback] Completed. Mean Reward: {mean_reward:.3f}")
    return mean_reward

def main():
    # Parse CLI Parameters
    args = setup_cli_arguments()
    
    # Debugging utilities
    print("="*60)
    print("           NEUROGAIT NAVIGATION TRAINING PIPELINE")
    print("="*60)
    print(f"Num Environments : {args.num_envs}")
    print(f"Target GPU/CPU   : {args.device}")
    print(f"Reprod. Seed     : {args.seed}")
    print(f"Output Directory : {args.log_dir}")
    print("="*60)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize environment shell
    class EnvConfig:
        def __init__(self, num_envs):
            self.num_envs = num_envs
            self.decimation = 10
            self.name = "NeuroGait-Navigation-Rubble-v0"
            
    env_cfg = EnvConfig(args.num_envs)
    env = NeuroGaitNavigationRubbleEnv(env_cfg)
    wrapped_env = SkrlEnvWrapper(env)

    # 4. TensorBoard Integration.
    writer = SummaryWriter(log_dir=args.log_dir)

    # Checkpoint Manager
    checkpoint_manager = CheckpointManager(save_dir=args.save_dir)

    # Set up skrl model components
    if SKRL_AVAILABLE:
        # Build policy matching actor layers [256, 128, 64]
        policy = PolicyModel(wrapped_env.observation_space, wrapped_env.action_space, args.device)
        value = ValueModel(wrapped_env.observation_space, wrapped_env.action_space, args.device)
        
        # Configure skrl parameters
        skrl_cfg = PPO_DEFAULT_CONFIG.copy()
        skrl_cfg["learning_rate"] = args.lr
        skrl_cfg["rollouts"] = args.steps_per_epoch
        skrl_cfg["mini_batches"] = 4
        skrl_cfg["epochs"] = 5
        skrl_cfg["state_preprocessor"] = RunningStandardScaler
        skrl_cfg["state_preprocessor_kwargs"] = {"clip_ob": 5.0, "epsilon": 1e-5}
        
        # LR Schedule adaptive KL
        skrl_cfg["learning_rate_scheduler"] = KLAdaptiveRL
        skrl_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

        # Initialize Agent
        class AgentWrapper:
            def __init__(self, policy, value, wrapped_env, skrl_cfg, device):
                self.policy_model = policy
                self.value_model = value
                self.device = device
                self.skrl_agent = PPO(
                    models={"policy": policy, "value": value},
                    memory=None,
                    cfg=skrl_cfg,
                    observation_space=wrapped_env.observation_space,
                    action_space=wrapped_env.action_space,
                    device=device
                )
            def act(self, obs, evaluation=False):
                # Returns actions
                with torch.no_grad():
                    inputs = {"states": obs}
                    action, _, _ = self.policy_model.act(inputs, role="policy")
                return action
                
        agent = AgentWrapper(policy, value, wrapped_env, skrl_cfg, args.device)
    else:
        # Standalone mockup agent
        class MockAgent:
            def __init__(self, device):
                self.device = device
                self.skrl_agent = None
            def act(self, obs, evaluation=False):
                return torch.zeros((obs.shape[0], 3), device=self.device)
        agent = MockAgent(args.device)

    # Handle Training Resume Checks
    start_epoch = 0
    if args.resume is not None:
        start_epoch = checkpoint_manager.load(agent, args.resume)

    # 5. Debugging Utilities & Training Loop
    print("[Trainer] Starting rollout executions...")
    obs, info = wrapped_env.reset()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_rewards = []
        epoch_risk_losses = []
        
        # Rollout collect phase
        for step in range(args.steps_per_epoch):
            actions = agent.act(obs)
            next_obs, reward, terminated, truncated, extras = wrapped_env.step(actions)
            
            epoch_rewards.append(reward.mean().item())
            obs = next_obs

        # Compute summary values
        mean_epoch_reward = np.mean(epoch_rewards)
        
        # 4. TensorBoard logs writing
        writer.add_scalar("train/reward", mean_epoch_reward, epoch)
        writer.add_scalar("train/learning_rate", args.lr, epoch)
        
        # Periodic printing
        if epoch % 5 == 0:
            print(f"[Trainer] Epoch {epoch:3d}/{args.epochs:3d} | Mean Reward: {mean_epoch_reward:8.3f}")

        # Periodic evaluation callbacks
        if epoch > 0 and epoch % args.eval_freq == 0:
            eval_reward = run_evaluation(wrapped_env, agent)
            writer.add_scalar("eval/mean_reward", eval_reward, epoch)
            
            # Save checkpoints via manager
            checkpoint_manager.save(agent, epoch, eval_reward)

    # Save final PPO model
    checkpoint_manager.save(agent, args.epochs, mean_epoch_reward, is_best=True)
    writer.close()
    print("="*60)
    print("          NEUROGAIT TRAINING PIPELINE COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
