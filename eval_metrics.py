# d:/CAPSTONE/Navigation/eval_metrics.py
import os
import argparse
import csv
import torch
import numpy as np

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from neurogait.envs.navigation_env import NeuroGaitNavigationRubbleEnv
from neurogait.envs.wrappers import SkrlEnvWrapper
from neurogait.agents.ppo_agent import NavigationPPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="NeuroGait Performance Evaluation Metrics Collector")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/neurogait_final.pt", help="Path to policy checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save logs and plots")
    return parser.parse_args()

class Evaluator:
    """
    Evaluator executing rollout episodes and calculating key navigation metrics.
    """
    def __init__(self, env, agent, num_episodes, output_dir, device):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        
        # Metric storage
        self.episode_logs = []

    def run(self):
        print(f"[Evaluator] Starting {self.num_episodes} evaluation episodes...")
        dt = 0.05 # 20 Hz navigation step time
        mass = 15.0 # Go2 mass in kg
        g = 9.81
        
        for ep in range(self.num_episodes):
            obs, info = self.env.reset()
            done = False
            
            # Temporary state buffers
            trajectory = []
            actions = []
            collisions = 0
            steps = 0
            
            # Record start pos
            start_pos = self.env.env.robot_positions[0].cpu().numpy().copy()
            trajectory.append(start_pos)
            
            while not done and steps < 200:
                # Query action
                action = self.agent.act(obs, evaluation=True)
                
                # Step env
                next_obs, reward, terminated, truncated, extras = self.env.step(action)
                
                # Check collision force threshold
                has_collided = extras["collisions"][0].item()
                if has_collided:
                    collisions += 1
                    
                obs = next_obs
                steps += 1
                
                # Record trajectories
                trajectory.append(self.env.env.robot_positions[0].cpu().numpy().copy())
                actions.append(action[0].cpu().numpy().copy())
                
                done = terminated[0].item() or truncated[0].item()
            
            # Post episode calculation
            success = int(extras["success"][0].item())
            time_taken = steps * dt
            
            # Trajectory arrays
            trajectory = np.array(trajectory)
            actions = np.array(actions)
            
            # Path Length
            if len(trajectory) > 1:
                diffs = np.diff(trajectory[:, :2], axis=0)
                path_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=-1))))
            else:
                path_length = 0.0
                
            # A* Optimal Path Length Estimation
            # Straight line represents idealized optimal path on rubble
            goal_pos = self.env.env.goal_positions[0].cpu().numpy()
            optimal_length = float(np.norm(goal_pos[:2] - start_pos[:2]))
            
            # Path efficiency calculation
            path_efficiency = (optimal_length / path_length) if path_length > 0.1 else 0.0
            path_efficiency = min(path_efficiency, 1.0) # bound at 1.0
            
            # Energy consumption (Cost of Transport)
            # Proxy energy = sum of squared velocity commands
            if path_length > 0.1:
                energy_proxy = float(np.sum(np.square(actions)) * dt)
                cost_of_transport = energy_proxy / (mass * g * path_length)
            else:
                cost_of_transport = 0.0

            log_entry = {
                "episode": ep + 1,
                "success": success,
                "time_to_goal": time_taken if success else -1.0,
                "steps": steps,
                "collision_count": collisions,
                "path_length": path_length,
                "optimal_length": optimal_length,
                "path_efficiency": path_efficiency,
                "cost_of_transport": cost_of_transport,
                "trajectory": trajectory
            }
            self.episode_logs.append(log_entry)
            print(f"Episode {ep+1:02d} | Success: {success} | Time: {time_taken:.2f}s | Collisions: {collisions} | COT: {cost_of_transport:.4f}")

    def export_csv(self):
        """
        3. CSV Pipeline.
        Saves step results to a tabular CSV format.
        """
        csv_path = os.path.join(self.output_dir, "episode_metrics.csv")
        headers = ["episode", "success", "time_to_goal", "steps", "collision_count", "path_length", "path_efficiency", "cost_of_transport"]
        
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for entry in self.episode_logs:
                # Write matching row excluding trajectories
                row = {k: entry[k] for k in headers}
                writer.writerow(row)
        print(f"[Evaluator] Exported episode CSV logs to {csv_path}")

    def generate_benchmark_report(self):
        """
        5. Benchmark Report Generation.
        Compiles summary metrics for research papers.
        """
        num_runs = len(self.episode_logs)
        successes = [e["success"] for e in self.episode_logs]
        success_rate = np.mean(successes)
        
        # Filter successful episodes for time calculations
        success_times = [e["time_to_goal"] for e in self.episode_logs if e["success"] == 1]
        mean_time = np.mean(success_times) if len(success_times) > 0 else 0.0
        
        total_collisions = sum([e["collision_count"] for e in self.episode_logs])
        mean_path_efficiency = np.mean([e["path_efficiency"] for e in self.episode_logs])
        mean_cot = np.mean([e["cost_of_transport"] for e in self.episode_logs])
        
        report_path = os.path.join(self.output_dir, "benchmark_report.txt")
        
        report_text = f"""============================================================
              NEUROGAIT NAVIGATION BENCHMARK REPORT
============================================================
Date/Execution  : 2026-06-19
Policy Checkpoint: {self.agent.cfg.get('agile_weight', 'custom_run')}
Total Episodes  : {num_runs}

PERFORMANCE METRICS SUMMARY:
------------------------------------------------------------
Success Rate            : {success_rate * 100:.2f} %
Mean Time to Goal (sec) : {mean_time:.3f} s (Successful runs only)
Total Collision Events  : {total_collisions}
Mean Path Efficiency    : {mean_path_efficiency:.4f}
Mean Cost of Transport  : {mean_cot:.4f}

DETAILED EPISODE TABLE:
------------------------------------------------------------
{ 'Ep':<4 } | { 'Success':<8 } | { 'Time (s)':<8 } | { 'Collisions':<10 } | { 'Efficiency':<10 } | { 'COT':<8 }
"""
        for e in self.episode_logs:
            report_text += f"{e['episode']:<4d} | {e['success']:<8d} | {e['time_to_goal']:<8.2f} | {e['collision_count']:<10d} | {e['path_efficiency']:<10.4f} | {e['cost_of_transport']:<8.4f}\n"
            
        report_text += "============================================================\n"
        
        with open(report_path, "w") as f:
            f.write(report_text)
            
        print("="*60)
        print(report_text)
        print("="*60)

    def save_plots(self):
        """
        4. Plotting Utilities.
        Visualizes trajectories and metric distributions.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[Evaluator] [Warning] matplotlib is not installed. Skipping plot generation.")
            return

        # Plot 1: 2D Trajectories
        plt.figure(figsize=(8, 6))
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        
        for e in self.episode_logs:
            traj = e["trajectory"]
            color = 'g' if e["success"] == 1 else 'r'
            plt.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, label="Success" if e["episode"] == 1 else None)
            
        # Draw goal
        goal_pos = self.env.env.goal_positions[0].cpu().numpy()
        plt.scatter(goal_pos[0], goal_pos[1], color='gold', marker='*', s=150, zorder=5, label="Goal Target")
        
        plt.title("Robot Trajectories on Rubble Terrain")
        plt.xlabel("X Position (meters)")
        plt.ylabel("Y Position (meters)")
        plt.grid(True)
        plt.legend()
        traj_path = os.path.join(self.output_dir, "trajectories.png")
        plt.savefig(traj_path, dpi=150)
        plt.close()
        
        # Plot 2: Performance Distributions Boxplot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        cots = [e["cost_of_transport"] for e in self.episode_logs]
        plt.boxplot(cots)
        plt.title("Cost of Transport (COT)")
        plt.ylabel("COT Value")
        
        plt.subplot(1, 2, 2)
        efficiencies = [e["path_efficiency"] for e in self.episode_logs]
        plt.boxplot(efficiencies)
        plt.title("Path Efficiency")
        plt.ylabel("Ratio")
        
        plt.tight_layout()
        dist_path = os.path.join(self.output_dir, "metrics_distribution.png")
        plt.savefig(dist_path, dpi=150)
        plt.close()
        
        print(f"[Evaluator] Plots saved to {self.output_dir}/")

def main():
    args = parse_args()
    
    # Instantiate env wrapper
    class EnvConfig:
        def __init__(self, num_envs):
            self.num_envs = num_envs
            self.decimation = 10
            self.name = "NeuroGait-Navigation-Rubble-v0"
            
    env_cfg = EnvConfig(1) # Eval runs on single environments
    env = NeuroGaitNavigationRubbleEnv(env_cfg)
    wrapped_env = SkrlEnvWrapper(env)
    
    # Instantiate agent
    class MockConfig:
        def get(self, k, d):
            return d
            
    agent = NavigationPPOAgent(
        cfg=MockConfig(),
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=args.device
    )
    
    # Load weights
    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        
    evaluator = Evaluator(
        env=wrapped_env,
        agent=agent,
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    evaluator.run()
    
    # Export summaries
    evaluator.export_csv()
    evaluator.generate_benchmark_report()
    evaluator.save_plots()

if __name__ == "__main__":
    main()
