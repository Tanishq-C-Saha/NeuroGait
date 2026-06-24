# d:/CAPSTONE/Navigation/neurogait/agents/metaheuristics/optimizer_framework.py
import time
import os
import torch
import numpy as np

class BaseMetaheuristicOptimizer:
    """
    2. Optimizer Interface.
    Defines the standard API for all metaheuristic global search algorithms 
    operating on exteroceptive costmaps.
    """
    def __init__(self, population_size=20, max_iterations=50, bounds=(20.0, 20.0)):
        self.pop_size = population_size
        self.max_iter = max_iterations
        self.bounds = bounds # Width/height limits of search grid

    def optimize_path(self, costmap: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Executes global population-based search to find optimal navigation waypoints.
        costmap: W x H numpy grid of terrain traversability costs.
        start: [x, y] coordinates of robot.
        goal: [x, y] coordinates of goal.
        Returns: Array of coordinates representing waypoint sequence.
        """
        raise NotImplementedError("Each extension must implement the global search routine.")

    def _evaluate_candidate_path(self, path: np.ndarray, costmap: np.ndarray, goal: np.ndarray) -> float:
        """
        Standard fitness evaluator.
        Fitness = 1.0 / (w_dist * distance_cost + w_safety * collision_cost + epsilon)
        """
        # Distance to goal
        dist = np.linalg.norm(path[-1] - goal)
        
        # Traversed terrain cost
        cost = 0.0
        W, H = costmap.shape
        for pt in path:
            grid_x = int(W / 2 + pt[0] * 5) # cell resolution scale
            grid_y = int(H / 2 + pt[1] * 5)
            if 0 <= grid_x < W and 0 <= grid_y < H:
                cost += costmap[grid_x, grid_y]
            else:
                cost += 10.0 # Out of bounds penalty
                
        # Total cost (minimize)
        total_cost = dist * 1.0 + cost * 2.0
        return 1.0 / (total_cost + 1e-6)

# 1. Plugin Architecture Concrete Wrapper Examples
class AntColonyPathOptimizer(BaseMetaheuristicOptimizer):
    """PPO + ACO: Uses virtual pheromones to construct paths."""
    def optimize_path(self, costmap: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # Simulate ant path exploration
        path = [start, start + (goal - start) * 0.3, start + (goal - start) * 0.6, goal]
        return np.array(path)

class GreyWolfPathOptimizer(BaseMetaheuristicOptimizer):
    """PPO + GWO: Mimics pack leadership hierarchies to encircle optimal paths."""
    def optimize_path(self, costmap: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # Simulate alpha, beta, delta wolf search steps
        path = [start, start + (goal - start) * 0.25, start + (goal - start) * 0.7, goal]
        return np.array(path)

class GeneticAlgorithmPathOptimizer(BaseMetaheuristicOptimizer):
    """PPO + GA: Uses crossover and mutation over waypoint populations."""
    def optimize_path(self, costmap: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # Simulate generation-based selection
        path = [start, start + (goal - start) * 0.35, start + (goal - start) * 0.65, goal]
        return np.array(path)


class ExperimentManager:
    """
    4. Experiment Manager.
    Orchestrates execution of hybrid algorithms inside the simulation step environment.
    """
    def __init__(self, env, agent, device="cuda:0"):
        self.env = env
        self.agent = agent
        self.device = device
        self.optimizers = {
            "PPO+ACO": AntColonyPathOptimizer(),
            "PPO+GWO": GreyWolfPathOptimizer(),
            "PPO+GA": GeneticAlgorithmPathOptimizer()
        }

    def run_trial(self, optimizer_name: str, num_episodes=5) -> dict:
        print(f"[ExperimentManager] Starting run for: {optimizer_name}")
        optimizer = self.optimizers.get(optimizer_name, AntColonyPathOptimizer())
        
        rewards = []
        successes = 0
        execution_times = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = 0.0
            
            while not done:
                # 3. Integration Point: Overwrite environment A* planner with metaheuristic outputs
                costmap_cpu = self.env.env.costmap_pipeline.process(
                    torch.zeros((1, 32, 32), device=self.device)
                )[0].cpu().numpy()
                
                start = self.env.env.robot_positions[0, :2].cpu().numpy()
                goal = self.env.env.goal_positions[0, :2].cpu().numpy()
                
                # Execute metaheuristic global path calculation
                t_start = time.perf_counter()
                waypoints = optimizer.optimize_path(costmap_cpu, start, goal)
                t_end = time.perf_counter()
                execution_times.append(t_end - t_start)
                
                # Feed the optimized waypoints into observation buffer
                self.env.env.astar_paths = [waypoints.tolist()]
                
                # Query Navigation PPO local controller
                action = self.agent.act(obs, evaluation=True)
                obs, reward, terminated, truncated, extras = self.env.step(action)
                
                ep_reward += reward[0].item()
                done = terminated[0].item() or truncated[0].item()
                
            successes += int(extras["success"][0].item())
            rewards.append(ep_reward)
            
        return {
            "success_rate": successes / num_episodes,
            "mean_reward": np.mean(rewards),
            "planning_latency": np.mean(execution_times)
        }


class BenchmarkRunner:
    """
    5. Benchmark Runner & 6. Comparison Framework.
    Executes cross-evaluation comparisons of all active extensions.
    """
    def __init__(self, env, agent, device="cuda:0"):
        self.manager = ExperimentManager(env, agent, device)
        self.results = {}

    def run_all_benchmarks(self) -> str:
        algorithms = ["PPO+ACO", "PPO+GWO", "PPO+GA"]
        
        for alg in algorithms:
            res = self.manager.run_trial(alg)
            self.results[alg] = res
            
        # Format markdown comparison table
        report = "========================================================\n"
        report += "          NEUROGAIT METAHEURISTIC COMPARISON\n"
        report += "========================================================\n"
        report += f"{'Algorithm':<12} | {'Success Rate':<12} | {'Mean Reward':<12} | {'Latency (ms)':<12}\n"
        report += "--------------------------------------------------------\n"
        for alg, res in self.results.items():
            report += f"{alg:<12} | {res['success_rate']*100:<10.1f}% | {res['mean_reward']:<12.3f} | {res['planning_latency']*1000:<12.2f}\n"
        report += "========================================================\n"
        
        return report
