# d:/CAPSTONE/Navigation/neurogait/utils/metrics.py
import torch
import numpy as np

class MetricsCollector:
    """
    AT5: Evaluation Metrics.
    Tracks and summarizes training/evaluation runs for research publications.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.success_count = 0
        self.collision_count = 0
        self.episode_lengths = []
        self.path_lengths = []
        self.energy_expenditures = []
        self.jerks = []
        self.risk_prediction_errors = []

    def update_episode(self, success: bool, collision: bool, steps: int, trajectory: np.ndarray, actions: np.ndarray, risk_errors: list):
        """
        Record final results of a finished rollout episode.
        
        trajectory: (N, 3) robot positions
        actions: (N, 3) linear and angular velocities
        """
        self.success_count += int(success)
        self.collision_count += int(collision)
        self.episode_lengths.append(steps)
        
        # Calculate Path Length
        if len(trajectory) > 1:
            diffs = np.diff(trajectory[:, :2], axis=0)
            path_len = np.sum(np.sqrt(np.sum(diffs**2, axis=-1)))
            self.path_lengths.append(path_len)
        else:
            self.path_lengths.append(0.0)

        # Estimate Jerk (derivative of acceleration of action commands)
        if len(actions) > 2:
            accel = np.diff(actions, axis=0)
            jerk = np.mean(np.square(np.diff(accel, axis=0)))
            self.jerks.append(jerk)
        else:
            self.jerks.append(0.0)

        # Estimate Cost of Transport (COT = Energy / (Mass * gravity * distance))
        # Approximating energy as sum of squared action magnitudes
        mass = 15.0 # Go2 mass
        g = 9.81
        dist = self.path_lengths[-1]
        
        if dist > 0.1:
            energy = np.sum(np.square(actions)) * 0.1 # proxy for torque work done
            cot = energy / (mass * g * dist)
            self.energy_expenditures.append(cot)
        else:
            self.energy_expenditures.append(0.0)

        if len(risk_errors) > 0:
            self.risk_prediction_errors.append(np.mean(risk_errors))

    def get_summary(self) -> dict:
        total_episodes = max(1, len(self.episode_lengths))
        return {
            "success_rate": self.success_count / total_episodes,
            "collision_rate": self.collision_count / total_episodes,
            "mean_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "mean_path_length": np.mean(self.path_lengths) if self.path_lengths else 0.0,
            "mean_cost_of_transport": np.mean(self.energy_expenditures) if self.energy_expenditures else 0.0,
            "mean_jerk": np.mean(self.jerks) if self.jerks else 0.0,
            "mean_risk_error": np.mean(self.risk_prediction_errors) if self.risk_prediction_errors else 0.0
        }
