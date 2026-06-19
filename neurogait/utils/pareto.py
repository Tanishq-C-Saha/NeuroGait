# d:/CAPSTONE/Navigation/neurogait/utils/pareto.py
import numpy as np

class ParetoFrontOptimizer:
    """
    AT8: Pareto Optimization.
    Manages non-dominated solutions trade-off evaluation (e.g. Speed vs Safety vs COT).
    """
    def __init__(self):
        pass

    @staticmethod
    def identify_pareto(objectives: np.ndarray) -> np.ndarray:
        """
        Finds the Pareto-optimal points.
        
        objectives: (num_points, num_objectives) numpy array.
                    Assumes ALL objectives are to be MAXIMIZED (e.g. velocity, -risk, -COT).
        Returns a boolean mask of indices that lie on the Pareto front.
        """
        num_points = objectives.shape[0]
        pareto_mask = np.ones(num_points, dtype=bool)
        
        for i in range(num_points):
            for j in range(num_points):
                # Check if point j dominates point i
                # Point j dominates point i if all objectives of j are >= i and at least one is > i
                if i != j:
                    if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                        pareto_mask[i] = False
                        break
                        
        return pareto_mask

    def adapt_reward_weights(self, metrics: dict) -> dict:
        """
        Dynamically adjusts task reward weights based on objective violations.
        For example, if risk starts spiking, increase safety weight.
        """
        safety_violation = metrics.get("collision_rate", 0.0)
        
        # Default weight allocations
        w_progress = 1.0
        w_safety = 2.0
        
        if safety_violation > 0.15:
            # Shift weight towards conservative obstacle avoidance
            w_safety *= 1.5
            w_progress *= 0.8
            
        return {
            "w_progress": w_progress,
            "w_safety": w_safety
        }
