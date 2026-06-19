# d:/CAPSTONE/Navigation/neurogait/envs/locomotion/locomotion_policy.py
import torch
import os

class LocomotionPolicy:
    """
    1. Frozen Locomotion PPO Policy interface.
    Loads pre-trained rsl_rl model. Outputs 12 joint position targets.
    
    WARNING: THIS COMPONENT MUST NEVER BE MODIFIED.
    """
    def __init__(self, checkpoint_path=None, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dof = 12
        
        # Load the frozen model weights here (e.g. rsl_rl .pt format)
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.model = torch.jit.load(checkpoint_path, map_location=device)
            self.model.eval()
            print(f"Loaded frozen locomotion policy from: {checkpoint_path}")
        else:
            self.model = None
            print("[Warning] No locomotion checkpoint found. Running locomotion in pass-through model mode.")

    def compute_joints(self, low_level_obs: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
        """
        Computes joint position commands at 200 Hz.
        
        low_level_obs: low-level proprioceptive vectors (shape: num_envs, 48)
        target_velocity: navigation velocities (shape: num_envs, 3) -> [v_x, v_y, omega_z]
        """
        # If frozen model exists, evaluate it
        if self.model is not None:
            # Combine low-level state and velocities to feed to PPO locomotion policy
            input_tensor = torch.cat([low_level_obs, target_velocity], dim=-1)
            with torch.no_grad():
                joint_targets = self.model(input_tensor)
            return joint_targets
        else:
            # Standalone fallback output: nominal joint angles for Go2 stand stance
            num_envs = low_level_obs.shape[0]
            
            # Unitree Go2 nominal joint angles (radians): hip, thigh, calf
            stance_angles = torch.tensor([0.0, 0.9, -1.8], device=self.device).repeat(4) # 12 dof stance
            
            # Simple oscilation based on target_velocity to simulate joint actions
            osc = torch.sin(torch.arange(num_envs, device=self.device) * 0.1).view(-1, 1).repeat(1, self.dof)
            return stance_angles.repeat(num_envs, 1) + osc * target_velocity[:, 0].view(-1, 1) * 0.2
