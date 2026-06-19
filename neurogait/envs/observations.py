# d:/CAPSTONE/Navigation/neurogait/envs/observations.py
import torch

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotates a vector v by the inverse of quaternion q (converts world frame to body frame).
    q: Shape (N, 4) in [w, x, y, z]
    v: Shape (N, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Conjugate of unit quaternion: [w, -x, -y, -z]
    q_xyz = torch.stack([-x, -y, -z], dim=-1)
    q_w = w.unsqueeze(-1)
    
    # Vectorized rotation using cross products
    temp = torch.cross(q_xyz, v, dim=-1) + q_w * v
    v_rot = v + 2.0 * torch.cross(q_xyz, temp, dim=-1)
    
    # Replace any NaNs that might occur from cross-product errors
    v_rot = torch.nan_to_num(v_rot, nan=0.0)
    return v_rot

def get_goal_direction(env) -> torch.Tensor:
    """
    Observation Term: goal_direction (Shape: N, 2)
    Retrieves the 2D unit vector pointing towards the goal in the robot's local body frame.
    """
    if hasattr(env, "goal_positions"):
        goal_w = env.goal_positions
    else:
        goal_w = torch.tensor([[10.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

    if hasattr(env, "robot"):
        robot_pos = env.robot.data.root_pos_w
        robot_quat = env.robot.data.root_quat_w
    else:
        robot_pos = torch.zeros((env.num_envs, 3), device=env.device)
        robot_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

    # Relative target vector in world frame
    rel_goal_w = goal_w - robot_pos
    
    # Transform to robot base frame
    rel_goal_b = quat_rotate_inverse(robot_quat, rel_goal_w)
    
    # Extract 2D direction and normalize
    direction_2d = rel_goal_b[:, :2]
    norm = torch.norm(direction_2d, dim=-1, keepdim=True)
    unit_direction = direction_2d / (norm + 1e-6)
    
    # Ensure no NaN/Inf
    return torch.nan_to_num(unit_direction, nan=0.0)

def get_goal_distance(env) -> torch.Tensor:
    """
    Observation Term: goal_distance (Shape: N, 1)
    Retrieves the scalar Euclidean distance to target.
    """
    if hasattr(env, "goal_positions"):
        goal_w = env.goal_positions
    else:
        goal_w = torch.tensor([[10.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

    if hasattr(env, "robot"):
        robot_pos = env.robot.data.root_pos_w
    else:
        robot_pos = torch.zeros((env.num_envs, 3), device=env.device)

    distance = torch.norm(goal_w[:, :2] - robot_pos[:, :2], dim=-1, keepdim=True)
    
    # Clip extreme ranges and protect against NaNs
    distance = torch.clamp(distance, min=0.0, max=25.0)
    return torch.nan_to_num(distance, nan=0.0)

def get_robot_base_velocity(env) -> torch.Tensor:
    """
    Observation Term: robot_base_velocity (Shape: N, 3)
    Retrieves local base velocities [vx, vy, yaw_rate] in the body frame.
    """
    if hasattr(env, "robot"):
        # root_vel_b contains: [vx, vy, vz, wx, wy, wz]
        vel_b = env.robot.data.root_vel_b
        base_vel = torch.stack([vel_b[:, 0], vel_b[:, 1], vel_b[:, 5]], dim=-1)
    else:
        # Standalone mockup velocities
        base_vel = torch.zeros((env.num_envs, 3), device=env.device)
        
    return torch.nan_to_num(base_vel, nan=0.0)

def get_next_three_waypoints(env) -> torch.Tensor:
    """
    Observation Term: next_three_waypoints (Shape: N, 6)
    Retrieves 2D coordinates of the next 3 path waypoints relative to robot base frame.
    """
    num_envs = env.num_envs
    waypoints_b = torch.zeros((num_envs, 6), device=env.device)
    
    # Resolve root poses
    if hasattr(env, "robot"):
        robot_pos = env.robot.data.root_pos_w
        robot_quat = env.robot.data.root_quat_w
    else:
        robot_pos = torch.zeros((num_envs, 3), device=env.device)
        robot_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(num_envs, 1)

    # Resolve planned path array
    if hasattr(env, "astar_planner") and hasattr(env, "goal_positions"):
        # Run local planner query
        paths = getattr(env, "astar_paths", None)
        if paths is not None:
            for i in range(num_envs):
                path = paths[i]
                # Extract up to 3 future steps
                coords = []
                for idx in range(1, 4):
                    if idx < len(path):
                        coords.append(path[idx])
                    else:
                        # Pad with target coordinates if path ends
                        coords.append(path[-1] if len(path) > 0 else (robot_pos[i, 0].item(), robot_pos[i, 1].item()))
                
                # Transform coordinates to body frame
                pts_w = torch.tensor([[c[0], c[1], robot_pos[i, 2].item()] for c in coords], device=env.device)
                rel_pts_w = pts_w - robot_pos[i].view(1, 3)
                # Rotate
                q_i = robot_quat[i].view(1, 4).repeat(3, 1)
                rel_pts_b = quat_rotate_inverse(q_i, rel_pts_w)
                
                # Write to flattened tensor
                waypoints_b[i] = rel_pts_b[:, :2].flatten()
                
    return torch.nan_to_num(waypoints_b, nan=0.0)

def get_nearest_obstacle_positions(env) -> torch.Tensor:
    """
    Observation Term: nearest_obstacle_positions (Shape: N, 10)
    Retrieves relative 2D positions of the 5 nearest obstacles from the local occupancy grid.
    
    Future Extension Note: This block can be swapped with a CNN/MLP latent feature vector
    produced by an exteroceptive occupancy grid encoder.
    """
    num_envs = env.num_envs
    obstacles_b = torch.zeros((num_envs, 10), device=env.device)
    
    # Simulating sector-based nearest obstacle checks or scanning costmap
    if hasattr(env, "costmaps"):
        costmaps = env.costmaps # Shape: num_envs, W, H
        W, H = costmaps.shape[1], costmaps.shape[2]
        
        # Simple extraction of top high-cost coordinate offsets
        for i in range(num_envs):
            grid = costmaps[i]
            # Find indices where terrain cost is high (obstacle > 0.5)
            high_cost_indices = (grid > 0.5).nonzero(as_tuple=False)
            
            if high_cost_indices.shape[0] > 0:
                # Calculate relative distances from center (W/2, H/2)
                center = torch.tensor([W / 2, H / 2], device=env.device)
                rel_indices = high_cost_indices.float() - center.view(1, 2)
                dists = torch.norm(rel_indices, dim=-1)
                
                # Sort and select top 5 nearest
                _, sort_idx = torch.sort(dists)
                nearest_rel = rel_indices[sort_idx[:5]]
                
                # Scale cell indices to metric distances (e.g. 0.2m per cell)
                metric_obs = nearest_rel * 0.2
                
                # Flatten and pad if fewer than 5 obstacles found
                flat_obs = metric_obs.flatten()
                limit = min(flat_obs.shape[0], 10)
                obstacles_b[i, :limit] = flat_obs[:limit]
    else:
        # Default placeholder: simulate obstacle directly ahead
        obstacles_b[:, 0] = 2.0 # 2m ahead
        
    return torch.nan_to_num(obstacles_b, nan=0.0)

class NavigationObservationGroup:
    """
    Observation Group wrapper defining observation sizes and composition.
    Directly compatible with IsaacLab manager-based configurations.
    """
    def __init__(self):
        # Maps keys to corresponding evaluation functions
        self.terms = {
            "goal_direction": get_goal_direction,
            "goal_distance": get_goal_distance,
            "robot_base_velocity": get_robot_base_velocity,
            "next_three_waypoints": get_next_three_waypoints,
            "nearest_obstacle_positions": get_nearest_obstacle_positions
        }
        
    def get_observation_size(self) -> int:
        """Returns sum of flattened sizes = 2 + 1 + 3 + 6 + 10 = 22"""
        return 22

    def get_obs(self, env) -> torch.Tensor:
        """
        Concatenates all observation terms into a unified flat tensor.
        Enforces NaN validation checks.
        """
        obs_tensors = []
        for name, func in self.terms.items():
            val = func(env)
            # Ensure correct double dimensions (N, dim)
            if len(val.shape) == 1:
                val = val.unsqueeze(-1)
            obs_tensors.append(val)
            
        flat_obs = torch.cat(obs_tensors, dim=-1)
        
        # 4. Strict Validation Checks (Crucial to prevent training crashes)
        if torch.isnan(flat_obs).any():
            # Replace remaining NaNs with 0.0
            flat_obs = torch.nan_to_num(flat_obs, nan=0.0)
        if torch.isinf(flat_obs).any():
            flat_obs = torch.clamp(flat_obs, min=-1e5, max=1e5)
            
        return flat_obs
