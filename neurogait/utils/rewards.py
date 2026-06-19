# d:/CAPSTONE/Navigation/neurogait/utils/rewards.py
import torch

def reward_progress(env) -> torch.Tensor:
    """
    1. Mathematical Formulation:
       Measures linear progress along the goal direction vector.
       R_progress = (p_goal - p_robot) / ||p_goal - p_robot|| * v_robot / v_max
       
    2. Expected Scale:
       [0.0, 1.0] (for typical target velocities)
       
    3. Normalization Strategy:
       Divided by target speed (v_max) to bound maximum progression reward.
       
    4. TensorBoard Logging Name:
       "rewards/progress"
    """
    # Safeguard attribute extraction for IsaacLab environment types
    if hasattr(env, "goal_positions"):
        goal_pos = env.goal_positions[:, :2]
    elif hasattr(env, "commands"):
        goal_pos = env.commands[:, :2]
    else:
        # Standalone mockup fallback coordinates
        goal_pos = torch.tensor([[10.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

    # Robot positions & linear velocities
    if hasattr(env, "robot"):
        robot_pos = env.robot.data.root_pos_w[:, :2]
        robot_vel = env.robot.data.root_vel_w[:, :2] # linear velocity in world
    else:
        # Mock values for standalone testing
        robot_pos = getattr(env, "robot_positions", torch.zeros((env.num_envs, 2), device=env.device))[:, :2]
        robot_vel = torch.zeros((env.num_envs, 2), device=env.device)
        robot_vel[:, 0] = 0.5 # assumed forward movement

    # Calculate direction vector pointing from robot coordinates to goal coordinates
    to_goal = goal_pos - robot_pos
    distance = torch.norm(to_goal, dim=-1, keepdim=True)
    
    # Unit vector towards goal
    unit_to_goal = to_goal / (distance + 1e-6)
    
    # Projection of robot velocity onto unit goal vector
    velocity_proj = torch.sum(robot_vel * unit_to_goal.squeeze(-1), dim=-1)
    
    # Scale by nominal forward speed limit (e.g. 1.0 m/s)
    max_speed = 1.0
    r_progress = torch.clamp(velocity_proj / max_speed, min=-1.0, max=1.0)
    
    return r_progress

def reward_heading(env) -> torch.Tensor:
    """
    1. Mathematical Formulation:
       Exponential penalty on alignment yaw error relative to goal vector.
       theta_goal = atan2(y_goal - y_robot, x_goal - x_robot)
       theta_error = wrap_to_pi(theta_goal - yaw_robot)
       R_heading = exp(- (theta_error^2) / (2 * sigma^2))
       
    2. Expected Scale:
       [0.0, 1.0] (strictly bounded by Exponential)
       
    3. Normalization Strategy:
       Intrinsic Gaussian bell-curve normalization via variance scaling coefficient (sigma = 0.5 rad).
       
    4. TensorBoard Logging Name:
       "rewards/heading"
    """
    if hasattr(env, "goal_positions"):
        goal_pos = env.goal_positions[:, :2]
    elif hasattr(env, "commands"):
        goal_pos = env.commands[:, :2]
    else:
        goal_pos = torch.tensor([[10.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

    if hasattr(env, "robot"):
        robot_pos = env.robot.data.root_pos_w[:, :2]
        quat = env.robot.data.root_quat_w # quaternions [w, x, y, z]
        # Vectorized conversion: quaternion to yaw angle
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    else:
        robot_pos = getattr(env, "robot_positions", torch.zeros((env.num_envs, 2), device=env.device))[:, :2]
        yaw = torch.zeros(env.num_envs, device=env.device)

    # Goal direction angle
    to_goal = goal_pos - robot_pos
    theta_goal = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    
    # Heading angle error
    theta_error = theta_goal - yaw
    
    # Wrap angle error to [-pi, pi]
    theta_error = torch.atan2(torch.sin(theta_error), torch.cos(theta_error))
    
    # Bell-curve target alignment reward (sigma = 0.5 radians)
    sigma = 0.5
    r_heading = torch.exp(-(theta_error**2) / (2.0 * sigma**2))
    
    return r_heading

def penalty_collision(env) -> torch.Tensor:
    """
    1. Mathematical Formulation:
       Binary penalty checking contact forces on unauthorized links (chassis base).
       P_collision = -1.0 if ||f_contact|| > threshold else 0.0
       
    2. Expected Scale:
       [-1.0, 0.0]
       
    3. Normalization Strategy:
       Static scale multiplier (-1.0) bounding maximum single-step collision cost.
       
    4. TensorBoard Logging Name:
       "rewards/penalty_collision"
    """
    # In IsaacLab, contact forces can be queried from the robot contact sensor or scene interface
    if hasattr(env, "scene") and "robot" in env.scene:
        # Check contact force values on base link
        contact_forces = env.scene["robot"].data.net_contact_forces_w[:, 0, :] # base link net contact force
        force_magnitudes = torch.norm(contact_forces, dim=-1)
    else:
        # Fallback simulation mockup
        force_magnitudes = torch.zeros(env.num_envs, device=env.device)
        # Mock collision if robot height gets too low
        if hasattr(env, "robot_positions"):
            low_height = env.robot_positions[:, 2] < 0.15
            force_magnitudes[low_height] = 20.0 # Force trigger

    # Threshold for counting collision force (e.g. 10.0 Newtons)
    threshold = 10.0
    collisions = force_magnitudes > threshold
    
    # Negative penalty for active collision states
    p_collision = -1.0 * collisions.float()
    
    return p_collision

def reward_smoothness(env) -> torch.Tensor:
    """
    1. Mathematical Formulation:
       Quadratic cost penalizing high-frequency control chatter (jerk command).
       R_smoothness = - ||action_t - action_{t-1}||^2
       
    2. Expected Scale:
       [-0.5, 0.0]
       
    3. Normalization Strategy:
       Clipped to restrict outlier action steps.
       
    4. TensorBoard Logging Name:
       "rewards/smoothness"
    """
    # Extract current and previous action arrays
    if hasattr(env, "action_manager"):
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
    else:
        # Mock actions for standalone testing
        action = torch.zeros((env.num_envs, 3), device=env.device)
        prev_action = torch.zeros((env.num_envs, 3), device=env.device)

    # Quadratic penalty on difference vector
    diff = action - prev_action
    r_smoothness = -torch.sum(torch.square(diff), dim=-1)
    
    # Clamp scale to prevent extreme initialization jumps
    r_smoothness = torch.clamp(r_smoothness, min=-1.0, max=0.0)
    
    return r_smoothness
