# d:/CAPSTONE/Navigation/neurogait/envs/navigation_env.py
import torch
import numpy as np
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    class Box:
        def __init__(self, low=-np.inf, high=np.inf, shape=None):
            self.low = low
            self.high = high
            self.shape = shape


# Optional imports for IsaacLab
try:
    from omni.isaac.lab.envs import DirectRLEnv
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    # Mock DirectRLEnv for standalone testing
    class DirectRLEnv:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.num_envs = getattr(cfg, "num_envs", 10)
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,))
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(128,))
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.reward_buf = torch.zeros(self.num_envs, device=self.device)
            self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
            self.extras = {}

from neurogait.envs.locomotion.locomotion_policy import LocomotionPolicy
from neurogait.perception.occupancy import OccupancyGridPipeline
from neurogait.perception.costmap import CostMapPipeline
from neurogait.planning.planner import AStarPlanner

class NeuroGaitNavigationRubbleEnv(DirectRLEnv):
    """
    AT3, AT4: The high-level navigation environment for NeuroGait-Navigation-Rubble-v0.
    Integrates pointcloud perception, costmaps, A* path planning, and multi-rate control.
    """
    def __init__(self, cfg, **kwargs):
        if ISAAC_AVAILABLE:
            super().__init__(cfg, **kwargs)
        else:
            # Standalone mockup constructor
            self.cfg = cfg
            self.num_envs = getattr(cfg, "num_envs", 10)
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,))
            # Navigation observation space:
            # 1. Proprioceptive state: base velocities, orientation, joint feedback (48 dim)
            # 2. Exteroceptive state: Local occupancy grid + Costmap flattened (64 dim)
            # 3. Path guidance: Vector towards local A* waypoints (16 dim)
            # Total dimension: 128
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(128,))
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.reward_buf = torch.zeros(self.num_envs, device=self.device)
            self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
            self.extras = {}

        self.decimation = getattr(cfg, "decimation", 10) # Locomotion substeps per nav step
        
        # Initialize modules
        self.locomotion_policy = LocomotionPolicy()
        self.occupancy_pipeline = OccupancyGridPipeline(self.num_envs, device=self.device)
        self.costmap_pipeline = CostMapPipeline(self.num_envs, device=self.device)
        self.astar_planner = AStarPlanner()

        # Goal and robot status buffers
        self.robot_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_orientations = torch.zeros((self.num_envs, 4), device=self.device) # Quaternions
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_positions[:, 0] = 10.0 # Standard goal coordinates (10m in front)

    def step(self, action):
        """
        AT4: Environment step. Takes high-level velocity commands (v_x, v_y, omega_z) 
        and decimes execution over the low-level locomotion loop at 200 Hz.
        """
        # Convert action to torch tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

        # Decimated loop for hierarchical control
        # High level policy operates at e.g., 20 Hz, Low level locomotion policy operates at 200 Hz
        for _ in range(self.decimation):
            # 1. Fetch current proprioception inputs needed by frozen locomotion controller
            low_level_obs = self._get_proprioceptive_obs()
            
            # 2. Query frozen locomotion policy to output joint positions (12 DOF)
            joint_position_targets = self.locomotion_policy.compute_joints(low_level_obs, action)
            
            # 3. Apply torque controls in simulator
            if ISAAC_AVAILABLE:
                # self.robot.set_joint_position_targets(joint_position_targets)
                # self.sim.step()
                pass
            
        # Update simulation robot positions and orientations
        self._update_poses()
        
        # 4. Perception processing
        # Generate local grid maps from depth cameras
        depth_images = self._get_depth_images()
        occupancy_grids = self.occupancy_pipeline.process(depth_images, self.robot_positions, self.robot_orientations)
        costmaps = self.costmap_pipeline.process(occupancy_grids)
        
        # 5. Planning
        # Run A* planning over the terrain costmap to obtain reference guidance
        astar_paths = self.astar_planner.plan_batch(self.robot_positions, self.goal_positions, costmaps)

        # Assemble observations
        self.obs_buf = self._assemble_high_level_obs(occupancy_grids, costmaps, astar_paths)
        
        # Compute AT2 Reward
        self.reward_buf = self._compute_rewards(action, costmaps, astar_paths)
        
        # Reset environments if they exceed limits or collide
        self.reset_buf = self._check_resets(costmaps)
        
        # Extra stats for evaluation
        self.extras = {
            "success": (torch.norm(self.robot_positions[:, :2] - self.goal_positions[:, :2], dim=-1) < 0.5),
            "collisions": (self.reset_buf & (self.robot_positions[:, 2] > 0.1)), # Collided on rubble
            "fall": (self.robot_positions[:, 2] < 0.2) # Robot body height check
        }
        
        return self.obs_buf, self.reward_buf, self.reset_buf, self.reset_buf, self.extras

    def reset(self, seed=None, options=None):
        """Resets environments and generates initial observations."""
        self.robot_positions.zero_()
        self.goal_positions[:, 0] = 10.0
        
        depth_images = self._get_depth_images()
        occupancy_grids = self.occupancy_pipeline.process(depth_images, self.robot_positions, self.robot_orientations)
        costmaps = self.costmap_pipeline.process(occupancy_grids)
        astar_paths = self.astar_planner.plan_batch(self.robot_positions, self.goal_positions, costmaps)
        
        self.obs_buf = self._assemble_high_level_obs(occupancy_grids, costmaps, astar_paths)
        return self.obs_buf, {}

    def _get_proprioceptive_obs(self):
        """Proprioceptive details: base velocity, roll/pitch, joint positions."""
        # Flat mock array of size (num_envs, 48)
        return torch.zeros((self.num_envs, 48), device=self.device)

    def _get_depth_images(self):
        """Mock depth data."""
        return torch.zeros((self.num_envs, 1, 64, 64), device=self.device)

    def _update_poses(self):
        # Update dummy trajectories
        self.robot_positions[:, 0] += 0.05 # robot walks forward

    def _assemble_high_level_obs(self, occupancy, costmap, paths):
        """
        AT3: Observation design. Assemble high level observations.
        """
        # Flat representation: proprioceptive (48) + exteroceptive grid (64) + path guidance (16) = 128
        proprio = torch.zeros((self.num_envs, 48), device=self.device)
        extero = costmap.view(self.num_envs, -1)[:, :64]
        guidance = torch.zeros((self.num_envs, 16), device=self.device)
        
        # Populate guidance vectors
        for i in range(self.num_envs):
            path = paths[i]
            if len(path) > 1:
                waypoint = torch.tensor(path[1], device=self.device)
                vec = waypoint - self.robot_positions[i, :2]
                guidance[i, :2] = vec
                guidance[i, 2] = torch.norm(vec)
        
        return torch.cat([proprio, extero, guidance], dim=-1)

    def _compute_rewards(self, action, costmaps, paths):
        """
        AT2: Reward engineering.
        """
        # Progress towards goal
        dist_to_goal = torch.norm(self.robot_positions[:, :2] - self.goal_positions[:, :2], dim=-1)
        r_progress = -dist_to_goal
        
        # Safety/roughness penalty (costmap values under the robot foot collision radius)
        r_safety = -costmaps.mean(dim=(1, 2)) * 2.0
        
        # Action smoothing/jerk penalty
        r_action = -torch.norm(action, dim=-1) * 0.1
        
        return r_progress + r_safety + r_action

    def _check_resets(self, costmaps):
        # Falls (height < 0.15) or maximum environment limits reached
        falls = self.robot_positions[:, 2] < 0.15
        out_of_bounds = torch.norm(self.robot_positions[:, :2] - self.goal_positions[:, :2], dim=-1) > 20.0
        return falls | out_of_bounds
