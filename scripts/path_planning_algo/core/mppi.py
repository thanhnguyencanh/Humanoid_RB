import torch
import numpy as np
from pytorch_mppi import MPPI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultis import constants as C

class G1MPPIController:
    """
    MPPI Controller for the Unitree G1 robot.
    Implements the Model Predictive Path Integral control law to navigate 
    while avoiding obstacles and tracking a dynamic segment of a global path.
    """
    
    def __init__(self, device="cuda", local_target=None, obstacles=None, global_path=None, dt=C.MPPI_DT):
        """
        Initializes the MPPI solver with kinematics and cost objectives.
        
        Parameters:
            device: Computing device ('cuda' or 'cpu').
            local_target: Current local goal coordinates.
            obstacles: List of obstacle coordinates.
            global_path: Reference points from the A* planner.
            dt: Sampling time for trajectory rollouts.
        """
        self.device = device
        self.local_target = local_target
        self.dt = dt
        
        # Move obstacles to GPU for parallel distance computation
        if obstacles is not None:
            self.obstacles = torch.tensor(obstacles, dtype=torch.float32, device=device)
        else:
            self.obstacles = None
        
        # Store the initial global path
        if global_path is not None:
            self.global_path = torch.tensor(global_path, dtype=torch.float32, device=device)
            # Initialize a short starting path segment (prevents errors if command hasn't run yet)
            self.current_path_segment = self.global_path[:20] 
        else:
            self.global_path = None
            self.current_path_segment = None
            
        # Configure the MPPI optimizer
        self.mppi = MPPI(
            self.dynamics,      # Transition model for G1 robot
            self.cost,          # Objective function to minimize
            nx=3,               # State dimension: [x, y, yaw]
            horizon=C.MPPI_HORIZON,
            num_samples=C.MPPI_NUM_SAMPLES,
            lambda_=C.MPPI_LAMBDA,
            # Control noise covariance matrix
            noise_sigma=torch.diag(torch.tensor(C.MPPI_NOISE_SIGMA, device=device)),
            # Actuator constraints (Min/Max velocities)
            u_min=torch.tensor(C.MPPI_U_MIN, device=device),
            u_max=torch.tensor(C.MPPI_U_MAX, device=device),
            device=device,
        )

    def dynamics(self, state, action):
        """
        Kinematic model for the G1 robot.
        Performs Body-to-World frame transformation.
        
        State: [x, y, yaw]
        Action: [vx, vy, omega] 
        """
        x, y, yaw = state[:, 0], state[:, 1], state[:, 2] 
        vx, vy, omega = action[:, 0], action[:, 1], action[:, 2] 
        
        # Rotation Matrix: R(yaw) * [vx, vy]
        new_x = x + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * self.dt
        new_y = y + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * self.dt
        new_yaw = yaw + omega * self.dt
        
        return torch.stack([new_x, new_y, new_yaw], dim=1)

    def cost(self, state, action):
        """
        Multi-Objective Cost Function.
        Evaluates thousands of sampled trajectories in parallel.
        """
        
        # --- 1. TARGET TRACKING COST ---
        # Calculate Euclidean distance to the local target
        dx = self.local_target[0] - state[:, 0]
        dy = self.local_target[1] - state[:, 1]
        dist_to_target = torch.sqrt(dx**2 + dy**2)
        
        # Heading error: Align robot's yaw with the vector pointing to the target
        target_heading = torch.atan2(dy, dx)
        heading_error = torch.atan2(
            torch.sin(target_heading - state[:, 2]), 
            torch.cos(target_heading - state[:, 2])
        )

        # --- 2. OBSTACLE AVOIDANCE ---
        if self.obstacles is not None:
            # Batch distance calculation to all obstacles
            dist_obs = torch.cdist(state[:, :2], self.obstacles) 
            min_dist = torch.min(dist_obs, dim=1).values 
        else:
            min_dist = torch.ones(state.shape[0], device=self.device) * 10.0
        
        obstacle_cost = torch.zeros_like(min_dist)
        
        # Hard penalty: Extreme cost if the robot enters the collision zone
        obstacle_cost += torch.where(
            min_dist < C.OBSTACLE_COLLISION_DIST,
            C.OBSTACLE_HARD_PENALTY * (C.OBSTACLE_COLLISION_DIST - min_dist) ** 2,
            torch.zeros_like(min_dist)
        )
        
        # Soft penalty: Gradual repulsion as the robot approaches obstacles
        obstacle_cost += torch.where(
            (min_dist >= C.OBSTACLE_COLLISION_DIST) & (min_dist < C.OBSTACLE_SAFE_DIST),
            C.OBSTACLE_SOFT_PENALTY * (C.OBSTACLE_SAFE_DIST - min_dist) ** 2,  
            torch.zeros_like(min_dist)
        )

        # --- 3. LOCOMOTION AND SMOOTHNESS COST ---
        # Speed reward: Encourage forward motion to reach the goal faster
        speed_reward = C.COST_WEIGHT_SPEED_REWARD * action[:, 0]
        
        # Backward cost: Penalize reverse motion
        backward_cost = C.COST_WEIGHT_BACKWARD * torch.relu(-action[:, 0]) ** 2
        
        # Rotation penalty: Smooths out sharp and jerky turns
        omega_cost = C.COST_WEIGHT_ROTATION * action[:, 2] ** 2

        # --- 4. PATH TRACKING ---
        # Penalize cross-track error (distance from the current 20-point segment)
        path_cost = torch.zeros_like(dist_to_target)
        
        if self.current_path_segment is not None and self.current_path_segment.shape[0] > 0:
            # Calculate distance only to the 20 nearest points in the look-ahead window
            dist_to_path = torch.cdist(state[:, :2], self.current_path_segment) 
            min_dist_to_path = torch.min(dist_to_path, dim=1).values 
            path_cost = C.COST_WEIGHT_PATH_TRACKING * min_dist_to_path ** 2
        
        # Weighted sum of all cost components
        return (
            C.COST_WEIGHT_DISTANCE * dist_to_target +
            C.COST_WEIGHT_HEADING * torch.abs(heading_error) +
            obstacle_cost +
            path_cost +
            speed_reward +
            backward_cost + 
            omega_cost 
        )

    def command(self, state_tensor):
        """
        Executes the MPPI optimization loop.
        Input: state_tensor [x, y, yaw] or [[x, y, yaw]]
        Output: Optimal control [vx, vy, omega] for the next time step.
        """
        if self.global_path is not None:

            if state_tensor.dim() == 1:
                curr_pos = state_tensor[:2].unsqueeze(0) 
            else:
                curr_pos = state_tensor[:, :2] # Get all batches, first two columns (x, y)
            
            # Ensure global_path and curr_pos share the same dtype and device
            if self.global_path.device != curr_pos.device:
                curr_pos = curr_pos.to(self.global_path.device)

            # Calculate distance from current position to ALL global waypoints
            dists = torch.cdist(curr_pos, self.global_path).squeeze(0)
            
            # Find the index of the closest waypoint
            closest_idx = torch.argmin(dists).item()
            
            # Slice the next 20 waypoints (look-ahead window)
            # Use min() to handle cases near the end of the path
            end_idx = min(closest_idx + 20, self.global_path.shape[0])
            
            # Update the segment used by the cost function
            if end_idx > closest_idx:
                self.current_path_segment = self.global_path[closest_idx : end_idx]
        
        # Run MPPI solver with the updated path segment
        return self.mppi.command(state_tensor)