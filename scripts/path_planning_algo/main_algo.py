import numpy as np
import torch
from ultis import constants as C

class UnitreeNavigationSystem:
    """
    Unitree Navigation System - Integrates a 
    Global Planner (A*) and a Local Controller (MPPI).
    
    Responsibilities:
    1. Receive a goal point -> Compute the A* global path.
    2. Receive robot state -> Compute optimal velocities (MPPI).
    3. Return v_x, v_y, and omega commands.
    """

    def __init__(self, astar_planner, mppi_controller, device="cuda"):
        self.astar = astar_planner
        self.mppi = mppi_controller
        self.device = device

        # Internal state
        self.global_path = []      # List of waypoints [(x, y), ...]
        self.path_idx = 0          # Current waypoint index the robot is targeting
        self.is_goal_reached = False
        self.current_goal = None   # (x, y) of the final destination

        # Threshold parameters
        self.goal_threshold = 0.2      # 20cm considered as reached goal
        self.waypoint_threshold = 0.5  # 50cm to switch to the next waypoint
        
        # Velocity Safety Limits for Unitree G1
        # [v_x_min, v_y_min, w_min]
        self.cmd_min = np.array([-0.3, -0.35, -1.0]) 
        # [v_x_max, v_y_max, w_max]
        self.cmd_max = np.array([1.5, 0.35, 1.0])

    def set_goal(self, start_x, start_y, goal_x, goal_y):
        """
        Plans a new path using A* when a new goal is received.
        """
        # Optional: Skip recalculation if the goal hasn't changed and is not yet reached
        if self.current_goal == (goal_x, goal_y) and not self.is_goal_reached:
            return True

        print(f"[Nav System] Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})")
        
        # 1. Call the A* Global Planner
        # Note: The A* class returns separate rx and ry lists
        rx, ry = self.astar.planning(start_x, start_y, goal_x, goal_y)
        
        if not rx:
            print("[Nav System] A* failed to find a path!")
            return False

        # 2. Convert to [(x,y), ...] format (A* usually returns path from Goal -> Start)
        self.global_path = list(zip(rx, ry))
        
        # 3. Reset navigation state
        self.path_idx = 0
        self.is_goal_reached = False
        self.current_goal = (goal_x, goal_y)

        # 4. Load the path into MPPI (for path-tracking cost calculations)
        path_tensor = torch.tensor(self.global_path, dtype=torch.float32, device=self.device)
        self.mppi.global_path = path_tensor
        
        print(f"[Nav System] Found path with {len(self.global_path)} waypoints.")
        return True

    def compute_velocity_command(self, curr_x, curr_y, curr_yaw):
        """
        Main function to be called within the control loop.
        Input: Current robot state.
        Output: v_forward (vx), v_lateral (vy), yaw_rate (omega)
        """
        # 0. If no path exists or goal is reached -> Stop the robot
        if not self.global_path or self.is_goal_reached:
            return 0.0, 0.0, 0.0

        # 1. Check if the final goal has been reached
        dist_to_goal = np.hypot(self.current_goal[0] - curr_x, self.current_goal[1] - curr_y)
        if dist_to_goal < self.goal_threshold:
            print(f"[Nav System] Goal Reached! Distance: {dist_to_goal:.2f}")
            self.is_goal_reached = True
            return 0.0, 0.0, 0.0

        # 2. Waypoint Switching Logic (Pure Pursuit / Follow-the-carrot)
        # Identify the current target waypoint
        if self.path_idx < len(self.global_path):
            wp_x, wp_y = self.global_path[self.path_idx]
            dist_to_wp = np.hypot(wp_x - curr_x, wp_y - curr_y)

            # If close to the current waypoint, switch to the next one
            if dist_to_wp < self.waypoint_threshold and self.path_idx < len(self.global_path) - 1:
                self.path_idx += 1
                # Update to the new target waypoint
                wp_x, wp_y = self.global_path[self.path_idx]

        # 3. Configure the local target for MPPI
        state_tensor = torch.tensor([[curr_x, curr_y, curr_yaw]], dtype=torch.float32, device=self.device)
        self.mppi.local_target = torch.tensor([wp_x, wp_y], dtype=torch.float32, device=self.device)

        # 4. Execute MPPI Optimization
        # MPPI.command returns [vx, vy, omega]
        mppi_cmd_tensor = self.mppi.command(state_tensor)
        
        # Convert to numpy array for motor control
        cmd = mppi_cmd_tensor.detach().cpu().numpy().squeeze()

        # 5. Safety Clipping
        # Ensure commands stay within safe limits to prevent humanoid instability/falls
        cmd = np.clip(cmd, self.cmd_min, self.cmd_max)

        v_forward = cmd[0]
        v_lateral = cmd[1]
        yaw_rate  = cmd[2]

        return v_forward, v_lateral, yaw_rate

    def reset(self):
        """Resets the internal navigation state"""
        self.global_path = []
        self.path_idx = 0
        self.is_goal_reached = False
        self.current_goal = None