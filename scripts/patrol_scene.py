"""
Deployment script for a trained RL policy integrated with the Unitree Navigation System.
Features:
- Global path planning via A*
- Local control and trajectory following via MPPI
- Humanoid locomotion via RSL_RL (PPO) policy
- Integrated Warehouse Scene with dynamic obstacles
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from importlib.metadata import version
from scipy.spatial.transform import Rotation as R

# Isaac Lab core imports
from isaaclab.app import AppLauncher
from scripts.rsl_rl import cli_args  # Isaac Lab command line argument helper

# ============================================================================
# 1. Argument Configuration & Simulator Launch
# ============================================================================
parser = argparse.ArgumentParser(description="Deploy RL Policy with Unitree Navigation System.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulation environments.")
parser.add_argument("--task", type=str, default=None, help="Registered task name.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run simulation in real-time.")

# Add RSL_RL and Isaac Lab specific arguments
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator (Must be done before importing any Gym/Isaac modules)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. Post-Launch Imports
# ============================================================================
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# Setup paths for custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Path for WarehouseSceneCfg
sys.path.insert(0, os.path.join(parent_dir, "scene_env"))
from scene_1_cfg import WarehouseSceneCfg

# Navigation algorithms
from path_planning_algo.core.a_star import AStarPlanner
from path_planning_algo.core.mppi import G1MPPIController
from path_planning_algo.main_algo import UnitreeNavigationSystem

# ============================================================================
# UTILITY: Static Path Visualization
# ============================================================================
def visualize_static_path(map_yaml_path, rx, ry, start_xy, goal_xy, checkpoints):
    """Visualizes the 2D map and the generated A* path using Matplotlib."""
    try:
        with open(map_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        img_path = os.path.join(os.path.dirname(map_yaml_path), data['image'])
        img = plt.imread(img_path)
        
        # Coordinate transformation parameters
        h, w = img.shape[:2]
        res = data['resolution']
        origin = data['origin']
        extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]

        plt.figure(figsize=(10, 8))
        plt.imshow(np.flipud(img), cmap='gray', extent=extent, origin='lower')
        
        # Plot markers
        cps_x, cps_y = zip(*checkpoints)
        plt.plot(cps_x, cps_y, 'y*', markersize=12, label='Patrol Checkpoints')
        plt.plot(rx, ry, "-r", linewidth=2, label="A* Planned Path")
        plt.plot(start_xy[0], start_xy[1], "go", markersize=10, label="Robot Start")
        plt.plot(goal_xy[0], goal_xy[1], "bx", markersize=10, label="Next Goal")
        
        plt.title(f"Navigation Ready | Target: {goal_xy}")
        plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        
        print("\n[INFO] Displaying static map. Close the window to start robot execution...")
        plt.show()
    except Exception as e:
        print(f"[WARNING] Static visualization failed: {e}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    # --- 1. Environment Setup ---
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    
    # Inject Warehouse Scene (contains static/dynamic obstacles)
    env_cfg.scene = WarehouseSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    
    # Disable velocity debug arrows for cleaner visualization
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.debug_vis = False

    # Set long episode length for uninterrupted patrol
    env_cfg.episode_length_s = 10000.0 
    
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # --- 2. Load RL Locomotion Policy ---
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    
    # Resolve checkpoint path
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint \
                  else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    print(f"[INFO] Loading RL policy: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- 3. Navigation System Initialization ---
    map_yaml_path = os.path.join(current_dir, "path_planning_algo", "core", "map.yaml")
    astar = AStarPlanner(map_yaml_path, rr=0.45) # rr: Inflation radius for obstacle avoidance
    mppi = G1MPPIController(device=env.device)   # Local trajectory optimization
    nav_system = UnitreeNavigationSystem(astar, mppi, device=env.device)

    # --- 4. Patrol Mission Logic ---
    checkpoints = [
        (1.0, -2.0), (4.8, -5.0), (7.5, 0.0), (-6.5, -1.0),
        (-6.5, 15.0), (0.0, 10.0), (0.0, 0.0)
    ]
    current_cp_idx = 0 

    # Initial State and First Plan
    env.reset()
    robot = env.unwrapped.scene["robot"]
    start_pos = robot.data.root_pos_w[0]
    start_x, start_y = start_pos[0].item(), start_pos[1].item()
    first_goal = checkpoints[current_cp_idx]

    print(f"\n[MISSION] Start: ({start_x:.2f}, {start_y:.2f}) -> Initial Goal: {first_goal}")

    if nav_system.set_goal(start_x, start_y, first_goal[0], first_goal[1]):
        rx = [p[0] for p in nav_system.global_path]
        ry = [p[1] for p in nav_system.global_path]
        visualize_static_path(map_yaml_path, rx, ry, (start_x, start_y), first_goal, checkpoints)
    else:
        print("[ERROR] Initial path planning failed.")
        return

    # --- 5. Simulation Execution Loop ---
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations() if version("rsl-rl-lib").startswith("2.3.") else (env.get_observations(), None)
    
    log_counter = 0
    print("\n" + "="*70)
    print("MISSION STATUS: ACTIVE. Robot is patrolling checkpoints.")
    print("="*70 + "\n")

    while simulation_app.is_running():
        start_loop_time = time.time()
        log_counter += 1

        # A. State Estimation (Fetch Ground Truth from Sim)
        root_pos = robot.data.root_pos_w[0]
        curr_x, curr_y = root_pos[0].item(), root_pos[1].item()
        
        # Convert Isaac Sim Quaternion [W, X, Y, Z] to Scipy format [X, Y, Z, W]
        q = robot.data.root_quat_w[0]
        r = R.from_quat([q[1].item(), q[2].item(), q[3].item(), q[0].item()])
        curr_yaw = r.as_euler('xyz')[2]

        # B. Checkpoint Sequencing Logic
        if nav_system.is_goal_reached:
            print(f"\n[SUCCESS] Reached Checkpoint {current_cp_idx + 1}!")
            
            # Loop back to start if end of list is reached
            current_cp_idx = (current_cp_idx + 1) % len(checkpoints)
            next_goal = checkpoints[current_cp_idx]
            
            print(f"[PLANNING] Routing to next goal: {next_goal}")
            nav_system.set_goal(curr_x, curr_y, next_goal[0], next_goal[1])
            log_counter = 0 

        # C. Navigation Command Computation (A* + MPPI)
        # Returns linear velocities (vx, vy) and angular velocity (omega)
        vx, vy, omega = nav_system.compute_velocity_command(curr_x, curr_y, curr_yaw)
        
        # Set command for the environment command manager
        cmd = torch.zeros(env.num_envs, 3, device=env.device)
        cmd[:, 0], cmd[:, 1], cmd[:, 2] = vx, vy, omega
        env.unwrapped.command_manager.get_command("base_velocity")[:, :3] = cmd

        # D. Debug Logging (~1Hz)
        if log_counter % 50 == 0:
            target = checkpoints[current_cp_idx]
            dist_to_goal = np.hypot(target[0] - curr_x, target[1] - curr_y)
            print(f"[Log] CP:{current_cp_idx+1}/{len(checkpoints)} | "
                  f"Pos:({curr_x:.2f}, {curr_y:.2f}) | Dist:{dist_to_goal:.2f}m | Cmd:[v={vx:.2f}, w={omega:.2f}]")

        # E. Policy Inference and Environment Step
        with torch.inference_mode():
            # Policy outputs joint position targets based on observations
            actions = policy(obs)
        
        obs, _, _, _ = env.step(actions)

        # Sync with real-time if enabled
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()