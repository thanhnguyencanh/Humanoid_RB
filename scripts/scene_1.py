"""
Full script to deploy RL policy with Unitree Navigation System and RGBD Camera.
Optimized for readability while maintaining original logic.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from importlib.metadata import version
from scipy.spatial.transform import Rotation as R

# Isaac Lab / RL Imports
from scripts.rsl_rl import cli_args
from isaaclab.app import AppLauncher

# ============================================================================
# 1. Initialize AppLauncher (Must run before importing other Isaac modules)
# ============================================================================
parser = argparse.ArgumentParser(description="Deploy trained RL agent with Navigation and Camera.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Task name.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
# Reasoning Checkpoints
parser.add_argument("--yolo_checkpoint", type=str, default="scripts/reasoning/yolo_checkpoints/yolo11m.pt")
parser.add_argument("--ppe_yolo_checkpoint", type=str, default="scripts/reasoning/yolo_checkpoints/best.pt")
parser.add_argument("--osnet_checkpoint", type=str, default="scripts/reasoning/osnet_checkpoints/osnet.pth")
parser.add_argument("--video_path", type=str, default="scripts/reasoning/examples/videos/intruder_1.mp4")
parser.add_argument("--database", type=str, default="scripts/reasoning/database/database.pkl")
parser.add_argument("--threshold", type=float, default=0.8)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. Imports after Simulator starts
# ============================================================================
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from scripts.reasoning.reasonScene1 import REASONING

# Path setup cho Navigation & Scene
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "scene_env"))
sys.path.append(PARENT_DIR)

from scene_1_cfg import WarehouseSceneCfg
from path_planning_algo.core.a_star import AStarPlanner
from path_planning_algo.core.mppi import G1MPPIController
from path_planning_algo.main_algo import UnitreeNavigationSystem

# ============================================================================
# 3. Helper Functions
# ============================================================================

def process_depth_image(depth_data: torch.Tensor) -> np.ndarray:
    """Convert depth data from sensor to color image for visualization."""
    depth_np = depth_data.cpu().numpy()
    depth_vis = depth_np.copy()
    
    finite_mask = np.isfinite(depth_vis)
    max_d = depth_vis[finite_mask].max() if finite_mask.any() else 10.0
    
    depth_vis[~finite_mask] = max_d
    depth_norm = ((depth_vis / max_d) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

def get_robot_yaw(robot_quat: torch.Tensor) -> float:
    """Extract Yaw from Quaternion (Isaac format: [W, X, Y, Z])."""
    q = robot_quat.cpu().numpy()
    # Convert to format [X, Y, Z, W] for Scipy
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_euler('xyz')[2]

# ============================================================================
# 4. Main Execution
# ============================================================================

def main():
    # --- A. Setup Environment ---
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=True, entry_point_key="play_env_cfg_entry_point"
    )
    env_cfg.scene = WarehouseSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    env_cfg.episode_length_s = 10000.0
    
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.debug_vis = False
        
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # --- B. Load Policy & Reasoning ---
    reasoning = REASONING(args_cli)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint \
                  else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    print(f"[INFO] Loading policy: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- C. Navigation Setup ---
    map_path = os.path.join(CURRENT_DIR, "path_planning_algo", "core", "map.yaml")
    nav_system = UnitreeNavigationSystem(
        AStarPlanner(map_path, rr=0.45), 
        G1MPPIController(device=env.device), 
        device=env.device
    )

    # --- D. Initialization ---
    target_goal = (3.0, -6.0)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    rgbd_cam = env.unwrapped.scene["rgbd_camera"]
    
    start_pos = robot.data.root_pos_w[0]
    if not nav_system.set_goal(start_pos[0].item(), start_pos[1].item(), *target_goal):
        print("[ERROR] Navigation Planning Failed."); return

    # Visualization Setup
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    rgb_display = ax1.imshow(np.zeros((640, 640, 3), dtype=np.uint8))
    depth_display = ax2.imshow(np.zeros((640, 640, 3), dtype=np.uint8))
    plt.show(block=False)

    # --- E. Main Loop ---
    run_time_counter = 0
    has_reached = False
    dt = env.unwrapped.step_dt

    print(f"\n{'-'*30}\nSTARTING MISSION TO: {target_goal}\n{'-'*30}")

    while simulation_app.is_running():
        loop_start = time.time()

        # 1. Perception Processing
        rgb_raw = rgbd_cam.data.output["rgb"][0, ..., :3]
        depth_raw = rgbd_cam.data.output["distance_to_image_plane"][0]
        
        # Convert image format
        rgb_np = (rgb_raw.cpu().numpy() * 255).astype(np.uint8) if rgb_raw.max() <= 1.0 \
                 else rgb_raw.cpu().numpy().astype(np.uint8)
        
        center_depth = depth_raw[depth_raw.shape[0]//2, depth_raw.shape[1]//2].item()

        # 2. Reasoning Logic (Original Logic)
        if center_depth <= 1.9 and run_time_counter < 1:
            run_time_counter += 1
            time.sleep(0.5) # Reduce sleep to avoid hanging simulator too long
        if not reasoning.is_reasoning:
            reasoning.reasoning(rgb_np, depth=center_depth)
        else:
            break # Exit if reasoning module requests stop

        # 3. Update Visualizer
        rgb_display.set_data(rgb_np)
        depth_display.set_data(process_depth_image(depth_raw))
        fig.canvas.draw(); fig.canvas.flush_events()
        
        # # Auto-save images every save_interval frames
        # if frame_count % save_interval == 0:
        #     rgb_path = os.path.join(output_dir, f"rgb_{frame_count:06d}.png")
        #     depth_path = os.path.join(output_dir, f"depth_{frame_count:06d}.png")
        #     combined = np.hstack([rgb_np, depth_color_rgb])
        #     combined_path = os.path.join(output_dir, f"combined_{frame_count:06d}.png")

        #     cv2.imwrite(rgb_path, rgb_np)
        #     cv2.imwrite(depth_path, depth_color_rgb)
        #     cv2.imwrite(combined_path, combined)

        #     print(f"[INFO] Frame {frame_count} - SAVED to {output_dir}/")
        #     print(f"  - RGB shape: {rgb_np.shape}, dtype: {rgb_np.dtype}, range: [{rgb_np.min():.3f}, {rgb_np.max():.3f}]")
        #     print(f"  - Depth shape: {depth_np.shape}, range: {depth_vis.min():.2f} to {depth_vis.max():.2f} m")

        # frame_count += 1

        # 4. Navigation Control
        curr_pos = robot.data.root_pos_w[0]
        curr_yaw = get_robot_yaw(robot.data.root_quat_w[0])

        if nav_system.is_goal_reached and not has_reached:
            print(f"[SUCCESS] Arrived at destination."); has_reached = True

        vx, vy, omega = nav_system.compute_velocity_command(
            curr_pos[0].item(), curr_pos[1].item(), curr_yaw
        ) if not has_reached else (0.0, 0.0, 0.0)

        # Send control command to Command Manager
        cmd_tensor = torch.tensor([[vx, vy, omega]], device=env.device)
        env.unwrapped.command_manager.get_command("base_velocity")[:, :3] = cmd_tensor

        # 5. Policy Step
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # Real-time sync
        if args_cli.real_time:
            elapsed = time.time() - loop_start
            if dt > elapsed: time.sleep(dt - elapsed)

    plt.close('all')
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()