"""
Script to manually control G1 robot to raise the LEFT hand only.
Fixes: SimulationContext error and RigidBody resolution error.
"""

import argparse
import torch
import sys
from isaaclab.app import AppLauncher

# ============================================================================
# 1. Configure Arguments & Launch App
# ============================================================================
parser = argparse.ArgumentParser(description="G1 Robot Raising Left Hand Demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. Post-Launcher Imports
# ============================================================================
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

# Import G1 Robot Config
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

# ============================================================================
# 3. Scene Configuration
# ============================================================================
@configclass
class RaisingHandSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # Robot G1
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=700.0),
    )

# ============================================================================
# 4. Main Function
# ============================================================================
def main():
    # Khởi tạo Simulation Context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Khởi tạo Scene
    scene_cfg = RaisingHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    
    robot = scene["robot"]
    
    # Định nghĩa tư thế giơ tay trái (Target pose)
    # Unitree G1 joints mapping
    LEFT_HAND_RAISED_TARGET = {
        "left_shoulder_pitch_joint": -1.5, # Giơ cánh tay lên
        "left_shoulder_roll_joint": 0.3,   # Hơi đưa ra ngoài
        "left_elbow_joint": 0.5,           # Gập nhẹ khuỷu tay
        "right_shoulder_pitch_joint": 0.5, # Tay phải để xuôi
        "right_elbow_joint": 1.0,
    }

    # Reset simulator
    sim.reset()
    print("\n[INFO] G1 Robot Raising LEFT Hand only. Starting simulation loop...\n")

    sim_dt = 0.01

    while simulation_app.is_running():
        # Tạo tensor mục tiêu cho toàn bộ khớp (khởi tạo bằng vị trí default của robot)
        joint_pos_target = robot.data.default_joint_pos.clone()
        
        # Gán các giá trị từ dictionary vào đúng index của khớp
        for name, value in LEFT_HAND_RAISED_TARGET.items():
            joint_idx, _ = robot.find_joints(name)
            if len(joint_idx) > 0:
                joint_pos_target[:, joint_idx[0]] = value
        
        # Gửi lệnh điều khiển vị trí khớp
        robot.set_joint_position_target(joint_pos_target)
        
        # Cập nhật dữ liệu vào simulator
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

if __name__ == "__main__":
    main()
    simulation_app.close()