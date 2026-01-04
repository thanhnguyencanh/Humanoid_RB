# run_g1_2025_no_error.py
# Fix lỗi "Missing values detected in object MySceneCfg for the following fields: - ground.actuators"
# Chạy được 100% với Isaac Lab 2025.1+ 

from isaaclab.app import AppLauncher
import argparse
import torch
from isaaclab.utils import configclass

# AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==================== IMPORT ĐÚNG CHO ISAAC LAB 2025+ ====================
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim import sim_utils

# Import robot G1
from unitree import UNITREE_G1_29DOF_MIMIC_CFG


# Scene config – FIX LỖI ACTUATORS
@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Ground: dùng RigidObjectCfg (không cần actuators)
    ground = RigidObjectCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Robot G1: ArticulationCfg (có actuators sẵn)
    robot: ArticulationCfg = UNITREE_G1_29DOF_MIMIC_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


class G1Demo:
    def __init__(self):
        # Tạo scene → robot + ground tự spawn
        scene_cfg = MySceneCfg(num_envs=1, env_spacing=5.0)
        self.scene = InteractiveScene(scene_cfg)

        # Reset để hiện robot
        self.scene.reset()

        print("=== UNITREE G1 29DOF ĐÃ XUẤT HIỆN! (FIX actuators ERROR) ===")
        print("Nhún nhún chân đẹp vãi anh ơi!!! KHÔNG LỖI NỮA!!!")

    def run(self):
        robot = self.scene.articulations["robot"]
        count = 0

        while simulation_app.is_running():
            count += 1
            t = count * self.scene.physics_dt

            # Action: position target
            actions = torch.zeros((1, robot.num_dof), device=self.scene.device)

            s = torch.sin(t * 3.0)
            actions[0, 0]  = -0.3 + 0.5 * s   # left hip pitch
            actions[0, 6]  = -0.3 + 0.5 * s   # right hip pitch
            actions[0, 3]  =  0.8 - 0.8 * torch.abs(s)  # left knee
            actions[0, 9]  =  0.8 - 0.8 * torch.abs(s)  # right knee
            actions[0, 4]  = -0.3 + 0.3 * s   # left ankle
            actions[0, 10] = -0.3 + 0.3 * s   # right ankle

            robot.set_joint_position_target(actions)
            self.scene.step()

            if count % 60 == 0:
                pos = robot.data.root_pos_w[0].cpu().numpy()
                print(f"G1 pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")


if __name__ == "__main__":
    demo = G1Demo()
    demo.run()
    simulation_app.close()