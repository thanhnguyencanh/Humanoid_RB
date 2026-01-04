# ~/isaaclab_custom/tasks/unitree/g1/manual/unitree_g1_manual_env_cfg.py
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils

# Import robot G1 ngon nhất
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG


@configclass
class G1ManualSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = UNITREE_G1_29DOF_MIMIC_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_G1_29DOF_MIMIC_CFG.InitialStateCfg(
            pos=(0.0, 0.0, 0.9),
            joint_pos={
                ".*_hip_pitch_joint": -0.2,
                ".*_knee_joint": 0.4,
                ".*_ankle_pitch_joint": -0.2,
                ".*_shoulder_pitch_joint": 0.4,
                "left_shoulder_roll_joint": 0.25,
                "right_shoulder_roll_joint": -0.25,
                ".*_elbow_joint": 1.0,
                ".*_wrist_.*": 0.0,
            },
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )


@configclass
class G1ManualEnvCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = G1ManualSceneCfg(num_envs=1, env_spacing=5.0)

    def __post_init__(self):
        # Cài đặt viewer đẹp
        self.viewer.eye = [5.0, 0.0, 3.0]
        self.viewer.lookat = [0.0, 0.0, 1.0]
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = None  
        self.observations = None
        self.actions = None
        self.commands = None
        self.rewards = None
        self.terminations = None
        self.curriculum = None
        self.events = None