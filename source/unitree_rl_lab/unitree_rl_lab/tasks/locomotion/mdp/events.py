import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

def reset_two_robots_opposing(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Đặt 2 robot ở 2 vị trí khác nhau khi num_envs=2"""

    robot: Articulation = env.scene[asset_cfg.name]

    # Lấy root state
    root_state = robot.data.default_root_state[env_ids].clone()

    for i, env_id in enumerate(env_ids):
        if env_id == 0:
            root_state[i, 0:3] = torch.tensor([-1.0, 0.0, 0.82], device=root_state.device)
            root_state[i, 3:7] = torch.tensor([1, 0, 0, 0], device=root_state.device)
        elif env_id == 1:
            root_state[i, 0:3] = torch.tensor([2.0, 0.0, 0.82], device=root_state.device)
            root_state[i, 3:7] = torch.tensor([0, 0, 0, 1], device=root_state.device)

    robot.write_root_state_to_sim(root_state, env_ids)
def custom_initial_opposing_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """
    Custom event term: Set initial pose cho 2 robot đứng đối diện nhau khi num_envs == 2.
    Mode: startup
    """
    asset = env.scene[asset_cfg.name]

    num_envs = env.num_envs
    device = asset.device

    if num_envs == 2:
        # Vị trí
        positions = torch.zeros((2, 3), device=device)
        positions[0, 0] = -1.5   # env 0: bên trái
        positions[1, 0] = 1.5    # env 1: bên phải
        positions[:, 2] = 0.78   # height đứng

        # Quaternion wxyz
        quats = torch.zeros((2, 4), device=device)
        quats[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # yaw 0°
        quats[1] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # yaw 180°

        root_pose = torch.cat([positions, quats], dim=1)

        # Ở startup, env_ids là tất cả env (thường là torch.arange(num_envs))
        # Nhưng để an toàn, vẫn index theo env_ids nếu cần
        asset.write_root_pose_to_sim(root_pose[env_ids])
  


 