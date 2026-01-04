import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

# Base directory for scene assets
SCENE_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_USD_DIR = os.path.join(SCENE_ENV_DIR, "env_usd")

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):

    # Terrain - USD warehouse environment
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=os.path.join(ENV_USD_DIR, "env_scene_1", "warehouse_with_forklifts.usd"),
        collision_group=-1,
        debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    worker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Worker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ENV_USD_DIR, "human", "worker_1", "male_adult_construction_05_new.usd"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.5, 13, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Intruder
    intruder = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Intruder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ENV_USD_DIR, "human", "intruder_1", "male_adult_construction_03.usd"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(4.0, -7.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # Robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # RGBD Camera (combines RGB + Depth)
    rgbd_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/rgbd_cam",
        update_period=0.1,  # 10 Hz
        height=640,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],  # Both RGB and Depth
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 0.5),      # front – left – up
            rot=(0.5, -0.5, 0.5, -0.5)
        ),
    )

    # Height scanner sensor
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Contact forces sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True
    )

    # Sky light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
