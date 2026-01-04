# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)




## Project Structure

```
unitree_isaac/
├── scripts/                    # Main execution scripts
│   ├── scene_1.py             # Full demo scene 1
│   ├── raising_hand.py        # Manual joint control 
│   ├── rsl_rl/                # Training & inference scripts
│   │   ├── train.py           # Train RL policy
│   │   ├── play.py            # Run trained policy
│   │   └── cli_args.py        # Command line arguments
│   ├── path_planning_algo/    # Navigation algorithms
│   │   ├── main_algo.py       # UnitreeNavigationSystem
│   │   └── core/
│   │       ├── a_star.py      # A* global planner
│   │       ├── mppi.py        # MPPI local controller
│   │       └── map.yaml       # Occupancy grid map
│   └── reasoning/             # AI reasoning module
│       ├── think.py           # REASONING class (Gemini API)
│       ├── models/            # YOLO, OSNet, Face detection
│       └── utils/             # Helper functions
├── scene_env/                 # Environment configurations
│   ├── scene_1_cfg.py         # WarehouseSceneCfg class
│   └── env_usd/               # USD assets
│       ├── env_scene_1/       # Warehouse environment
│       └── human/             # Human models (workers, intruders)
├── source/                    # Source packages
│   └── unitree_rl_lab/        # Main RL lab package
│       └── unitree_rl_lab/
│           ├── assets/        # Robot configurations
│           └── tasks/         # RL task definitions
│               └── locomotion/robots/g1/29dof/
│                   └── velocity_env_cfg.py  # Training config
├── logs/                      # Training logs & checkpoints
│   └── rsl_rl/unitree_g1_29dof_velocity/
│       └── model_9200.pt      # Trained policy checkpoint
└── unitree_model/             # Robot URDF/USD models
    └── G1/29dof/
```

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/dsc-labs/DSC-Humanoid
    ```
- Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    cd DSC-Humanoid
    conda create -n myenv python=3.11
    conda activate dsc-humanoid
    pip install -r requirements.txt
    ./dsc_unitree.sh -i
    # restart your shell to activate the environment changes.
    ```

- Verify that the environments are correctly installed by:

    ```bash
    ./dsc_unitree.sh -l #  Listing the available tasks
    ```

   

## Quick Start

### 1. Training RL Policy
```bash
cd DSC-Humanoid
./dsc_unitree.sh -t --task Unitree-G1-29dof-Velocity
```

### 2. Test Trained Policy
```bash
cd DSC-Humanoid
./dsc_unitree.sh -p --task Unitree-G1-29dof-Velocity --enable_cameras
```

### 3. Demo 
#### Patrol
In this scenario, a humanoid-robot security will automatically patrol around the environment. When humanoid approached the sub goal, it will stop and look around.
```bash
cd DSC-Humanoid
python scripts/patrol_scene.py --task Unitree-G1-29dof-Velocity --enable_cameras
```

#### Scene 1
In this scenario, a humanoid robot security officer detects a person, approaches them, and proceeds with the identification process.

```bash
cd DSC-Humanoid
python scripts/scene_1.py --task Unitree-G1-29dof-Velocity --enable_cameras
```

#### Scene 2


#### Scene 3


### Video Demo

[Google Drive Folder](https://drive.google.com/drive/folders/1vk9eKnO36aTkRKzGAkk2hCIhYoLtvHiN)

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
