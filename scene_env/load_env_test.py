"""
Basic script to load and visualize the Warehouse environment.
This script is intended for scene inspection and layout verification only.
"""

import argparse
import sys
import os
from isaaclab.app import AppLauncher

# ============================================================================
# 1. Simulator Configuration
# ============================================================================
parser = argparse.ArgumentParser(description="Load and visualize the Warehouse scene.")
# Simplified: Removed --num_envs, defaulting to 1 for visualization
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator (Must occur before any Isaac Lab module imports)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. Post-Launch Imports
# ============================================================================
import torch
from isaaclab.scene import InteractiveScene

# Ensure the project root is in path to find custom scene configs
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the specific Warehouse configuration
from scene_env.scene_1_cfg import WarehouseSceneCfg 

# ============================================================================
# 3. Main Execution Loop
# ============================================================================
def main():
    """Initializes the scene and enters the simulation update loop."""
    
    # Define scene configuration (Fixed to 1 environment for simplicity)
    scene_cfg = WarehouseSceneCfg(num_envs=1, env_spacing=2.5)
    
    # Initialize the Interactive Scene
    # This parses the config and spawns assets into the USD stage
    scene = InteractiveScene(scene_cfg)
    
    print("\n" + "="*60)
    print("WAREHOUSE SCENE LOADED SUCCESSFULLY")
    print("Status: Visualization Mode (No Training)")
    print("="*60)
    print("\n[INFO] Control the camera in the viewport. Press Ctrl+C to exit.\n")
    
    # Define simulation time step (Default 200Hz)
    sim_dt = 0.005
    
    # Keep the simulator running
    while simulation_app.is_running():
        # Update scene data (Physics state, buffers, etc.)
        scene.update(sim_dt)
        
        # Advance the simulation step
        simulation_app.update()
    
    # Reset scene before shutdown
    scene.reset()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Shutdown requested by user.")
    finally:
        simulation_app.close()