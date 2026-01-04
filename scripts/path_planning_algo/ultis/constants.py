# Velocity smoothing (Exponential filter for smooth motion)
# Smoothing factor (0-1): higher = smoother, lower = more responsive
VELOCITY_SMOOTH_ALPHA = 0.3  

# ============================================
# MPPI CONTROLLER PARAMETERS
# ============================================
MPPI_HORIZON = 12             # Number of look-ahead steps (prediction horizon)
MPPI_NUM_SAMPLES = 800        # Number of trajectory samples
MPPI_LAMBDA = 0.15            #  enforces stricter path following
MPPI_DT = 0.04                # Control time step (40ms = 50Hz)

# Control Limits - Max speed 2 m/s, faster rotation
# [vx_min, vy_min, omega_min]
MPPI_U_MIN = [0.0, -0.65, -1.2]    
# [vx_max, vy_max, omega_max]
MPPI_U_MAX = [2.0, 0.65, 1.2]      

# Noise Sigma for MPPI exploration
MPPI_NOISE_SIGMA = [0.30, 0.25, 0.15]  # [vx, vy, omega]

# ============================================
# OBSTACLE AVOIDANCE PARAMETERS
# ============================================
OBSTACLE_COLLISION_DIST = 0.20      # Meters - hard collision threshold (reduced)
OBSTACLE_SAFE_DIST = 0.40           # Meters - soft avoidance threshold (reduced from 0.50)
OBSTACLE_HARD_PENALTY = 1e4         # Hard collision penalty weight (reduced from 5e4)
OBSTACLE_SOFT_PENALTY = 50.0        # Soft avoidance penalty weight (reduced from 100.0)

# ============================================
# COST FUNCTION WEIGHTS
# ============================================
COST_WEIGHT_DISTANCE = 1.0          # Weight for distance to target
COST_WEIGHT_HEADING = 3.0           # Weight for heading error
COST_WEIGHT_SPEED_REWARD = -2.0     # Reward for forward velocity (negative cost)
COST_WEIGHT_BACKWARD = 1.0          # Penalty for backward motion
COST_WEIGHT_ROTATION = 0.2          # Penalty for angular velocity (yaw rate)
COST_WEIGHT_PATH_TRACKING = 18.0    # Weight for following the global path

# ============================================
# PATH PLANNING PARAMETERS (A*)
# ============================================
ASTAR_RESOLUTION = 0.1              # Grid resolution (10cm)
ASTAR_ROBOT_RADIUS = 0.10           # Robot collision radius (10cm)
PATH_SMOOTH_RESOLUTION = 0.1        # Path interpolation resolution (10cm)
PATH_SMOOTH_FACTOR = 0.3            # B-spline smoothing factor

DEBUG_MODE = True
DEBUG_PRINT_INTERVAL = 50