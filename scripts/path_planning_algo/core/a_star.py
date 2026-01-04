import math
import yaml
import os
import cv2
import numpy as np
from scipy import interpolate  # Used for B-Spline path smoothing

class AStarPlanner:
    def __init__(self, yaml_path, rr=0.37):
        """
        Initialize A* Planner.
        :param yaml_path: Path to the map metadata file (.yaml)
        :param rr: Robot radius [m] used for obstacle inflation
        """
        self.rr = rr
        self.load_map(yaml_path)
        self.motion = self.get_motion_model()

    def load_map(self, yaml_path):
        """Loads and processes the occupancy grid map from a YAML configuration."""
        print(f"Loading map from {yaml_path}...")
        
        # 1. Read metadata from YAML
        with open(yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)

        self.resolution = map_data['resolution']
        self.origin_x = map_data['origin'][0]
        self.origin_y = map_data['origin'][1]
        
        # Calculate threshold for occupancy (pixel intensity)
        thresh_value = 255 * (1 - map_data['occupied_thresh'])

        # 2. Read Map Image
        map_dir = os.path.dirname(os.path.abspath(yaml_path))
        image_path = os.path.join(map_dir, map_data['image'])
        grid_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if grid_img is None:
            raise FileNotFoundError(f"Map image not found at: {image_path}")

        # 3. Image Pre-processing
        # Flip image vertically to align with coordinate systems where (0,0) is bottom-left
        grid_img = np.flipud(grid_img) 

        self.height, self.width = grid_img.shape
        self.min_x = self.origin_x
        self.min_y = self.origin_y
        self.max_x = self.min_x + self.width * self.resolution
        self.max_y = self.min_y + self.height * self.resolution
        
        self.x_width = self.width
        self.y_width = self.height

        # 4. Obstacle Inflation
        # Convert intensity to binary mask (1 for obstacle, 0 for free)
        obstacle_mask = np.where(grid_img <= thresh_value, 1, 0).astype(np.uint8)
        
        # Convert robot radius from meters to pixels
        inflate_px = int(math.ceil(self.rr / self.resolution))
        
        if inflate_px > 0:
            # Dilate obstacles to account for robot size
            kernel = np.ones((2 * inflate_px + 1, 2 * inflate_px + 1), np.uint8)
            self.obstacle_map_grid = cv2.dilate(obstacle_mask, kernel, iterations=1)
        else:
            self.obstacle_map_grid = obstacle_mask

        # Transpose to align with (x, y) indexing
        self.obstacle_map = self.obstacle_map_grid.T > 0
        print("Map loaded successfully!")

    class Node:
        """Internal class to represent a grid cell in A* search."""
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # grid x index
            self.y = y  # grid y index
            self.cost = cost  # g-cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy, smooth=True):
        """
        Execute A* path planning.
        :param sx, sy: Start coordinates [m]
        :param gx, gy: Goal coordinates [m]
        :param smooth: Boolean, if True applies B-Spline smoothing
        :return: Final x, y coordinate lists
        """
        # Convert world coordinates [m] to grid indices
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        # Validate start and goal positions
        if not self.verify_node(start_node) or not self.verify_node(goal_node):
            print("[ERROR] Start or Goal point is in an occupied or invalid area!")
            return [], []

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        print("A* searching for path...")
        found = False
        while len(open_set) > 0:
            # Select node with minimum f-cost (g-cost + heuristic)
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # Check if goal is reached
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                found = True
                break

            # Move current node from open set to closed set
            del open_set[c_id]
            closed_set[c_id] = current

            # Expand neighbors based on motion model
            for move in self.motion:
                node = self.Node(current.x + move[0],
                                 current.y + move[1],
                                 current.cost + move[2], c_id)
                n_id = self.calc_grid_index(node)

                # Skip invalid nodes or already visited nodes
                if not self.verify_node(node) or n_id in closed_set:
                    continue

                # Update open set if a cheaper path is found
                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        if not found:
            print("[ERROR] Path not found!")
            return [], []

        # Backtrack to reconstruct the raw path
        rx, ry = self.calc_final_path(goal_node, closed_set)
        
        # --- PATH SMOOTHING ---
        if smooth and len(rx) > 3:
            print("Smoothing path with B-Spline...")
            try:
                rx, ry = self.smooth_path_spline(rx, ry)
            except Exception as e:
                print(f"[WARNING] Smoothing failed: {e}. Returning raw path.")

        return rx, ry

    def smooth_path_spline(self, rx, ry, resolution=0.1):
        """
        Smoothens the path using B-Spline interpolation.
        :param resolution: Distance between points on the smoothed path [m]
        """
        # 1. Remove duplicate points (Spline fitting requires distinct points)
        path = list(zip(rx, ry))
        unique_path = [path[0]]
        for i in range(1, len(path)):
            if math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]) > 0.01:
                unique_path.append(path[i])
        
        if len(unique_path) < 3:
            return rx, ry

        x = [p[0] for p in unique_path]
        y = [p[1] for p in unique_path]

        # 2. Compute B-Spline representation
        # k=3: Cubic Spline, s=smoothing factor (s=0 forces interpolation through all points)
        tck, u = interpolate.splprep([x, y], k=3, s=0.5) 

        # 3. Generate higher resolution points along the spline
        total_dist = sum(math.hypot(x[i+1]-x[i], y[i+1]-y[i]) for i in range(len(x)-1))
        num_points = max(int(total_dist / resolution), 10)
        u_new = np.linspace(0, 1, num_points)
        
        # 4. Evaluate spline to get new coordinates
        new_points = interpolate.splev(u_new, tck)
        
        return new_points[0].tolist(), new_points[1].tolist()

    def calc_final_path(self, goal_node, closed_set):
        """Backtracks from goal node to start node to reconstruct coordinates."""
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx[::-1], ry[::-1]

    def calc_heuristic(self, n1, n2):
        """Euclidean distance heuristic."""
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, min_pos):
        """Converts grid index to world position [m]."""
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        """Converts world position [m] to grid index."""
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        """Calculates unique 1D ID for a 2D node."""
        return node.y * self.x_width + node.x

    def verify_node(self, node):
        """Checks if a node is within bounds and not on an obstacle."""
        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    @staticmethod
    def get_motion_model():
        """
        Defines 8-way movement model.
        Format: [dx, dy, step_cost]
        """
        return [
            [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
            [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]
        ]