import numpy as np
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.grid import Grid
from utils.nodes import Nodes


# Import algorithm classes
from .bfs import BFS
# from .dfs import DFS
# from .dijkstra import Dijkstra
# from .astar import AStar
# from .gbfs import GBFS

class PathPlanner:
    """
    N-dimensional path planner that uses various algorithms to find paths through occupancy grids.
    
    Supported algorithms:
    - 'dijkstra': Dijkstra's algorithm (guaranteed shortest path)
    - 'astar': A* algorithm (heuristic-guided shortest path)
    - 'bfs': Breadth-First Search (unweighted shortest path)
    - 'dfs': Depth-First Search (finds a path, not necessarily shortest)
    - 'gbfs': Greedy best-first search (fast but not optimal)
    
    Modes:
    - 'cell': Planning from cell centers (coordinates are offset by 0.5)
    - 'vertex': Planning from vertex coordinates (integer coordinates)
    """
    def __init__(self, start_coords, end_coords, occupancy_grid, origin=None, loose=1, algorithm='bfs', mode='cell'):
        # Validate and convert coordinates to integers
        self.start_coords = np.array(start_coords, dtype=int)
        self.end_coords = np.array(end_coords, dtype=int)
        self.algorithm = algorithm.lower()
        self.mode = mode.lower()
        
        # Calculate actual raytracing coordinates based on mode
        if self.mode == 'cell':
            # Cell mode: start from cell centers (add 0.5 offset)
            self.raytracing_start = self.start_coords + 0.5
            self.raytracing_end = self.end_coords + 0.5
        elif self.mode == 'vertex':
            # Vertex mode: start from integer coordinates
            self.raytracing_start = self.start_coords.astype(float)
            self.raytracing_end = self.end_coords.astype(float)
        else:
            raise ValueError(f"Mode '{self.mode}' not supported. Use 'cell' or 'vertex'")
        
        self.grid = Grid(occupancy_grid, loose=loose, origin=origin)
        
        # Then initialize nodes based on mode
        if self.mode == 'cell':
            self.nodes = Nodes(self.grid.num_cells)
        elif self.mode == 'vertex':
            self.nodes = Nodes(self.grid.num_vertices)
        

        # Algorithm mapping
        self.algorithms = {
            # 'dijkstra': Dijkstra,
            # 'astar': AStar,
            'bfs': BFS,
            # 'dfs': DFS,
            # 'gbfs': GBFS
        }
    
    def _validate_inputs(self):
        if len(self.start_coords) != self.grid.dimensions:
            print(f"Error: start_coords must have {self.grid.dimensions} coordinates")
            return False
        
        if len(self.end_coords) != self.grid.dimensions:
            print(f"Error: end_coords must have {self.grid.dimensions} coordinates")
            return False
        
        if self.algorithm not in self.algorithms:
            print(f"Error: algorithm '{self.algorithm}' not supported. Choose from: {list(self.algorithms.keys())}")
            return False
        
        # Check integer coordinates
        if not all(isinstance(coord, (int, np.integer)) for coord in self.start_coords):
            print("Error: Start coordinates must be integers")
            return False
        
        if not all(isinstance(coord, (int, np.integer)) for coord in self.end_coords):
            print("Error: End coordinates must be integers")
            return False
        
        # Check occupancy based on mode
        if self.mode == 'cell':
            # In cell mode, check if the cell itself is occupied
            if self.grid.is_cell_occupied(self.start_coords):
                print("Error: Start cell is occupied")
                return False
            
            if self.grid.is_cell_occupied(self.end_coords):
                print("Error: End cell is occupied")
                return False
        elif self.mode == 'vertex':
            # In vertex mode, we need to check if any adjacent cells are blocking
            # For now, just check if coordinates are within bounds
            pass
        
        # Check bounds based on mode
        if not self.grid.is_within_bounds_for_mode(self.start_coords, self.mode):
            bounds_name = "cell" if self.mode == 'cell' else "vertex"
            bounds = self.grid.num_cells if self.mode == 'cell' else self.grid.num_vertices
            print(f"Error: Start coordinates {self.start_coords} are out of {bounds_name} bounds {[f'[0, {b-1}]' for b in bounds]}")
            return False
        
        return True
    
    def plan_path(self):
        # Validate inputs before planning
        if not self._validate_inputs():
            print("❌ Validation failed")
            return []
        
        print(f"Planning path using {self.algorithm.lower()} algorithm in {self.mode} mode...")
        print(f"Start: {self.start_coords}, End: {self.end_coords}")
        print(f"Raytracing: {self.raytracing_start} -> {self.raytracing_end}")
        
        # Execute the selected algorithm
        algorithm_class = self.algorithms[self.algorithm]
        algo = algorithm_class(self.grid, self.nodes, self.start_coords, self.end_coords)
        path = algo.run()
        
        if path:
            print(f"✅ Path found! Length: {len(path)} cells")
        else:
            print("❌ No path found")
        
        return path


def plan_path(start_coords, end_coords, occupancy_grid, origin=None, loose=1, algorithm='bfs', mode='cell'):
    planner = PathPlanner(start_coords, end_coords, occupancy_grid, origin, loose, algorithm, mode)
    return planner.plan_path()
