import numpy as np
from itertools import product

"""
Path planner that uses raytracing to navigate through an occupancy grid.

The planner is blocked when ALL front cells along a ray are occupied.
Movement is constrained by the 'loose' parameter which determines how many 
dimensions can move simultaneously:
- loose = 1: can only move in one dimension at a time
- loose = 2: can move in one or two dimensions simultaneously  
- loose = n: can move in up to n dimensions simultaneously
"""
class Grid:
    def __init__(self, occupancy_grid, loose=1, origin=None):
        self.occupancy_grid = occupancy_grid
        self.loose = loose
        self.dimensions = len(occupancy_grid.shape)
        
        # Set grid origin coordinates (defaults to zero origin)
        if origin is None:
            self.origin = np.zeros(self.dimensions, dtype=int)
        else:
            self.origin = np.array(origin, dtype=int)
            if len(self.origin) != self.dimensions:
                raise ValueError(f"origin must have {self.dimensions} coordinates, got {len(self.origin)}")
        
        self.num_cells = np.array(occupancy_grid.shape)  # Number of cells in each dimension
        self.num_vertices = self.num_cells + 1           # Number of vertices in each dimension
        
        # Validate inputs
        if not self._validate_inputs():
            raise ValueError("Invalid inputs provided to GridPathPlanner")
        
        self.valid_directions = self._generate_valid_directions()
    
    def _validate_inputs(self):
        if self.occupancy_grid is None:
            print("Error: No occupancy grid provided")
            return False
        
        if self.loose < 1:
            print("Error: 'loose' parameter must be >= 1")
            return False
        
        if self.loose > self.dimensions:
            print(f"Error: 'loose' parameter cannot be greater than dimensions ({self.dimensions})")
            return False
        
        if self.dimensions < 2:
            print("Error: Occupancy grid must have at least 2 dimension")
            return False
        
        return True
    
    def is_cell_occupied(self, cell_coords):
        # Convert world coordinates to grid indices
        grid_indices = np.array(cell_coords) - self.origin
        
        # Check if coordinates are within grid bounds
        for i, idx in enumerate(grid_indices):
            if idx < 0 or idx >= self.occupancy_grid.shape[i]:
                return True  # Consider out-of-bounds as occupied
        
        # Use direct indexing for top-left origin coordinate system
        # For all dimensions: coordinates (x, y, z, ...) map to array indices [..., z, y, x]
        # This maintains the visual expectation where the last coordinate corresponds to the first array dimension
        array_indices = tuple(grid_indices[::-1])  # Simply reverse coordinate order
        
        # Check occupancy
        return bool(self.occupancy_grid[array_indices])
    
    def world_to_grid(self, world_coords):
        return np.array(world_coords) - self.origin
    
    def grid_to_world(self, grid_indices):
        return np.array(grid_indices) + self.origin
    
    def get_grid_bounds(self):
        min_coords = self.origin.copy()
        max_coords = self.origin + self.num_cells - 1
        return min_coords, max_coords
    
    def is_within_grid_bounds(self, world_coords):
        grid_indices = self.world_to_grid(world_coords)
        for i, idx in enumerate(grid_indices):
            if idx < 0 or idx >= self.occupancy_grid.shape[i]:
                return False
        return True
    
    def is_within_bounds_for_mode(self, coords, mode):
        """Check if coordinates are within bounds for the specified mode."""
        if mode == 'cell':
            bounds = self.num_cells
        elif mode == 'vertex':
            bounds = self.num_vertices
        else:
            return False
        
        for i, coord in enumerate(coords):
            if not (0 <= coord < bounds[i]):
                return False
        return True

    # Get valid movement directions based on the 'loose' constraint.
    def _generate_valid_directions(self):
        directions = []
        for offset in product([-1, 0, 1], repeat=self.dimensions):
            if offset == (0,) * self.dimensions:
                continue  # Skip no-movement case

            # Check dimensionality constraint (loose)
            if 1 <= sum(1 for o in offset if o != 0) <= self.loose:
                directions.append(offset)

        return directions

