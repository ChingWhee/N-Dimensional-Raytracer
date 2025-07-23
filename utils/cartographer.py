import sys
import os
import itertools
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.raytracer import Raytracer

class Cartographer():
    def __init__(self, dimensions, start_coords, end_coords, occupancy_grid, origin, loose):
        self.all_traversed_front_cells = set()  # Store front cells discovered by the raytracer (no duplicates)

        self.raytracer = Raytracer(dimensions, start_coords, end_coords)

        self.occupancy_grid = occupancy_grid
        self.loose = loose
        self.dimensions = dimensions
        
        # Set grid origin coordinates (defaults to zero origin)
        if origin is None:
            self.origin = np.zeros(self.dimensions, dtype=int)
        else:
            self.origin = np.array(origin, dtype=int)
            if len(self.origin) != self.dimensions:
                raise ValueError(f"origin must have {self.dimensions} coordinates, got {len(self.origin)}")

    def map(self):
        previous_cells = None  # Track previous front cells
        
        while not self.raytracer.reached():
            # Get accessible front cells at current position
            current_cells = set()
            for cell in self.raytracer.front_cells():
                cell_tuple = tuple(cell)
                if self.is_accessible(cell_tuple):
                    current_cells.add(cell_tuple)

            # Check if we have any accessible front cells
            if not current_cells:
                return self._handle_raytracing_failure("No accessible front cells found.")
            
            # Filter current cells to only include those reachable from the previous front
            if previous_cells:
                current_cells = self._get_reachable_front_cells(previous_cells, current_cells)
                if not current_cells:
                    return self._handle_raytracing_failure("Current front is not reachable from the previous front.")

            self.all_traversed_front_cells.update(current_cells)
            
            # Move to next grid crossing
            if not self.raytracer.next():
                break
            
            # Update previous cells for next iteration
            previous_cells = current_cells
        
        # Only process final cells if we haven't already reached the goal during the loop
        if not self.raytracer.reached():
            # Handle final cells at the goal position
            # Handle final accessible cells at the goal position
            final_cells = set()
            for cell in self.raytracer.front_cells():
                cell_tuple = tuple(cell)
                if self.is_accessible(cell_tuple):
                    final_cells.add(cell_tuple)

            # Check if we have any final cells
            if not final_cells:
                return self._handle_raytracing_failure("No accessible final front cells found.")

            # Check reachability from the last set of front cells
            if previous_cells:
                final_cells = self._get_reachable_front_cells(previous_cells, final_cells)
                if not final_cells:
                    return self._handle_raytracing_failure("Final front is not reachable from the previous front.")
            
            self.all_traversed_front_cells.update(final_cells)
        
        print(f"Raytracing completed. Total traversed front cells: {self.all_traversed_front_cells}")
        # Return success result with traversed cells
        return {
            'success': True,
            'traversed_front_cells': list(self.all_traversed_front_cells),
            'raytracer_position': {
                'current_coords': self.raytracer.coords().tolist(),
                'parametric_position': self.raytracer.t,
                'reached_goal': self.raytracer.reached()
            }
        }    
    
    def _handle_raytracing_failure(self, error_message):
        return {
            'success': False,
            'error': error_message,
            'traversed_front_cells': list(self.all_traversed_front_cells),
            'raytracer_position': {
                'current_coords': self.raytracer.coords().tolist(),
                'parametric_position': self.raytracer.t,
                'reached_goal': self.raytracer.reached()
            }
        }

    def _get_reachable_front_cells(self, previous_cells, current_cells):
        """
        Finds all cells in current_cells that are reachable from any cell in previous_cells
        using BFS within the combined bounding box. The path cannot go through other
        current_cells, only through accessible cells.
        """
        # Bounding box for the search
        dimensions = self.raytracer.dimensions
        all_cells_for_bbox = previous_cells.union(current_cells)
        min_coords = tuple(min(c[i] for c in all_cells_for_bbox) for i in range(dimensions))
        max_coords = tuple(max(c[i] for c in all_cells_for_bbox) for i in range(dimensions))

        # BFS implementation
        queue = list(previous_cells)
        visited = set(previous_cells)
        reachable_current_cells = set()

        while queue:
            cell = queue.pop(0)

            # Get neighbors based on 'loose' and bounding box
            for neighbor in self._get_neighbors(cell, min_coords, max_coords):
                if neighbor not in visited:
                    if neighbor in current_cells:
                        visited.add(neighbor)
                        reachable_current_cells.add(neighbor)
                        queue.append(neighbor)

                    elif self.is_accessible(neighbor):
                        visited.add(neighbor)
                        queue.append(neighbor)
        print(f"Reachable current cells: {reachable_current_cells} from previous cells: {previous_cells}")
        return reachable_current_cells
    
    def is_accessible(self, cell_coords):
        # Convert world coordinates to grid indices by subtracting the origin
        indices = np.array(cell_coords, dtype=int) - self.origin

        # Grid shape is typically (z, y, x), so we reverse it for (x, y, z) checks
        grid_shape_rev = self.occupancy_grid.shape[::-1]

        # Check if indices are within the grid boundaries
        if not all(0 <= idx < size for idx, size in zip(indices, grid_shape_rev)):
            return False

        # Check the occupancy status of the cell and return true if not occupied
        return self.occupancy_grid[tuple(indices[::-1])] == 0
    
    def _get_neighbors(self, cell, min_coords, max_coords):
        neighbors = []
        dimensions = len(cell)

        # Generate all possible moves: {-1, 0, 1} for each dimension
        for displacement in itertools.product([-1, 0, 1], repeat=dimensions):
            if all(d == 0 for d in displacement):
                continue

            # Check if the number of changing dimensions is within the 'loose' limit
            num_changed_dims = sum(1 for d in displacement if d != 0)
            if num_changed_dims > self.loose:
                continue

            neighbor = tuple(cell[i] + displacement[i] for i in range(dimensions))

            # Check if neighbor is within the search's bounding box
            is_within_box = all(min_coords[i] <= neighbor[i] <= max_coords[i] for i in range(dimensions))

            if is_within_box:
                neighbors.append(neighbor)
                
        return neighbors
