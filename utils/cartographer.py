import sys
import os
import itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grid import Grid
from utils.raytracer import Raytracer

class Cartographer():
    def __init__(self, dimensions, start_coords, end_coords, occupancy_grid, origin, loose):
        self.all_traversed_front_cells = set()  # Store front cells discovered by the raytracer (no duplicates)

        self.raytracer = Raytracer(dimensions, start_coords, end_coords)
        # write lambda to return t/f of numpy array instead of using grid 
        self.grid = Grid(occupancy_grid, loose=loose, origin=origin)

    def map(self):
        previous_cells = None  # Track previous front cells
        
        while not self.raytracer.reached():
            # Get front cells at current position 
            current_cells = set(tuple(cell) for cell in self.raytracer.front_cells())
            
            # Check if we have any front cells
            if not current_cells:
                return self._handle_raytracing_failure("No front cells found.")
            
            # Check if next step is valid, if all cells are blocked or out of bounds, raytracing failed
            if not self._has_accessible_cells(current_cells):
                return self._handle_raytracing_failure("One or more front cells are inaccessible.")
            
            # Check reachability from the last set of front cells with looseness constraint
            if previous_cells and not self._check_front_reachability(previous_cells, current_cells):
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
            final_cells = set(tuple(cell) for cell in self.raytracer.front_cells())
            
            # Check if we have any final cells
            if not final_cells:
                return self._handle_raytracing_failure("No final front cells found.")
            
            # Check final cells accessibility
            if not self._has_accessible_cells(final_cells):
                return self._handle_raytracing_failure("One or more final front cells are inaccessible.")

            # Check reachability from the last set of front cells
            if previous_cells and not self._check_front_reachability(previous_cells, final_cells):
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
    
    def _has_accessible_cells(self, cells):
        for cell in cells:
            if self.grid.is_within_grid_bounds(cell) and not self.grid.is_cell_occupied(cell):
                return True
        return False
    
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

    def _check_front_reachability(self, previous_cells, current_cells):
        """
        Checks if any cell in current_cells is reachable from any cell in previous_cells
        using BFS within the combined bounding box.
        """
        # Bounding box for the search
        dimensions = self.raytracer.dimensions
        all_cells_for_bbox = previous_cells.union(current_cells)
        min_coords = tuple(min(c[i] for c in all_cells_for_bbox) for i in range(dimensions))
        max_coords = tuple(max(c[i] for c in all_cells_for_bbox) for i in range(dimensions))

        # BFS implementation
        queue = list(previous_cells)
        visited = set(previous_cells)

        while queue:
            cell = queue.pop(0)

            if cell in current_cells:
                return True  # Reachability confirmed

            # Get neighbors based on 'loose' and bounding box
            for neighbor in self._get_neighbors(cell, min_coords, max_coords):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Path can only go through accessible cells (or the target cells)
                    if neighbor in current_cells or (self.grid.is_within_grid_bounds(neighbor) and not self.grid.is_cell_occupied(neighbor)):
                        queue.append(neighbor)
        
        return False # No path found

    def _get_neighbors(self, cell, min_coords, max_coords):
        """
        Get valid neighboring cells based on 'loose' connectivity and a bounding box.
        """
        neighbors = []
        dimensions = len(cell)

        # Generate all possible moves: {-1, 0, 1} for each dimension
        for displacement in itertools.product([-1, 0, 1], repeat=dimensions):
            if all(d == 0 for d in displacement):
                continue

            # Check if the number of changing dimensions is within the 'loose' limit
            num_changed_dims = sum(1 for d in displacement if d != 0)
            if num_changed_dims > self.grid.loose:
                continue

            # Use numpy instead of tuple
            neighbor = tuple(cell[i] + displacement[i] for i in range(dimensions))

            # Check if neighbor is within the search's bounding box
            is_within_box = all(min_coords[i] <= neighbor[i] <= max_coords[i] for i in range(dimensions))

            if is_within_box:
                neighbors.append(neighbor)
                
        return neighbors
