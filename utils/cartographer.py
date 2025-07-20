import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grid import Grid
from utils.raytracer import Raytracer

class Cartographer():
    def __init__(self, dimensions, start_coords, end_coords, occupancy_grid, origin, loose):
        self.all_traversed_front_cells = set()  # Store front cells discovered by the raytracer (no duplicates)
        self.all_expanded_cells = set()  # Store all cells (front cells + bounding box cells)

        self.raytracer = Raytracer(dimensions, start_coords, end_coords)
        self.grid = Grid(occupancy_grid, loose=loose, origin=origin)

    def map(self):
        previous_cells = None  # Track previous front cells
        
        while not self.raytracer.reached():
            # Get front cells at current position 
            current_cells = set(tuple(cell) for cell in self.raytracer.front_cells())
            
            # Check if we have any front cells
            if not current_cells:
                return self._handle_raytracing_failure()
            
            # Check if next step is valid, if all cells are blocked or out of bounds, raytracing failed
            if not self._has_accessible_cells(current_cells):
                return self._handle_raytracing_failure()
            
            accessible_current_cells = self._filter_accessible_cells(current_cells)
            self.all_traversed_front_cells.update(accessible_current_cells)
            self.all_expanded_cells.update(accessible_current_cells) 
            
            # If we have both previous and current cells, do further processing
            if previous_cells is not None and current_cells:
                self._process_cell_transition(previous_cells, current_cells)
            
            # Move to next grid crossing
            if not self.raytracer.next():
                break
            
            # Update previous cells for next iteration
            previous_cells = current_cells
        
        # Only process final cells if we haven't already reached the goal during the loop
        if not self.raytracer.reached():
            # Handle final cells at the goal position
            final_front_cells = self.raytracer.front_cells()
            final_cells = set(tuple(cell) for cell in final_front_cells)
            
            # Check if we have any final cells
            if not final_cells:
                return self._handle_raytracing_failure()
            
            # Check final cells accessibility
            if not self._has_accessible_cells(final_cells):
                return self._handle_raytracing_failure()
            
            accessible_final_cells = self._filter_accessible_cells(final_cells)
            self.all_traversed_front_cells.update(accessible_final_cells)
            self.all_expanded_cells.update(accessible_final_cells)  # Add final front cells to combined set
            
            # Process final transition if we have previous cells
            if previous_cells is not None and final_cells:
                self._process_cell_transition(previous_cells, final_cells)
        
        print(f"Raytracing completed. Total traversed front cells: {self.all_traversed_front_cells}")
        print(f"Total expanded cells (including bounding boxes): {self.all_expanded_cells}")
        # Return success result with traversed cells
        return {
            'success': True,
            'traversed_front_cells': list(self.all_traversed_front_cells),
            'additional_expanded_cells': list(self.all_expanded_cells - self.all_traversed_front_cells),
            'all_expanded_cells': list(self.all_expanded_cells),
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
    
    def _filter_accessible_cells(self, cells):
        """Filter cells to only include those that are within bounds and not occupied."""
        accessible_cells = set()
        for cell in cells:
            if self.grid.is_within_grid_bounds(cell) and not self.grid.is_cell_occupied(cell):
                accessible_cells.add(cell)
        return accessible_cells
    
    def _handle_raytracing_failure(self):
        return {
            'success': False,
            'error': 'Front cells are inaccessible',
            'traversed_front_cells': list(self.all_traversed_front_cells),
            'additional_expanded_cells': list(self.all_expanded_cells - self.all_traversed_front_cells),
            'all_expanded_cells': list(self.all_expanded_cells),
            'raytracer_position': {
                'current_coords': self.raytracer.coords().tolist(),
                'parametric_position': self.raytracer.t,
                'reached_goal': self.raytracer.reached()
            }
        }
    
    def _process_cell_transition(self, previous_cells, current_cells):
        # Find bounding box cells between previous and current front cells
        bounding_box_cells = self._find_bounding_box_cells(previous_cells, current_cells)
        
        # Filter to only include accessible cells and add to all_expanded_cells
        for cell in bounding_box_cells:
            if self.grid.is_within_grid_bounds(cell) and not self.grid.is_cell_occupied(cell):
                self.all_expanded_cells.add(cell)
    
    def _find_bounding_box_cells(self, previous_cells, current_cells):
        all_bounding_box_cells = set()
        
        # Create bounding boxes between each previous cell and each current cell
        for prev_cell in previous_cells:
            for curr_cell in current_cells:
                # Get cells in the bounding box between these two cells
                box_cells = self._get_cells_in_bounding_box(prev_cell, curr_cell)
                all_bounding_box_cells.update(box_cells)
        
        return list(all_bounding_box_cells)
    
    def _get_cells_in_bounding_box(self, cell1, cell2):
        if len(cell1) != len(cell2):
            raise ValueError("Cells must have the same dimensionality")
        
        dimensions = len(cell1)
        
        # Find min and max coordinates for each dimension
        min_coords = []
        max_coords = []
        for i in range(dimensions):
            min_coords.append(min(cell1[i], cell2[i]))
            max_coords.append(max(cell1[i], cell2[i]))
        
        # Generate all combinations of coordinates within the bounding box
        bounding_box_cells = set()
        self._generate_box_cells_recursive(min_coords, max_coords, [], 0, bounding_box_cells)
        
        return bounding_box_cells
    
    def _generate_box_cells_recursive(self, min_coords, max_coords, current_cell, dim, result_set):
        if dim == len(min_coords):
            # We've processed all dimensions, add the complete cell
            result_set.add(tuple(current_cell))
            return
        
        # For current dimension, iterate through all integer coordinates in range
        for coord in range(int(min_coords[dim]), int(max_coords[dim]) + 1):
            current_cell.append(coord)
            self._generate_box_cells_recursive(min_coords, max_coords, current_cell, dim + 1, result_set)
            current_cell.pop()  # Backtrack
