import numpy as np
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.raytracer import Raytracer

class BFS:
    """Breadth-First Search algorithm implementation with raytracing integration."""
    def __init__(self, grid, nodes, start_coords, end_coords, mode):
        self.grid = grid
        self.nodes = nodes
        self.start_coords = np.array(start_coords, dtype=int)
        self.end_coords = np.array(end_coords, dtype=int)
        self.mode = mode
    
    def _get_neighbors(self, coords):
        """Get valid neighboring coordinates using raytracing."""
        neighbors = []
        coords_array = np.array(coords)
        
        # Convert current grid coordinates to raytracing coordinates based on mode
        if self.mode == 'vertex':
            current_raytracing = coords_array.astype(float)
        else:  # cell mode
            current_raytracing = coords_array + 0.5
        
        for direction in self.grid.valid_directions:
            neighbor = coords_array + np.array(direction)
            
            # Check if neighbor is within valid bounds for the current mode
            if not self.grid.is_within_bounds_for_mode(neighbor, self.mode):
                continue  # Skip out-of-bounds neighbors
            
            # Convert neighbor to raytracing coordinates based on mode
            if self.mode == 'vertex':
                neighbor_raytracing = neighbor.astype(float)
            else:  # cell mode
                neighbor_raytracing = neighbor + 0.5
            
            # Use raytracer to trace from current to neighbor and check accessibility
            if self._is_neighbor_accessible(current_raytracing, neighbor_raytracing):
                neighbors.append(neighbor)
        
        return neighbors
    
    def _is_neighbor_accessible(self, start_ray, end_ray):
        """Check if a neighbor is accessible by tracing the ray and checking intersected cells."""
        try:
            # Create raytracer for this specific ray
            raytracer = Raytracer(self.grid.dimensions, start_ray, end_ray)
                       
            # Get all intersected cells along the ray
            intersected_cells = raytracer.trace()
            
            # If mode is 'cell', remove the current node from intersected cells
            if self.mode == 'cell':
                current_node = tuple((start_ray - 0.5).astype(int))  # Convert current ray position to grid coordinates
                intersected_cells = [cell for cell in intersected_cells if tuple(cell) != current_node]
            
            # The neighbor is accessible if one intersected cell are accessible
            # A cell is accessible if it's within bounds and not occupied
            for cell_coords in intersected_cells:
                # Check if cell is within grid bounds
                if self._is_within_bounds(cell_coords) and not self.grid.is_cell_occupied(cell_coords):
                    return True  # Valid intersected cell found
            
            return False  # All valid intersected cells are accessible
        
        except Exception as e:
            print(f"Error in raytracing: {e}")
            return False
    
    def _is_within_bounds(self, cell_coords):
        """Check if cell coordinates are within grid bounds."""
        for i, coord in enumerate(cell_coords):
            if coord < 0 or coord >= self.grid.num_cells[i]:
                return False
        return True
    
    def run(self):
        """Run BFS algorithm to find path using Nodes."""
        start_coords = self.start_coords
        end_coords = self.end_coords
        
        # Initialize start node in the nodes container
        start_node = self.nodes.getNode(*start_coords)
        if start_node is None:
            print("Error: Cannot get start node")
            return []
        start_node.expanded = True
        
        queue = deque([start_coords])
        
        while queue:
            current = queue.popleft()
            
            if np.array_equal(current, end_coords):
                path = self._reconstruct_path_from_nodes(current)
                print(f"📍 Final path found: {path}")
                return path
            
            for neighbor in self._get_neighbors(current):
                neighbor_node = self.nodes.getNode(*neighbor)
                
                # Check if this node hasn't been expanded yet
                if neighbor_node and not neighbor_node.expanded:
                    neighbor_node.expanded = True
                    neighbor_node.parent = current
                    queue.append(neighbor)
        
        # If no complete path found, reconstruct the attempted path from the farthest explored node
        attempted_path = self._get_attempted_path()
        print(f"📍 No complete path found. Attempted path: {attempted_path}")
        return attempted_path
    
    def _reconstruct_path_from_nodes(self, end_coords):
        """Reconstruct path using the nodes' parent relationships."""
        path = []
        current = end_coords
        
        while current is not None:
            # Convert grid coordinates to actual coordinates based on mode
            if self.mode == 'vertex':
                actual_coords = np.array(current, dtype=float)
            else:  # cell mode
                actual_coords = np.array(current, dtype=float) + 0.5
            
            path.append(tuple(actual_coords))  # Convert to tuple for consistent output
            current_node = self.nodes.getNode(*current)
            if current_node is None:
                print("Error: Cannot reconstruct path - invalid node coordinates")
                return []
            current = current_node.parent
        
        return path[::-1]  # Reverse to get start->end order
    
    def _get_attempted_path(self):
        """Get the attempted path to the node closest to the goal when no complete path exists."""
        # Find the explored node closest to the end goal
        min_distance = float('inf')
        closest_node_coords = self.start_coords
        
        # Check all expanded nodes to find the one closest to the goal
        for coords_tuple, node in self.nodes.nodes.items():
            if node.expanded:
                coords = np.array(coords_tuple)
                distance = np.linalg.norm(coords - self.end_coords)
                if distance < min_distance:
                    min_distance = distance
                    closest_node_coords = coords
        
        # Reconstruct path to the closest node
        return self._reconstruct_path_from_nodes(closest_node_coords)
