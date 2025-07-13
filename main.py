import numpy as np
from raytracer import Raytracer
from visualization import visualize_raytracer

# Test parameters: [dimensions, start_coords, end_coords, min_grid_size]
test_parameters = [
    # Basic 2D test with negative coordinates
    [2, [0.0, 0.0], [3.0, 3.0], 5],
    
    # 3D raytracer test
    [3, [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 10],
    
    # Edge cases
    [2, [1.0, 2.5], [6.0, 2.5], 10],  # Horizontal ray
    [2, [3.5, 1.0], [3.5, 5.0], 10],  # Vertical ray
    [2, [0.0, 0.0], [4.0, 4.0], 10],  # Diagonal ray
    
    # Additional test cases
    [2, [1.2, 1.8], [5.5, 4.2], 6],   # For visualization
    [4, [0.5, 1.5, 2.5, 0.0], [3.5, 4.5, 1.5, 3.0], 15],  # 4D test
    [2, [-1.0, -1.0], [1.0, 1.0], 8], # Centered around origin
]

def main():
    """Execute all test cases"""
    for i, params in enumerate(test_parameters):
        dimensions, start_coords, end_coords, min_grid_size = params
        print(f"\n=== Test Case {i+1}: {dimensions}D Ray ===")
        print(f"Start: {start_coords}, End: {end_coords}")
        
        raytracer = Raytracer(dimensions, start_coords, end_coords, min_grid_size)
        
        if raytracer.grid is not None:
            print(f"Success! Grid shape: {raytracer.grid.shape}, Ray length: {raytracer.ray_length:.3f}")
            
            # Use the unified visualization function
            intersected_cells = visualize_raytracer(raytracer, f"Test Case {i+1}: {dimensions}D Ray")
            print(f"Intersected {len(intersected_cells)} cells")
        else:
            print("Failed to initialize raytracer")

if __name__ == "__main__":
    main()