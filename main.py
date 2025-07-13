import numpy as np
from raytracer import Raytracer
from visualization import visualize_raytracer

# Test parameters: [dimensions, start_coords, end_coords, min_grid_size]
test_parameters = [
    [2, [0.5, 0.0], [3.5, 3.5], 5],
    [2, [-3.0, -3.0], [3.0, 3.0], 5],
    [2, [0.3, 0.0], [0.2, 3.1], 5],
    [2, [0.0, 0.0], [0.0, 0.0], 5],
    
    # 3D raytracer test
    [3, [0.1, 0.0, 0.0], [5.2, 5.0, 5.3], 10],
    [3, [0.0, 0.0, 0.0], [0.0, 5.0, 5.0], 10],
    [3, [0.0, 0.0, 0.0], [0.0, 0.0, 5.0], 10],
    [3, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 10],
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