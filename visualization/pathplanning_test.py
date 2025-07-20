import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.pathplanning_viz import visualize_pathplanning

# 0 = free space, 1 = obstacle
def create_2d_test_grids():
    """Create 2D test grids for path planning visualization."""
    test_cases = []
    
    grid_2d_1 = np.array([
        [0, 1],  
        [1, 0],
    ], dtype=bool)

    test_cases.append([
        2, [0, 0], [1, 1], grid_2d_1, None, 1, "1-loose cell test", "cell"
    ])
    test_cases.append([
        2, [0, 0], [1, 1], grid_2d_1, None, 2, "2-loose cell test", "cell"
    ])
    test_cases.append([
        2, [0, 0], [1, 1], grid_2d_1, None, 3, "3-loose cell test", "cell"
    ])

    test_cases.append([
        2, [0, 0], [1, 1], grid_2d_1, None, 1, "1-loose cell test", "vertex"
    ])
    test_cases.append([
        2, [1, 1], [2, 2], grid_2d_1, None, 1, "1-loose cell test", "vertex"
    ])
    test_cases.append([
        2, [0, 0], [1, 1], grid_2d_1, None, 2, "2-loose cell test", "vertex"
    ])
    test_cases.append([
        2, [1, 1], [2, 2], grid_2d_1, None, 2, "2-loose cell test", "vertex"
    ])
    test_cases.append([
        2, [0, 0], [1, 1], grid_2d_1, None, 3, "3-loose cell test", "vertex"
    ])

    grid_2d_2 = np.array([
        [0, 1],  
        [1, 1],
    ], dtype=bool)

    test_cases.append([
        2, [1, 1], [2, 2], grid_2d_2, None, 1, "1-loose cell test", "vertex"
    ])

    grid_2d_3 = np.array([
        [0, 0, 0, 0],  
        [0, 1, 1, 0],  
        [1, 1, 0, 0],
    ], dtype=bool)

    # test_cases.append([
    #     2, [1, 2], [2, 3], grid_2d_3, None, 1, "1-loose vertex test", "vertex"
    # ])

    # test_cases.append([
    #     2, [1, 2], [2, 3], grid_2d_3, None, 2, "2-loose vertex test", "vertex"
    # ])


    # test_cases.append([
    #     2, [-1, -1], [1, 1], grid_2d_1, [-1, -1], 1, "2D shift origin test", "vertex"
    # ])

    # grid_2d_2 = np.array([
    #     [0, 1, 1],  # Top row: clear-obstacle-obstacle
    #     [0, 1, 0],  # Middle row: clear-obstacle-clear
    #     [1, 1, 0],  # Bottom row: obstacle-obstacle-clear  
    # ], dtype=bool)
    # test_cases.append([
    #     2, [0, 0], [2, 2], grid_2d_2, None, 1, "2D Complex maze test", "cell"
    # ])
    # test_cases.append([
    #     2, [0, 0], [2, 2], grid_2d_2, None, 2, "2D Complex maze loose=2", "vertex"
    # ])
    # test_cases.append([
    #     2, [0, 0], [2, 2], grid_2d_2, None, 3, "2D Complex maze loose=3", "cell"
    # ])
    # test_cases.append([
    #     2, [-1, -1], [1, 1], grid_2d_2, [-1, -1], 1, "2D Complex maze origin shift", "vertex"
    # ])
    
    return test_cases

def create_3d_test_grids():
    """Create 3D test grids for path planning visualization."""
    test_cases = []
    # 0 = free space, 1 = obstacle
    
    # 3D Test Cases - Simple 3D grid with obstacles - middle layer blocked
    # Each layer represents a Z-slice
    grid_3d_simple = np.array([
        # Layer 0 (Z=0) - Free space
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        # Layer 1 (Z=1) - Free space
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        # Layer 2 (Z=2) - Middle layer with central obstacle block
        [[0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0]],
        
        # Layer 3 (Z=3) - Free space
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        # Layer 4 (Z=4) - Free space
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    ], dtype=bool)
    test_cases.append([
        3, [0, 0, 0], [4, 4, 4], grid_3d_simple, None, 1, "3D Simple Obstacle", "cell"
    ])
    
    return test_cases

def create_test_grids():
    """Create all test grids for path planning visualization (combines 2D and 3D)."""
    test_cases = []
    test_cases.extend(create_2d_test_grids())
    # test_cases.extend(create_3d_test_grids())
    return test_cases

def test_pathplanning_viz():
    """Execute all test cases and display path planning visualizations."""
    test_cases = create_test_grids()
    
    for i, test_case in enumerate(test_cases):
        dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description, mode = test_case
        
        print(f"\n=== Test Case {i+1}: {description} ===")
        print(f"Dimensions: {dimensions}D")
        print(f"Grid shape: {occupancy_grid.shape}")
        print(f"Start: {start_coords}, End: {end_coords}")
        print(f"Origin: {origin}, Loose: {loose}, Mode: {mode}")
        
        # Test the specific mode for this case
        print(f"\n--- Testing {mode} mode ---")
        title = f"Test Case {i+1}: {description} ({mode} mode)"
        
        try:
            path = visualize_pathplanning(start_coords, end_coords, occupancy_grid, 
                                        origin=origin, loose=loose, algorithm='bfs', mode=mode, title=title)
            if path:
                print(f"[SUCCESS] Path found with {len(path)} steps")
            else:
                print("[FAILED] No path found")
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print("\n" + "=" * 50)
    print("All path planning visualization tests completed!")

def run_specific_test(test_index):
    """Run a specific test case by index.
    
    Args:
        test_index (int): Index of the test case to run (0-based)
    """
    test_cases = create_test_grids()
    
    if test_index < 0 or test_index >= len(test_cases):
        print(f"Error: Test index {test_index} is out of range. Available tests: 0-{len(test_cases)-1}")
        return
    
    test_case = test_cases[test_index]
    dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description, mode = test_case
    
    print(f"\n=== Test Case {test_index+1}: {description} ===")
    print(f"Dimensions: {dimensions}D")
    print(f"Grid shape: {occupancy_grid.shape}")
    print(f"Start: {start_coords}, End: {end_coords}")
    print(f"Origin: {origin}, Loose: {loose}, Mode: {mode}")
    
    print(f"\n--- Testing {mode} mode ---")
    title = f"Test Case {test_index+1}: {description} ({mode} mode)"
    
    try:
        path = visualize_pathplanning(start_coords, end_coords, occupancy_grid, 
                                    origin=origin, loose=loose, algorithm='bfs', mode=mode, title=title)
        if path:
            print(f"[SUCCESS] Path found with {len(path)} steps")
        else:
            print("[FAILED] No path found")
    except Exception as e:
        print(f"[ERROR] {e}")

def list_test_cases():
    """List all available test cases."""
    test_cases = create_test_grids()
    
    print("Available test cases:")
    print("=" * 50)
    for i, test_case in enumerate(test_cases):
        dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description, mode = test_case
        print(f"{i:2d}: {description} ({dimensions}D, {occupancy_grid.shape}, loose={loose}, mode={mode})")

if __name__ == "__main__":
    # You can run different test functions:
    # test_pathplanning_viz()           # Run all tests
    # run_specific_test(0)              # Run test case 0 in both modes
    # run_specific_test(0, 'cell')      # Run test case 0 in cell mode only
    # list_test_cases()                 # List all available test cases
    
    test_pathplanning_viz()
