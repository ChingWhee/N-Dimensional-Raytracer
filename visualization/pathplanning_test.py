import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.pathplanning_viz import visualize_pathplanning

def create_test_grids():
    """Create various test grids for path planning visualization."""
    test_cases = []
    # 0 = free space, 1 = obstacle
    
    # 2D Test Cases
    grid_2d_1 = np.array([
        [0, 0, 1],  # Top row: clear-clear-obstacle
        [0, 1, 0],  # Middle row: clear-obstacle-clear
        [1, 0, 0],  # Bottom row: obstacle-clear-clear  
    ], dtype=bool)
    test_cases.append([
        2, [0, 0], [2, 2], grid_2d_1, None, 1, "1-loose test"
    ])
    test_cases.append([
        2, [0, 0], [2, 2], grid_2d_1, None, 2, "2-loose test"
    ])
    test_cases.append([
        2, [0, 0], [2, 2], grid_2d_1, None, 3, "3-loose test"
    ])
    test_cases.append([
        2, [-1, -1], [1, 1], grid_2d_1, [-1, -1], 1, "shift origin test"
    ])

    grid_2d_2 = np.array([
        [0, 1, 1],  # Top row: clear-obstacle-obstacle
        [0, 1, 0],  # Middle row: clear-obstacle-clear
        [1, 1, 0],  # Bottom row: obstacle-obstacle-clear  
    ], dtype=bool)
    test_cases.append([
        2, [0, 0], [2, 2], grid_2d_2, None, 1, "Complex maze test"
    ])
    test_cases.append([
        2, [0, 0], [2, 2], grid_2d_2, None, 2, "Complex maze loose=2"
    ])
    test_cases.append([
        2, [0, 0], [2, 2], grid_2d_2, None, 3, "Complex maze loose=3"
    ])
    test_cases.append([
        2, [-1, -1], [1, 1], grid_2d_2, [-1, -1], 1, "Complex maze origin shift"
    ])
    
    # # 2D Grid with L-shaped obstacle
    # # Clear visual L-shape pattern
    # grid_2d_l = np.array([
    #     [0, 0, 0, 0, 0, 0],  # Row 0
    #     [0, 0, 1, 0, 0, 0],  # Row 1 - Start of vertical part
    #     [0, 0, 1, 0, 0, 0],  # Row 2
    #     [0, 0, 1, 1, 1, 0],  # Row 3 - Horizontal part of L
    #     [0, 0, 0, 0, 0, 0],  # Row 4
    #     [0, 0, 0, 0, 0, 0],  # Row 5
    # ], dtype=bool)
    # test_cases.append([
    #     2, [0, 0], [5, 5], grid_2d_l, None, 1, "2D L-shaped Obstacle"
    # ])
    
    # # 2D Maze with horizontal stripes
    # # Multiple horizontal barriers to navigate around
    # grid_2d_maze = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 0
    #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Row 1 - First barrier (gap at ends)
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 2
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Row 3 - Second barrier (gap at right)
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 4
    #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Row 5 - Third barrier (gap at ends)
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 6
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Row 7 - Fourth barrier (gap at right)
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 8
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 9
    # ], dtype=bool)
    # test_cases.append([
    #     2, [0, 0], [9, 9], grid_2d_maze, None, 1, "2D Complex Maze"
    # ])
    
    # # 3D Test Cases
    # # Simple 3D grid with obstacles - middle layer blocked
    # # 0 = free space, 1 = obstacle
    # # Each layer represents a Z-slice
    # grid_3d_simple = np.array([
    #     # Layer 0 (Z=0) - Free space
    #     [[0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0]],
        
    #     # Layer 1 (Z=1) - Free space
    #     [[0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0]],
        
    #     # Layer 2 (Z=2) - Middle layer with central obstacle block
    #     [[0, 0, 0, 0, 0],
    #      [0, 1, 1, 1, 0],
    #      [0, 1, 1, 1, 0],
    #      [0, 1, 1, 1, 0],
    #      [0, 0, 0, 0, 0]],
        
    #     # Layer 3 (Z=3) - Free space
    #     [[0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0]],
        
    #     # Layer 4 (Z=4) - Free space
    #     [[0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0]]
    # ], dtype=bool)
    # test_cases.append([
    #     3, [0, 0, 0], [4, 4, 4], grid_3d_simple, None, 1, "3D Simple Obstacle"
    # ])
    
    # # 3D Tower obstacle - vertical tower through all layers
    # grid_3d_tower = np.array([
    #     # Layer 0 (Z=0) - Tower base
    #     [[0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0]],
        
    #     # Layer 1 (Z=1) - Tower continues
    #     [[0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0]],
        
    #     # Layer 2 (Z=2) - Tower continues
    #     [[0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0]],
        
    #     # Layer 3 (Z=3) - Tower continues
    #     [[0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0]],
        
    #     # Layer 4 (Z=4) - Tower continues
    #     [[0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0]],
        
    #     # Layer 5 (Z=5) - Tower top
    #     [[0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 1, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0]]
    # ], dtype=bool)
    # test_cases.append([
    #     3, [0, 0, 0], [5, 5, 5], grid_3d_tower, None, 1, "3D Tower Obstacle"
    # ])
    
    # # 3D Tunnel - mostly blocked with tunnel through middle
    # grid_3d_tunnel = np.array([
    #     # Layer 0 (Z=0) - Blocked tunnel entrance
    #     [[1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 0, 0, 1, 1],  # Tunnel entrance
    #      [1, 1, 0, 0, 1, 1],  # Tunnel entrance
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1]],
        
    #     # Layer 1 (Z=1) - Tunnel continues
    #     [[1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 0, 0, 1, 1],  # Tunnel path
    #      [1, 1, 0, 0, 1, 1],  # Tunnel path
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1]],
        
    #     # Layer 2 (Z=2) - Tunnel continues
    #     [[1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 0, 0, 1, 1],  # Tunnel path
    #      [1, 1, 0, 0, 1, 1],  # Tunnel path
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1]],
        
    #     # Layer 3 (Z=3) - Tunnel exit
    #     [[1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 0, 0, 1, 1],  # Tunnel exit
    #      [1, 1, 0, 0, 1, 1],  # Tunnel exit
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1]],
        
    #     # Layer 4 (Z=4) - Blocked beyond tunnel
    #     [[1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1]],
        
    #     # Layer 5 (Z=5) - Blocked beyond tunnel
    #     [[1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1]]
    # ], dtype=bool)
    # test_cases.append([
    #     3, [1, 2, 0], [1, 3, 3], grid_3d_tunnel, None, 1, "3D Tunnel Navigation"
    # ])
    
    # # Example with custom origin and loose parameters
    # # 2D grid with offset origin - simple obstacle course
    # grid_2d_offset = np.array([
    #     [0, 0, 0, 0, 0],  # Row 0
    #     [0, 0, 0, 0, 0],  # Row 1
    #     [0, 1, 1, 1, 0],  # Row 2 - Horizontal wall obstacle
    #     [0, 0, 0, 0, 0],  # Row 3
    #     [0, 0, 0, 0, 0],  # Row 4
    # ], dtype=bool)
    # test_cases.append([
    #     2, [2, 1], [4, 4], grid_2d_offset, [2, 1], 1, "2D Offset Origin Test"
    # ])
    
    # # 3D grid with loose movement (can move in 2 dimensions simultaneously)
    # # Central obstacle block to test diagonal movement
    # grid_3d_loose = np.array([
    #     # Layer 0 (Z=0) - Free corners, blocked center
    #     [[0, 0, 0, 0],
    #      [0, 1, 1, 0],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0]],
        
    #     # Layer 1 (Z=1) - Central obstacle block
    #     [[0, 0, 0, 0],
    #      [0, 1, 1, 0],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0]],
        
    #     # Layer 2 (Z=2) - Central obstacle block
    #     [[0, 0, 0, 0],
    #      [0, 1, 1, 0],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0]],
        
    #     # Layer 3 (Z=3) - Free corners, blocked center
    #     [[0, 0, 0, 0],
    #      [0, 1, 1, 0],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0]]
    # ], dtype=bool)
    # test_cases.append([
    #     3, [0, 0, 0], [3, 3, 3], grid_3d_loose, None, 2, "3D Loose Movement Test"
    # ])
    
    return test_cases

def test_pathplanning_viz():
    """Execute all test cases and display path planning visualizations."""
    test_cases = create_test_grids()
    
    for i, test_case in enumerate(test_cases):
        dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description = test_case
        
        print(f"\n=== Test Case {i+1}: {description} ===")
        print(f"Dimensions: {dimensions}D")
        print(f"Grid shape: {occupancy_grid.shape}")
        print(f"Start: {start_coords}, End: {end_coords}")
        print(f"Origin: {origin}, Loose: {loose}")
        
        # Test both cell and vertex modes for each case
        for mode in ['cell', 'vertex']:
            print(f"\n--- Testing {mode} mode ---")
            title = f"Test Case {i+1}: {description} ({mode} mode)"
            
            try:
                path = visualize_pathplanning(start_coords, end_coords, occupancy_grid, 
                                            origin=origin, loose=loose, algorithm='bfs', mode=mode, title=title)
                if path:
                    print(f"✅ Success! Path found with {len(path)} steps")
                else:
                    print("❌ No path found")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 All path planning visualization tests completed!")

def run_specific_test(test_index, mode='both'):
    """Run a specific test case by index.
    
    Args:
        test_index (int): Index of the test case to run (0-based)
        mode (str): 'cell', 'vertex', or 'both' to specify which modes to test
    """
    test_cases = create_test_grids()
    
    if test_index < 0 or test_index >= len(test_cases):
        print(f"Error: Test index {test_index} is out of range. Available tests: 0-{len(test_cases)-1}")
        return
    
    test_case = test_cases[test_index]
    dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description = test_case
    
    print(f"\n=== Test Case {test_index+1}: {description} ===")
    print(f"Dimensions: {dimensions}D")
    print(f"Grid shape: {occupancy_grid.shape}")
    print(f"Start: {start_coords}, End: {end_coords}")
    print(f"Origin: {origin}, Loose: {loose}")
    
    modes_to_test = ['cell', 'vertex'] if mode == 'both' else [mode]
    
    for test_mode in modes_to_test:
        print(f"\n--- Testing {test_mode} mode ---")
        title = f"Test Case {test_index+1}: {description} ({test_mode} mode)"
        
        try:
            path = visualize_pathplanning(start_coords, end_coords, occupancy_grid, 
                                        origin=origin, loose=loose, algorithm='bfs', mode=test_mode, title=title)
            if path:
                print(f"✅ Success! Path found with {len(path)} steps")
            else:
                print("❌ No path found")
        except Exception as e:
            print(f"❌ Error: {e}")

def list_test_cases():
    """List all available test cases."""
    test_cases = create_test_grids()
    
    print("Available test cases:")
    print("=" * 50)
    for i, test_case in enumerate(test_cases):
        dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description = test_case
        print(f"{i:2d}: {description} ({dimensions}D, {occupancy_grid.shape}, loose={loose})")

if __name__ == "__main__":
    # You can run different test functions:
    # test_pathplanning_viz()           # Run all tests
    # run_specific_test(0)              # Run test case 0 in both modes
    # run_specific_test(0, 'cell')      # Run test case 0 in cell mode only
    # list_test_cases()                 # List all available test cases
    
    test_pathplanning_viz()
