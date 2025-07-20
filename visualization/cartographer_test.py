import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.cartographer_viz import visualize_cartographer

def create_2d_test_grids():
    test_cases = []
    
    grid_2d_01 = np.array([
        [0, 0],  
        [0, 0],
    ], dtype=bool)

    grid_2d_02 = np.array([
        [0, 1],  
        [1, 0],
    ], dtype=bool)

    test_cases.append([
        2, [0.3, 0.3], [1.6, 1.6], grid_2d_01, None, 1, "2x2 no obstacles 1"
    ])

    test_cases.append([
        2, [0.3, 0.3], [2.0, 1.6], grid_2d_01, None, 1, "2x2 no obstacles 2"
    ])

    test_cases.append([
        2, [0.0, 1.0], [2.0, 1.0], grid_2d_01, None, 1, "2x2 no obstacles (straight)"
    ])

    test_cases.append([
        2, [0.3, 0.3], [1.6, 1.6], grid_2d_02, None, 1, "2x2 with obstacles"
    ])

    test_cases.append([
        2, [0.3, 0.3], [2.0, 1.6], grid_2d_02, None, 1, "2x2 with blocking obstacles"
    ])

    grid_2d_11 = np.array([
        [0, 0, 0],  
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=bool)

    grid_2d_12 = np.array([
        [0, 0, 1],  
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=bool)

    grid_2d_13 = np.array([
        [0, 0, 1],  
        [0, 1, 1],
        [1, 1, 0],
    ], dtype=bool)

    test_cases.append([
        2, [0.3, 0.3], [2.6, 2.6], grid_2d_11, None, 1, "3x3 no obstacles 1"
    ])

    test_cases.append([
        2, [0.3, 0.3], [2.6, 1.6], grid_2d_11, None, 1, "3x3 no obstacles 2"
    ])

    test_cases.append([
        2, [1.5, 0.5], [2.5, 1.5], grid_2d_12, None, 1, "3x3 obstacles (not blocked)"
    ])

    test_cases.append([
        2, [0.0, 0.0], [2.0, 3.0], grid_2d_12, None, 1, "3x3 obstacles (blocked)"
    ])

    test_cases.append([
        2, [2.0, 0.0], [2.0, 3.0], grid_2d_12, None, 1, "3x3 obstacles (straight)"
    ])

    test_cases.append([
        2, [2.0, 0.0], [2.0, 3.0], grid_2d_13, None, 1, "3x3 obstacles (straight & blocked)"
    ])
    
    return test_cases

def create_3d_test_grids():
    test_cases = []
    
    grid_3d_01 = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    ], dtype=bool)
    
    test_cases.append([
        3, [0.0, 0.0, 0.0], [3.0, 3.0, 3.0], grid_3d_01, None, 1, "3D no obstacle diagonal"
    ])

    test_cases.append([
        3, [1.0, 1.0, 0.0], [1.0, 1.0, 3.0], grid_3d_01, None, 1, "3D no obstacle straight"
    ])

    grid_3d_02 = np.array([
        [[0, 1],
         [1, 1]],
        
        [[1, 1],
         [1, 0]],
    ], dtype=bool)

    test_cases.append([
        3, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], grid_3d_02, None, 1, "3D obstacle (not blocked)"
    ])

    test_cases.append([
        3, [0.0, 0.0, 0.0], [2.0, 1.6, 1.8], grid_3d_02, None, 1, "3D obstacle (blocked)"
    ])

    grid_3d_10 = np.array([       
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],
    ], dtype=bool)

    grid_3d_11 = np.array([       
        [[0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0]],
    ], dtype=bool)

    test_cases.append([
        3, [0.0, 0.0, 0.0], [5.0, 2.0, 2.0], grid_3d_10, None, 1, "3D obstacle (allowed)"
    ])

    test_cases.append([
        3, [0.0, 0.0, 0.0], [5.0, 2.0, 2.0], grid_3d_11, None, 1, "3D obstacle (blocked)"
    ])

    return test_cases

def create_test_grids():
    """Create all test grids for cartographer visualization (combines 2D and 3D)."""
    test_cases = []
    test_cases.extend(create_2d_test_grids())
    test_cases.extend(create_3d_test_grids())
    return test_cases

def test_cartographer_viz():
    """Execute all test cases and display cartographer visualizations."""
    test_cases = create_test_grids()
    
    for i, test_case in enumerate(test_cases):
        dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description = test_case
        
        print(f"\n=== Test Case {i+1}: {description} ===")
        print(f"Dimensions: {dimensions}D")
        print(f"Grid shape: {occupancy_grid.shape}")
        print(f"Start: {start_coords}, End: {end_coords}")
        print(f"Origin: {origin}, Loose: {loose}")
        
        # Test cartographer visualization
        print(f"\n--- Testing cartographer ---")
        title = f"Test Case {i+1}: {description}"
        
        try:
            result = visualize_cartographer(start_coords, end_coords, occupancy_grid, 
                                          origin=origin, loose=loose, title=title)
            if result['success']:
                print(f"[SUCCESS] Ray traced successfully")
                print(f"  Front cells: {len(result['traversed_front_cells'])}")
                print(f"  Additional cells: {len(result['additional_expanded_cells'])}")
                print(f"  Total cells: {len(result['all_expanded_cells'])}")
            else:
                print(f"[FAILED] Ray tracing failed: {result.get('error', 'Unknown error')}")
                print(f"  Front cells reached: {len(result['traversed_front_cells'])}")
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print("\n" + "=" * 50)
    print("All cartographer visualization tests completed!")

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
    dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description = test_case
    
    print(f"\n=== Test Case {test_index+1}: {description} ===")
    print(f"Dimensions: {dimensions}D")
    print(f"Grid shape: {occupancy_grid.shape}")
    print(f"Start: {start_coords}, End: {end_coords}")
    print(f"Origin: {origin}, Loose: {loose}")
    
    print(f"\n--- Testing cartographer ---")
    title = f"Test Case {test_index+1}: {description}"
    
    try:
        result = visualize_cartographer(start_coords, end_coords, occupancy_grid, 
                                      origin=origin, loose=loose, title=title)
        if result['success']:
            print(f"[SUCCESS] Ray traced successfully")
            print(f"  Front cells: {len(result['traversed_front_cells'])}")
            print(f"  Additional cells: {len(result['additional_expanded_cells'])}")
            print(f"  Total cells: {len(result['all_expanded_cells'])}")
        else:
            print(f"[FAILED] Ray tracing failed: {result.get('error', 'Unknown error')}")
            print(f"  Front cells reached: {len(result['traversed_front_cells'])}")
    except Exception as e:
        print(f"[ERROR] {e}")

def list_test_cases():
    """List all available test cases."""
    test_cases = create_test_grids()
    
    print("Available cartographer test cases:")
    print("=" * 50)
    for i, test_case in enumerate(test_cases):
        dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description = test_case
        print(f"{i:2d}: {description} ({dimensions}D, {occupancy_grid.shape}, loose={loose})")

if __name__ == "__main__":
    # You can run different test functions:
    # test_cartographer_viz()           # Run all tests
    # run_specific_test(0)              # Run test case 0
    # list_test_cases()                 # List all available test cases
    
    test_cartographer_viz()
