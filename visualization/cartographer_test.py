import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.cartographer_viz import visualize_cartographer

def create_2d_test_grids():
    """Create 2D test grids for cartographer visualization."""
    test_cases = []
    
    # Simple 2x2 grid with obstacles
    grid_2d_1 = np.array([
        [0, 1],  
        [1, 0],
    ], dtype=bool)

    test_cases.append([
        2, [0.0, 0.0], [1.5, 1.5], grid_2d_1, None, 1, "2D Simple diagonal", "cell"
    ])
    test_cases.append([
        2, [0.0, 0.0], [1.0, 1.0], grid_2d_1, None, 1, "2D Simple diagonal vertex", "vertex"
    ])
    
    # Larger grid with clear path
    grid_2d_2 = np.array([
        [0, 0, 0, 0],  
        [0, 1, 1, 0],  
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=bool)

    test_cases.append([
        2, [0.5, 0.5], [3.5, 3.5], grid_2d_2, None, 1, "4x3 diagonal path", "cell"
    ])
    test_cases.append([
        2, [0.0, 0.0], [3.0, 3.0], grid_2d_2, None, 1, "4x3 diagonal path vertex", "vertex"
    ])
    
    # Blocked path test
    grid_2d_3 = np.array([
        [0, 1, 0],  
        [1, 1, 1],  
        [0, 1, 0],
    ], dtype=bool)

    test_cases.append([
        2, [0.5, 0.5], [2.5, 2.5], grid_2d_3, None, 1, "Blocked path test", "cell"
    ])
    
    return test_cases

def create_3d_test_grids():
    """Create 3D test grids for cartographer visualization."""
    test_cases = []
    
    # 3D Test Case - Simple 3D grid with obstacles
    grid_3d_simple = np.array([
        # Layer 0 (Z=0) - Free space
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        
        # Layer 1 (Z=1) - Central obstacle
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        
        # Layer 2 (Z=2) - Free space
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    ], dtype=bool)
    
    test_cases.append([
        3, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], grid_3d_simple, None, 1, "3D Simple diagonal", "cell"
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
            result = visualize_cartographer(start_coords, end_coords, occupancy_grid, 
                                          origin=origin, loose=loose, mode=mode, title=title)
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
    dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description, mode = test_case
    
    print(f"\n=== Test Case {test_index+1}: {description} ===")
    print(f"Dimensions: {dimensions}D")
    print(f"Grid shape: {occupancy_grid.shape}")
    print(f"Start: {start_coords}, End: {end_coords}")
    print(f"Origin: {origin}, Loose: {loose}, Mode: {mode}")
    
    print(f"\n--- Testing {mode} mode ---")
    title = f"Test Case {test_index+1}: {description} ({mode} mode)"
    
    try:
        result = visualize_cartographer(start_coords, end_coords, occupancy_grid, 
                                      origin=origin, loose=loose, mode=mode, title=title)
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
        dimensions, start_coords, end_coords, occupancy_grid, origin, loose, description, mode = test_case
        print(f"{i:2d}: {description} ({dimensions}D, {occupancy_grid.shape}, loose={loose}, mode={mode})")

if __name__ == "__main__":
    # You can run different test functions:
    # test_cartographer_viz()           # Run all tests
    # run_specific_test(0)              # Run test case 0
    # list_test_cases()                 # List all available test cases
    
    test_cartographer_viz()
