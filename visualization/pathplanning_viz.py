import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.planner import plan_path

def visualize_pathplanning(start_coords, end_coords, occupancy_grid, origin=None, loose=1, algorithm='bfs', mode='cell', title="Path Planning Visualization"):
    """
    Main visualization function that handles both 2D and 3D path planning visualization.
    """
    # Plan the path
    path = plan_path(start_coords, end_coords, occupancy_grid, origin=origin, loose=loose, algorithm=algorithm, mode=mode)
    
    dimensions = len(occupancy_grid.shape)
    
    if dimensions == 2:
        return _visualize_2d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title)
    elif dimensions == 3:
        return _visualize_3d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title)
    else:
        print(f"Visualization not supported for {dimensions}D. Returning path array.")
        return path

def _visualize_2d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title):
    """Visualize 2D path planning to match raytracing visualization exactly."""
    # Add the parent directory to the path to import raytracer
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.raytracer import Raytracer
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Convert coordinates based on mode for visualization
    if mode == 'vertex':
        vis_start_coords = np.array(start_coords, dtype=float)
        vis_end_coords = np.array(end_coords, dtype=float)
    else:  # cell mode 
        vis_start_coords = np.array(start_coords, dtype=float) + 0.5
        vis_end_coords = np.array(end_coords, dtype=float) + 0.5
    
    # Get all intersected cells along the path using raytracing
    path_intersected_cells = set()
    if path and len(path) > 1:
        # Trace each segment of the path to get intersected cells
        for i in range(len(path) - 1):
            segment_start = np.array(path[i])
            segment_end = np.array(path[i + 1])
            
            # Create raytracer for this path segment
            raytracer = Raytracer(2, segment_start, segment_end)
            segment_cells = raytracer.trace()
            
            # Add all intersected cells from this segment
            for cell in segment_cells:
                path_intersected_cells.add(tuple(cell))
    
    # Set visualization bounds to exact grid boundaries (no padding beyond grid)
    height, width = occupancy_grid.shape
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    
    # Store path array for line drawing
    if path:
        path_array = np.array(path)
    else:
        print("No path found - creating visualization with start/end points only")
        path_array = None
    
    # Draw grid lines at integer positions to show exact grid boundaries
    for i in range(int(x_min), int(x_max) + 1):
        ax.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.7)
    for i in range(int(y_min), int(y_max) + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Draw obstacles (occupied cells) in dark gray with high transparency
    for y in range(height):
        for x in range(width):
            if occupancy_grid[y, x]:  # Occupied cell
                rect = patches.Rectangle((x, y), 1, 1, 
                                       linewidth=1, edgecolor='darkgray', 
                                       facecolor='darkgray', alpha=0.6)
                ax.add_patch(rect)
    
    # Draw path intersected cells (like raytracing intersected cells)
    if path_intersected_cells:
        for cell_coords in path_intersected_cells:
            x, y = int(cell_coords[0]), int(cell_coords[1])  # Ensure integer coordinates like raytracing
            rect = patches.Rectangle((x, y), 1, 1, 
                                   linewidth=1.5, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.6)
            ax.add_patch(rect)
    
    # Draw path line connecting centers (like ray line in raytracing)
    if path and len(path) > 1:
        ax.plot(path_array[:, 0], path_array[:, 1], 
                'r-', linewidth=3, label='Path', alpha=0.8)
    
    # Mark start and end points with mode-aware coordinates (like raytracing)
    ax.scatter(vis_start_coords[0], vis_start_coords[1], color='green', s=150, label='Start', zorder=5)
    ax.scatter(vis_end_coords[0], vis_end_coords[1], color='red', s=150, label='End', zorder=5)
    
    # Set axis properties to show exact grid bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y-axis for top-left origin
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set ticks to show exact grid boundaries
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    
    # Add path information (positioned like raytracing info)
    if path:
        # Determine if this is a complete path (reaches the goal) or attempted path
        if len(path) > 0:
            last_point = path[-1]
            goal_reached = (abs(last_point[0] - vis_end_coords[0]) < 0.1 and 
                          abs(last_point[1] - vis_end_coords[1]) < 0.1)
            
            if goal_reached:
                info_text = f'✅ Complete path: {len(path)} steps\n'
            else:
                info_text = f'⚠️ Attempted path: {len(path)} steps\n'
        else:
            info_text = f'Path cells: {len(path)}\n'
        
        # Add intersected cells info like raytracing
        info_text += f'Intersected cells: {len(path_intersected_cells)}\n'
        info_text += f'Start: ({vis_start_coords[0]:.1f}, {vis_start_coords[1]:.1f})\n'
        info_text += f'End: ({vis_end_coords[0]:.1f}, {vis_end_coords[1]:.1f})\n'
        info_text += f'Mode: {mode}'
    else:
        info_text = f'❌ No path found\n'
        info_text += f'Start: ({vis_start_coords[0]:.1f}, {vis_start_coords[1]:.1f})\n'
        info_text += f'End: ({vis_end_coords[0]:.1f}, {vis_end_coords[1]:.1f})\n'
        info_text += f'Mode: {mode}'
    
    ax.text(0.02, 0.02, info_text, 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return path

def _visualize_3d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title):
    """Visualize 3D path planning with multiple orthogonal views."""
    # Add the parent directory to the path to import raytracer
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.raytracer import Raytracer
    
    # Get grid dimensions
    depth, height, width = occupancy_grid.shape
    
    # Convert coordinates based on mode for visualization
    if mode == 'vertex':
        vis_start_coords = np.array(start_coords, dtype=float)
        vis_end_coords = np.array(end_coords, dtype=float)
    else:  # cell mode 
        vis_start_coords = np.array(start_coords, dtype=float) + 0.5
        vis_end_coords = np.array(end_coords, dtype=float) + 0.5
    
    # Get all intersected cells along the path using raytracing
    path_intersected_cells = set()
    if path and len(path) > 1:
        # Trace each segment of the path to get intersected cells
        for i in range(len(path) - 1):
            segment_start = np.array(path[i])
            segment_end = np.array(path[i + 1])
            
            # Create raytracer for this path segment
            raytracer = Raytracer(3, segment_start, segment_end)
            segment_cells = raytracer.trace()
            
            # Add all intersected cells from this segment
            for cell in segment_cells:
                path_intersected_cells.add(tuple(cell))
    
    # Always create visualization, even if no complete path found
    print(f"Visualizing 3D pathplanning - Path length: {len(path) if path else 0}, Intersected cells: {len(path_intersected_cells)}")
    
    # Create figure with multiple subplots for orthogonal views
    fig = plt.figure(figsize=(20, 12))
    
    # 3D perspective view
    ax1 = fig.add_subplot(221, projection='3d')
    _draw_3d_pathplanning_view(ax1, vis_start_coords, vis_end_coords, occupancy_grid, path, path_intersected_cells, mode,
                              f"{title} - 3D View")
    
    # XY view (top-down, looking along Z-axis)
    ax2 = fig.add_subplot(222)
    _draw_2d_projection_view(ax2, vis_start_coords, vis_end_coords, occupancy_grid, path, path_intersected_cells, 'xy', mode,
                            f"{title} - XY View (Top)")
    
    # XZ view (side view, looking along Y-axis)
    ax3 = fig.add_subplot(223)
    _draw_2d_projection_view(ax3, vis_start_coords, vis_end_coords, occupancy_grid, path, path_intersected_cells, 'xz', mode,
                            f"{title} - XZ View (Side)")
    
    # YZ view (front view, looking along X-axis)
    ax4 = fig.add_subplot(224)
    _draw_2d_projection_view(ax4, vis_start_coords, vis_end_coords, occupancy_grid, path, path_intersected_cells, 'yz', mode,
                            f"{title} - YZ View (Front)")
    
    plt.tight_layout()
    plt.show()
    
    return path

def _draw_3d_pathplanning_view(ax, vis_start_coords, vis_end_coords, occupancy_grid, path, path_intersected_cells, mode, title):
    """Draw the 3D perspective view for path planning to match raytracing."""
    depth, height, width = occupancy_grid.shape
    
    # Draw obstacles as semi-transparent cubes (like raytracing intersected cells)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if occupancy_grid[z, y, x]:  # Occupied cell
                    _draw_cube(ax, x, y, z, color='darkgray', alpha=0.3)
    
    # Draw path intersected cells as light blue cubes (like raytracing intersected cells)
    if path_intersected_cells:
        for cell_coords in path_intersected_cells:
            x, y, z = int(cell_coords[0]), int(cell_coords[1]), int(cell_coords[2])  # Ensure integer coordinates
            _draw_cube(ax, x, y, z, color='lightblue', alpha=0.6)
        
        # Draw path line connecting centers (like ray line)
        if path and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                    'r-', linewidth=4, label='Path', alpha=0.9)
    
    # Mark start and end points with mode-aware coordinates (like raytracing)
    ax.scatter(vis_start_coords[0], vis_start_coords[1], vis_start_coords[2], 
               color='green', s=150, label='Start', alpha=1.0)
    ax.scatter(vis_end_coords[0], vis_end_coords[1], vis_end_coords[2], 
               color='red', s=150, label='End', alpha=1.0)
    
    # Set bounds to exact grid boundaries (no padding beyond grid)
    x_min, x_max = 0, width
    y_min, y_max = 0, height  
    z_min, z_max = 0, depth
    
    # Set axis properties to show exact grid bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_aspect('equal')  
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    
    # Set ticks to show exact grid boundaries
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    ax.set_zticks(range(int(z_min), int(z_max) + 1))

def _draw_cube(ax, x, y, z, color='blue', alpha=0.3):
    """Draw a 3D cube at the given grid position (identical to raytracing)."""
    # Define the vertices of a unit cube
    vertices = [
        [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],  # bottom face
        [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]  # top face
    ]
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[4], vertices[7], vertices[3], vertices[0]]   # left
    ]
    
    # Create 3D polygon collection
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.8)
    ax.add_collection3d(poly3d)

def _draw_2d_projection_view(ax, vis_start_coords, vis_end_coords, occupancy_grid, path, path_intersected_cells, view_type, mode, title):
    """Draw a 2D projection view for 3D path planning to match raytracing."""
    depth, height, width = occupancy_grid.shape
    
    # Determine projection parameters based on view type
    if view_type == 'xy':
        coord1_idx, coord2_idx = 0, 1  # X, Y
        xlabel, ylabel = 'X', 'Y'
        # Project along Z-axis
        projected_obstacles = np.any(occupancy_grid, axis=0)
        max_coord1, max_coord2 = width, height
    elif view_type == 'xz':
        coord1_idx, coord2_idx = 0, 2  # X, Z
        xlabel, ylabel = 'X', 'Z'
        # Project along Y-axis
        projected_obstacles = np.any(occupancy_grid, axis=1)
        max_coord1, max_coord2 = width, depth
    else:  # yz
        coord1_idx, coord2_idx = 1, 2  # Y, Z
        xlabel, ylabel = 'Y', 'Z'
        # Project along X-axis
        projected_obstacles = np.any(occupancy_grid, axis=2)
        max_coord1, max_coord2 = height, depth
    
    # Draw projected obstacles
    grid_shape = projected_obstacles.shape
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if projected_obstacles[i, j]:
                if view_type == 'xy':
                    ax.add_patch(Rectangle((j, i), 1, 1, color='darkgray', alpha=0.3))
                elif view_type == 'xz':
                    ax.add_patch(Rectangle((j, i), 1, 1, color='darkgray', alpha=0.3))
                else:  # yz
                    ax.add_patch(Rectangle((i, j), 1, 1, color='darkgray', alpha=0.3))
    
    # Draw path intersected cells
    if path_intersected_cells:
        # Project path intersected coordinates
        projected_path_cells = set()
        for cell_coords in path_intersected_cells:
            coord1, coord2 = int(cell_coords[coord1_idx]), int(cell_coords[coord2_idx])
            projected_path_cells.add((coord1, coord2))
        
        # Draw path intersected cells
        for coord1, coord2 in projected_path_cells:
            ax.add_patch(Rectangle((coord1, coord2), 1, 1, color='lightblue', alpha=0.6))
        
        # Draw path line
        if path and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, coord1_idx], path_array[:, coord2_idx], 
                    'r-', linewidth=3, label='Path', alpha=0.8)
    
    # Mark start and end points with mode-aware coordinates
    start_coord1, start_coord2 = vis_start_coords[coord1_idx], vis_start_coords[coord2_idx]
    end_coord1, end_coord2 = vis_end_coords[coord1_idx], vis_end_coords[coord2_idx]
    
    ax.scatter(start_coord1, start_coord2, color='green', s=150, label='Start', zorder=5)
    ax.scatter(end_coord1, end_coord2, color='red', s=150, label='End', zorder=5)
    
    # Set bounds to exact grid boundaries (no padding beyond grid)
    x_min, x_max = 0, max_coord1
    y_min, y_max = 0, max_coord2
    
    # Set axis properties to show exact grid bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y-axis for top-left origin in projections
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    
    # Draw grid lines at integer positions to show exact grid boundaries
    for x in range(int(x_min), int(x_max) + 1):
        ax.axvline(x, color='lightgray', linewidth=0.5, alpha=0.7)
    for y in range(int(y_min), int(y_max) + 1):
        ax.axhline(y, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Set ticks to show exact grid boundaries
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    ax.legend(fontsize=8)
