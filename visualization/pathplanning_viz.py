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
        return _visualize_2d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title, origin)
    elif dimensions == 3:
        return _visualize_3d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title, origin)
    else:
        print(f"Visualization not supported for {dimensions}D. Returning path array.")
        return path

def _visualize_2d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title, origin=None):
    """Visualize 2D path planning to match raytracing visualization exactly."""
    # Add the parent directory to the path to import raytracer
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.raytracer import Raytracer
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Handle origin offset for coordinate display
    if origin is None:
        origin = [0, 0]
    origin = np.array(origin)
    
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
    
    # Set visualization bounds with origin offset
    height, width = occupancy_grid.shape
    x_min, x_max = origin[0], origin[0] + width
    y_min, y_max = origin[1], origin[1] + height
    
    # Store path array for line drawing
    if path:
        path_array = np.array(path)
    else:
        print("No path found - creating visualization with start/end points only")
        path_array = None
    
    # Draw grid lines at integer positions to show exact grid boundaries with origin offset
    for i in range(int(x_min), int(x_max) + 1):
        ax.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.7)
    for i in range(int(y_min), int(y_max) + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Draw obstacles (occupied cells) with origin offset
    for y in range(height):
        for x in range(width):
            if occupancy_grid[y, x]:  # Occupied cell
                # Apply origin offset to grid position
                grid_x, grid_y = x + origin[0], y + origin[1]
                rect = patches.Rectangle((grid_x, grid_y), 1, 1, 
                                       linewidth=1, edgecolor='darkgray', 
                                       facecolor='darkgray', alpha=1.0)
                ax.add_patch(rect)
    
    # Draw path intersected cells with origin offset
    if path_intersected_cells:
        for cell_coords in path_intersected_cells:
            # Apply origin offset to cell coordinates
            x, y = int(cell_coords[0]) + origin[0], int(cell_coords[1]) + origin[1]
            rect = patches.Rectangle((x, y), 1, 1, 
                                   linewidth=1.5, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.6)
            ax.add_patch(rect)
    
    # Draw path line connecting centers (like ray line in raytracing) with origin offset
    if path and len(path) > 1:
        # Apply origin offset to path coordinates for display
        path_display = path_array + origin
        ax.plot(path_display[:, 0], path_display[:, 1], 
                'r-', linewidth=3, label='Path', alpha=0.8)
    
    # Mark start and end points with origin offset applied
    start_display = vis_start_coords + origin
    end_display = vis_end_coords + origin
    ax.scatter(start_display[0], start_display[1], color='green', s=150, label='Start', zorder=5)
    ax.scatter(end_display[0], end_display[1], color='red', s=150, label='End', zorder=5)
    
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
    
    # Add comprehensive path information
    if path:
        # Determine if this is a complete path (reaches the goal) or attempted path
        if len(path) > 0:
            last_point = path[-1]
            goal_reached = (abs(last_point[0] - vis_end_coords[0]) < 0.1 and 
                          abs(last_point[1] - vis_end_coords[1]) < 0.1)
            
            if goal_reached:
                status_text = '[SUCCESS] PATH FOUND'
                status_color = 'lightgreen'
                path_info = f'Complete path: {len(path)} steps'
            else:
                status_text = '[PARTIAL] PARTIAL PATH'
                status_color = 'lightyellow'
                path_info = f'Attempted path: {len(path)} steps'
        else:
            status_text = '[FAILED] NO PATH'
            status_color = 'lightcoral'
            path_info = f'Path cells: {len(path)}'
        
        # Create detailed path information
        info_text = f'{status_text}\n'
        info_text += f'{path_info}\n'
        info_text += f'Intersected cells: {len(path_intersected_cells)}\n'
        info_text += f'====================\n'
        info_text += f'START: ({start_coords[0]}, {start_coords[1]})\n'
        info_text += f'END: ({end_coords[0]}, {end_coords[1]})\n'
        info_text += f'Mode: {mode.upper()}\n'
        
        # Add path waypoints if path exists and is reasonable length
        if path and len(path) > 1:
            info_text += f'====================\n'
            if len(path) <= 10:  # Show all points for short paths
                info_text += f'PATH POINTS:\n'
                for i, point in enumerate(path):
                    info_text += f'  {i+1}: ({point[0]:.1f}, {point[1]:.1f})\n'
            else:  # Show first few and last few for long paths
                info_text += f'PATH SAMPLE:\n'
                for i in range(3):
                    point = path[i]
                    info_text += f'  {i+1}: ({point[0]:.1f}, {point[1]:.1f})\n'
                info_text += f'  ... ({len(path)-6} more points)\n'
                for i in range(len(path)-3, len(path)):
                    point = path[i]
                    info_text += f'  {i+1}: ({point[0]:.1f}, {point[1]:.1f})\n'
    else:
        status_text = '[FAILED] NO PATH FOUND'
        status_color = 'lightcoral'
        info_text = f'{status_text}\n'
        info_text += f'No valid path to destination\n'
        info_text += f'====================\n'
        info_text += f'START: ({start_coords[0]}, {start_coords[1]})\n'
        info_text += f'END: ({end_coords[0]}, {end_coords[1]})\n'
        info_text += f'Mode: {mode.upper()}'
    
    # Position info box in top-left corner for better visibility
    ax.text(0.02, 0.98, info_text, 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.9, edgecolor='black'))
    
    # Add coordinate labels directly on start/end points with origin-aware coordinates
    ax.annotate(f'START\n({start_coords[0]}, {start_coords[1]})', 
                xy=(start_display[0], start_display[1]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                fontsize=8, ha='left')
    
    ax.annotate(f'END\n({end_coords[0]}, {end_coords[1]})', 
                xy=(end_display[0], end_display[1]), 
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                fontsize=8, ha='left')
    
    plt.tight_layout(pad=10.0)  
    plt.show()
    
    return path

def _visualize_3d_pathplanning(start_coords, end_coords, occupancy_grid, path, mode, title, origin=None):
    """Visualize 3D path planning with multiple orthogonal views."""
    # Add the parent directory to the path to import raytracer
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.raytracer import Raytracer
    
    # Get grid dimensions
    depth, height, width = occupancy_grid.shape
    
    # Handle origin offset for coordinate display
    if origin is None:
        origin = [0, 0, 0]
    origin = np.array(origin)
    
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
    
    # Determine path status for console output and title enhancement
    if path:
        last_point = path[-1]
        goal_reached = (abs(last_point[0] - vis_end_coords[0]) < 0.1 and 
                       abs(last_point[1] - vis_end_coords[1]) < 0.1 and
                       abs(last_point[2] - vis_end_coords[2]) < 0.1)
        
        if goal_reached:
            path_status = "[SUCCESS] PATH FOUND"
            print(f"[SUCCESS] Visualizing 3D pathplanning - COMPLETE PATH: {len(path)} steps, Intersected cells: {len(path_intersected_cells)}")
        else:
            path_status = "[PARTIAL] PARTIAL PATH"
            print(f"[PARTIAL] Visualizing 3D pathplanning - PARTIAL PATH: {len(path)} steps, Intersected cells: {len(path_intersected_cells)}")
    else:
        path_status = "[FAILED] NO PATH"
        print(f"[FAILED] Visualizing 3D pathplanning - NO PATH FOUND, Intersected cells: {len(path_intersected_cells)}")
    
    # Create figure with multiple subplots for orthogonal views
    fig = plt.figure(figsize=(20, 12))
    
    # Add main title with path status
    fig.suptitle(f'{title} | {path_status} | Start: {start_coords} → End: {end_coords}', 
                 fontsize=16, fontweight='bold')
    
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
    
    # Mark start and end points with mode-aware coordinates and labels
    ax.scatter(vis_start_coords[0], vis_start_coords[1], vis_start_coords[2], 
               color='green', s=200, label=f'Start {tuple(map(int, vis_start_coords))}', alpha=1.0)
    ax.scatter(vis_end_coords[0], vis_end_coords[1], vis_end_coords[2], 
               color='red', s=200, label=f'End {tuple(map(int, vis_end_coords))}', alpha=1.0)
    
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
    
    # Enhanced title with path information
    if path:
        last_point = path[-1]
        goal_reached = (abs(last_point[0] - vis_start_coords[0]) > 0.1 or 
                       abs(last_point[1] - vis_start_coords[1]) > 0.1 or
                       abs(last_point[2] - vis_start_coords[2]) > 0.1)
        
        if goal_reached:
            title_suffix = f"[SUCCESS] {len(path)} steps"
        else:
            title_suffix = f"[PARTIAL] {len(path)} steps (partial)"
    else:
        title_suffix = "[FAILED] No path"
    
    ax.set_title(f"{title}\n{title_suffix}", fontsize=12)
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
    
    # Mark start and end points with mode-aware coordinates and enhanced labels
    start_coord1, start_coord2 = vis_start_coords[coord1_idx], vis_start_coords[coord2_idx]
    end_coord1, end_coord2 = vis_end_coords[coord1_idx], vis_end_coords[coord2_idx]
    
    ax.scatter(start_coord1, start_coord2, color='green', s=150, 
               label=f'Start ({int(vis_start_coords[0])},{int(vis_start_coords[1])},{int(vis_start_coords[2])})', zorder=5)
    ax.scatter(end_coord1, end_coord2, color='red', s=150, 
               label=f'End ({int(vis_end_coords[0])},{int(vis_end_coords[1])},{int(vis_end_coords[2])})', zorder=5)
    
    # Set bounds to exact grid boundaries (no padding beyond grid)
    x_min, x_max = 0, max_coord1
    y_min, y_max = 0, max_coord2
    
    # Set axis properties to show exact grid bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y-axis for top-left origin in projections
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    # Enhanced title with path information for 2D projections
    if path:
        path_count = len([cell for cell in path_intersected_cells 
                         if (int(cell[coord1_idx]), int(cell[coord2_idx])) in 
                         {(int(c[coord1_idx]), int(c[coord2_idx])) for c in path_intersected_cells}])
        title_suffix = f" | {len(path)} steps, {len(path_intersected_cells)} intersected"
    else:
        title_suffix = " | No path found"
    
    ax.set_title(f"{title}{title_suffix}", fontsize=10)
    
    # Draw grid lines at integer positions to show exact grid boundaries
    for x in range(int(x_min), int(x_max) + 1):
        ax.axvline(x, color='lightgray', linewidth=0.5, alpha=0.7)
    for y in range(int(y_min), int(y_max) + 1):
        ax.axhline(y, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Set ticks to show exact grid boundaries
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    ax.legend(fontsize=8)
