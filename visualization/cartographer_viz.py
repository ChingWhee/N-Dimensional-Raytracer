import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cartographer import Cartographer

def visualize_cartographer(start_coords, end_coords, occupancy_grid, origin=None, loose=1, title="Cartographer Visualization"):
    """
    Main visualization function that handles both 2D and 3D cartographer visualization.
    """
    # Create and run cartographer
    cartographer = Cartographer(len(occupancy_grid.shape), start_coords, end_coords, occupancy_grid, origin, loose)
    result = cartographer.map()
    
    dimensions = len(occupancy_grid.shape)
    
    if dimensions == 2:
        return _visualize_2d_cartographer(start_coords, end_coords, occupancy_grid, result, title, origin)
    elif dimensions == 3:
        return _visualize_3d_cartographer(start_coords, end_coords, occupancy_grid, result, title, origin)
    else:
        print(f"Visualization not supported for {dimensions}D. Returning result.")
        return result

def _visualize_2d_cartographer(start_coords, end_coords, occupancy_grid, result, title, origin=None):
    """Visualize 2D cartographer results following pathplanning conventions."""
    # Create figure with extra width for info panel
    fig, (ax, info_ax) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [6, 1]})
    
    # Hide the info axis (we'll just use it for text placement)
    info_ax.axis('off')
    
    # Handle origin offset for coordinate display
    if origin is None:
        origin = [0, 0]
    origin = np.array(origin)
    
    # Use coordinates as-is (no mode-based shifting for cartographer)
    vis_start_coords = np.array(start_coords, dtype=float)
    vis_end_coords = np.array(end_coords, dtype=float)
    
    # Extract data from result
    success = result.get('success', False)
    traversed_front_cells = result.get('traversed_front_cells', [])
    all_expanded_cells = result.get('traversed_front_cells', [])
    
    # Set visualization bounds with origin offset
    height, width = occupancy_grid.shape
    x_min, x_max = origin[0], origin[0] + width
    y_min, y_max = origin[1], origin[1] + height
    
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
    
    # Draw traversed front cells (light blue like pathplanning)
    if traversed_front_cells:
        for cell_coords in traversed_front_cells:
            # Apply origin offset to cell coordinates
            x, y = int(cell_coords[0]) + origin[0], int(cell_coords[1]) + origin[1]
            rect = patches.Rectangle((x, y), 1, 1, 
                                   linewidth=1.5, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.6)
            ax.add_patch(rect)
    
    
    # Draw ray line from start to end with origin offset
    start_display = vis_start_coords + origin
    end_display = vis_end_coords + origin
    ax.plot([start_display[0], end_display[0]], [start_display[1], end_display[1]], 
            'r-', linewidth=3, label='Ray', alpha=0.8)
    
    # Mark start and end points with origin offset applied
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
    
    # Add comprehensive cartographer information in separate panel
    if success:
        status_text = '[SUCCESS] RAY TRACED'
        status_color = 'lightgreen'
    else:
        status_text = '[FAILED] RAY BLOCKED'
        status_color = 'lightcoral'
    
    # Create detailed information for info panel
    info_text = f'{status_text}\n\n'
    if not success:
        info_text += f'Error: {result.get("error", "Unknown error")}\n\n'
    info_text += f'Front cells: {len(traversed_front_cells)}\n'
    info_text += f'Total cells: {len(all_expanded_cells)}\n\n'
    info_text += f'START: ({start_coords[0]}, {start_coords[1]})\n'
    info_text += f'END: ({end_coords[0]}, {end_coords[1]})\n\n'
    
    # Add raytracer position information if available
    if 'raytracer_position' in result:
        raytracer_pos = result['raytracer_position']
        info_text += f'Ray Length: {raytracer_pos.get("parametric_position", 0):.3f}\n'
        info_text += f'Reached Goal: {raytracer_pos.get("reached_goal", False)}\n'
    
    # Place info text in the info panel
    info_ax.text(0.05, 0.95, info_text, 
                transform=info_ax.transAxes, fontsize=11, verticalalignment='top',
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
    
    plt.tight_layout(pad=5.0)  
    plt.show()
    
    return result

def _visualize_3d_cartographer(start_coords, end_coords, occupancy_grid, result, title, origin=None):
    """Visualize 3D cartographer results with multiple orthogonal views."""
    # Get grid dimensions
    depth, height, width = occupancy_grid.shape
    
    # Handle origin offset for coordinate display
    if origin is None:
        origin = [0, 0, 0]
    origin = np.array(origin)
    
    # Use coordinates as-is (no mode-based shifting for cartographer)
    vis_start_coords = np.array(start_coords, dtype=float)
    vis_end_coords = np.array(end_coords, dtype=float)
    
    # Extract data from result
    success = result.get('success', False)
    traversed_front_cells = result.get('traversed_front_cells', [])
    all_expanded_cells = result.get('traversed_front_cells', [])
    
    # Determine status for title enhancement
    if success:
        cartographer_status = "[SUCCESS] RAY TRACED"
    else:
        cartographer_status = "[FAILED] RAY BLOCKED"
    
    # Create figure with extra width for all views plus info panel
    fig = plt.figure(figsize=(24, 9))
    fig.subplots_adjust(
        left=0.05,    
        right=0.75,   
        top=0.95,
        bottom=0.05,
        wspace=0.3,
        hspace=0.3
    )
    
    # Add main title with status
    fig.suptitle(f'{title} | {cartographer_status} | Start: {start_coords} → End: {end_coords}', 
                 fontsize=16, fontweight='bold')
    
    # 3D perspective view (larger)
    ax1 = fig.add_subplot(221, projection='3d')
    _draw_3d_cartographer_view(ax1, vis_start_coords, vis_end_coords, occupancy_grid, result,
                              f"{title} - 3D View")
    
    # XY view (top-down, looking along Z-axis)
    ax2 = fig.add_subplot(222)
    _draw_2d_cartographer_projection_view(ax2, vis_start_coords, vis_end_coords, occupancy_grid, result, 'xy',
                            f"{title} - XY View (Top)")
    
    # XZ view (side view, looking along Y-axis)
    ax3 = fig.add_subplot(223)
    _draw_2d_cartographer_projection_view(ax3, vis_start_coords, vis_end_coords, occupancy_grid, result, 'xz',
                            f"{title} - XZ View (Side)")
    
    # YZ view (side view, looking along X-axis)
    ax4 = fig.add_subplot(224)
    _draw_2d_cartographer_projection_view(ax4, vis_start_coords, vis_end_coords, occupancy_grid, result, 'yz',
                            f"{title} - YZ View (Front)")
    
    # Info panel in dedicated space on the right
    ax_info = fig.add_axes([0.78, 0.05, 0.20, 0.90])
    ax_info.axis('off')
    
    # Create comprehensive info text for 3D
    info_text = f'{cartographer_status}\n\n'
    if not success:
        info_text += f'Error: {result.get("error", "Unknown error")}\n\n'
    info_text += f'Front cells: {len(traversed_front_cells)}\n'
    info_text += f'Total cells: {len(all_expanded_cells)}\n\n'
    info_text += f'START: ({start_coords[0]}, {start_coords[1]}, {start_coords[2]})\n'
    info_text += f'END: ({end_coords[0]}, {end_coords[1]}, {end_coords[2]})\n\n'
    
    # Add raytracer position information if available
    if 'raytracer_position' in result:
        raytracer_pos = result['raytracer_position']
        info_text += f'Ray Length: {raytracer_pos.get("parametric_position", 0):.3f}\n'
        info_text += f'Reached Goal: {raytracer_pos.get("reached_goal", False)}\n'
    
    # Set info panel colors
    if success:
        status_color = 'lightgreen'
    else:
        status_color = 'lightcoral'
    
    # Place info text in the dedicated info panel
    ax_info.text(0.05, 0.95, info_text, 
            transform=ax_info.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.9, edgecolor='black'))
    
    plt.show()
    
    return result

def _draw_3d_cartographer_view(ax, vis_start_coords, vis_end_coords, occupancy_grid, result, title):
    """Draw the 3D perspective view for cartographer results."""
    depth, height, width = occupancy_grid.shape
    
    # Extract data from result
    traversed_front_cells = result.get('traversed_front_cells', [])
    
    # Draw obstacles as semi-transparent cubes
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if occupancy_grid[z, y, x]:  # Occupied cell
                    _draw_cube(ax, x, y, z, color='darkgray', alpha=0.8)
    
    # Draw traversed front cells as light blue cubes
    if traversed_front_cells:
        for cell_coords in traversed_front_cells:
            x, y, z = int(cell_coords[0]), int(cell_coords[1]), int(cell_coords[2])
            _draw_cube(ax, x, y, z, color='lightblue', alpha=0.6)
    
    
    # Draw ray line from start to end
    ax.plot([vis_start_coords[0], vis_end_coords[0]], 
            [vis_start_coords[1], vis_end_coords[1]], 
            [vis_start_coords[2], vis_end_coords[2]], 
            'r-', linewidth=3, alpha=0.8, label='Ray')
    
    # Mark start and end points with mode-aware coordinates and labels
    ax.scatter(vis_start_coords[0], vis_start_coords[1], vis_start_coords[2], 
               color='green', s=200, label=f'Start {tuple(map(int, vis_start_coords))}', alpha=1.0)
    ax.scatter(vis_end_coords[0], vis_end_coords[1], vis_end_coords[2], 
               color='red', s=200, label=f'End {tuple(map(int, vis_end_coords))}', alpha=1.0)
    
    # Set bounds to exact grid boundaries
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
    
    # Enhanced title with cartographer information
    success = result.get('success', False)
    front_count = len(traversed_front_cells)
    if success:
        title_suffix = f"SUCCESS | Front: {front_count}"
    else:
        title_suffix = f"FAILED | Front: {front_count}"
    
    ax.set_title(f"{title}\n{title_suffix}", fontsize=12)
    ax.legend(fontsize=8)
    
    # Set ticks to show exact grid boundaries
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    ax.set_zticks(range(int(z_min), int(z_max) + 1))

def _draw_cube(ax, x, y, z, color='blue', alpha=0.3):
    """Draw a 3D cube at the given grid position (identical to pathplanning)."""
    # Define the vertices of a unit cube
    vertices = [
        [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
        [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
    ]
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    
    # Create 3D polygon collection
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.8)
    ax.add_collection3d(poly3d)

def _draw_2d_cartographer_projection_view(ax, vis_start_coords, vis_end_coords, occupancy_grid, result, view_type, title):
    """Draw a 2D projection view for 3D cartographer results."""
    depth, height, width = occupancy_grid.shape
    
    # Extract data from result
    traversed_front_cells = result.get('traversed_front_cells', [])
    
    # Determine projection parameters based on view type
    if view_type == 'xy':
        projected_obstacles = np.any(occupancy_grid, axis=0)  # Project along Z
        xlabel, ylabel = 'X', 'Y'
    elif view_type == 'xz':
        projected_obstacles = np.any(occupancy_grid, axis=1)  # Project along Y
        xlabel, ylabel = 'X', 'Z'
    else:  # yz
        projected_obstacles = np.any(occupancy_grid, axis=2)  # Project along X
        xlabel, ylabel = 'Y', 'Z'
    
    # Draw projected obstacles
    grid_shape = projected_obstacles.shape
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if projected_obstacles[i, j]:  # Occupied cell
                rect = patches.Rectangle((j, i), 1, 1, 
                                       linewidth=1, edgecolor='darkgray', 
                                       facecolor='darkgray', alpha=1.0)
                ax.add_patch(rect)
    
    # Draw traversed front cells
    if traversed_front_cells:
        for cell_coords in traversed_front_cells:
            if view_type == 'xy':
                x, y = int(cell_coords[0]), int(cell_coords[1])
            elif view_type == 'xz':
                x, y = int(cell_coords[0]), int(cell_coords[2])
            else:  # yz
                x, y = int(cell_coords[1]), int(cell_coords[2])
            
            rect = patches.Rectangle((x, y), 1, 1, 
                                   linewidth=1.5, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.6)
            ax.add_patch(rect)
    
    
    # Draw ray line
    if view_type == 'xy':
        start_coord1, start_coord2 = vis_start_coords[0], vis_start_coords[1]
        end_coord1, end_coord2 = vis_end_coords[0], vis_end_coords[1]
    elif view_type == 'xz':
        start_coord1, start_coord2 = vis_start_coords[0], vis_start_coords[2]
        end_coord1, end_coord2 = vis_end_coords[0], vis_end_coords[2]
    else:  # yz
        start_coord1, start_coord2 = vis_start_coords[1], vis_start_coords[2]
        end_coord1, end_coord2 = vis_end_coords[1], vis_end_coords[2]
    
    ax.plot([start_coord1, end_coord1], [start_coord2, end_coord2], 
            'r-', linewidth=3, alpha=0.8, label='Ray')
    
    # Mark start and end points with mode-aware coordinates and enhanced labels
    ax.scatter(start_coord1, start_coord2, color='green', s=150, 
               label=f'Start ({int(vis_start_coords[0])},{int(vis_start_coords[1])},{int(vis_start_coords[2])})', zorder=5)
    ax.scatter(end_coord1, end_coord2, color='red', s=150, 
               label=f'End ({int(vis_end_coords[0])},{int(vis_end_coords[1])},{int(vis_end_coords[2])})', zorder=5)
    
    # Set bounds to exact grid boundaries
    if view_type == 'xy':
        x_min, x_max = 0, width
        y_min, y_max = 0, height
    elif view_type == 'xz':
        x_min, x_max = 0, width
        y_min, y_max = 0, depth
    else:  # yz
        x_min, x_max = 0, height
        y_min, y_max = 0, depth
    
    # Set axis properties to show exact grid bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y-axis for top-left origin in projections
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    # Enhanced title with cartographer information for 2D projections
    success = result.get('success', False)
    front_count = len(traversed_front_cells)
    if success:
        title_suffix = f" | SUCCESS | F:{front_count}"
    else:
        title_suffix = f" | FAILED | F:{front_count}"
    
    ax.set_title(f"{title}{title_suffix}", fontsize=10)
    
    # Draw grid lines at integer positions to show exact grid boundaries
    for x in range(int(x_min), int(x_max) + 1):
        ax.axvline(x=x, color='lightgray', linewidth=0.5, alpha=0.7)
    for y in range(int(y_min), int(y_max) + 1):
        ax.axhline(y=y, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Set ticks to show exact grid boundaries
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    ax.legend(fontsize=8)
