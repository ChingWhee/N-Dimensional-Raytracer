import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from raytracer import Raytracer

def visualize_raytracer(raytracer, title="Raytracer Visualization"):
    """
    Main visualization function that handles both 2D and 3D cases.
    For other dimensions, returns the intersected cells array.
    """
    # Get all intersected cells using trace method
    intersected_cells = raytracer.trace()
    
    if raytracer.dimensions == 2:
        return _visualize_2d(raytracer, intersected_cells, title)
    elif raytracer.dimensions == 3:
        return _visualize_3d(raytracer, intersected_cells, title)
    else:
        # For other dimensions, just return the array of intersected cells
        print(f"Visualization not supported for {raytracer.dimensions}D. Returning intersected cells array.")
        return intersected_cells

def _visualize_2d(raytracer, intersected_cells, title):
    """Visualize 2D raytracer."""
    if raytracer.grid is None:
        print("Raytracer initialization failed")
        return intersected_cells
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Convert intersected cells to numpy array for easier manipulation
    if not intersected_cells:
        print("No intersected cells found")
        return intersected_cells
    
    cells_array = np.array(intersected_cells)
    
    # Calculate visualization bounds based on ray coordinates and intersected cells
    ray_coords = np.array([raytracer.start_coords, raytracer.end_coords])
    
    # Find the bounds that encompass both the ray and all intersected cells
    x_coords = np.concatenate([ray_coords[:, 0], cells_array[:, 0]])
    y_coords = np.concatenate([ray_coords[:, 1], cells_array[:, 1]])
    
    x_min, x_max = np.floor(x_coords.min()), np.ceil(x_coords.max())
    y_min, y_max = np.floor(y_coords.min()), np.ceil(y_coords.max())
    
    # Add some padding
    x_min -= 1
    x_max += 1
    y_min -= 1
    y_max += 1
    
    # Draw grid lines
    for i in range(int(x_min), int(x_max) + 1):
        ax.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.7)
    for i in range(int(y_min), int(y_max) + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Color intersected cells
    for cell in intersected_cells:
        if len(cell) == 2:  # Ensure it's 2D
            x, y = int(cell[0]), int(cell[1])  # Ensure integer coordinates
            rect = patches.Rectangle((x, y), 1, 1, 
                                   linewidth=1.5, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.6)
            ax.add_patch(rect)
    
    # Draw the ray line
    start_x, start_y = raytracer.start_coords[0], raytracer.start_coords[1]
    end_x, end_y = raytracer.end_coords[0], raytracer.end_coords[1]
    
    ax.plot([start_x, end_x], [start_y, end_y], 
            'r-', linewidth=3, label='Ray Path', alpha=0.8)
    
    # Mark start and end points
    ax.plot(start_x, start_y, 'go', markersize=12, label='Start', zorder=5)
    ax.plot(end_x, end_y, 'ro', markersize=12, label='End', zorder=5)
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Force integer ticks only
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    
    # Add text showing number of intersected cells and coordinates
    info_text = f'Intersected cells: {len(intersected_cells)}\n'
    info_text += f'Start: ({raytracer.start_coords[0]:.1f}, {raytracer.start_coords[1]:.1f})\n'
    info_text += f'End: ({raytracer.end_coords[0]:.1f}, {raytracer.end_coords[1]:.1f})'
    
    ax.text(0.02, 0.02, info_text, 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return intersected_cells

def _visualize_3d(raytracer, intersected_cells, title):
    """Visualize 3D raytracer with multiple orthogonal views."""
    if raytracer.grid is None:
        print("Raytracer initialization failed")
        return intersected_cells
    
    if not intersected_cells:
        print("No intersected cells found")
        return intersected_cells
    
    # Convert intersected cells to numpy array for easier manipulation
    cells_array = np.array(intersected_cells)
    
    # Calculate visualization bounds
    ray_coords = np.array([raytracer.start_coords, raytracer.end_coords])
    
    # Find the bounds that encompass both the ray and all intersected cells
    x_coords = np.concatenate([ray_coords[:, 0], cells_array[:, 0]])
    y_coords = np.concatenate([ray_coords[:, 1], cells_array[:, 1]])
    z_coords = np.concatenate([ray_coords[:, 2], cells_array[:, 2]])
    
    x_min, x_max = np.floor(x_coords.min()), np.ceil(x_coords.max())
    y_min, y_max = np.floor(y_coords.min()), np.ceil(y_coords.max())
    z_min, z_max = np.floor(z_coords.min()), np.ceil(z_coords.max())
    
    # Add some padding
    x_min -= 1
    x_max += 1
    y_min -= 1
    y_max += 1
    z_min -= 1
    z_max += 1
    
    # Create figure with multiple subplots for orthogonal views
    fig = plt.figure(figsize=(20, 12))
    
    # 3D perspective view
    ax1 = fig.add_subplot(221, projection='3d')
    _draw_3d_view(ax1, raytracer, intersected_cells, x_min, x_max, y_min, y_max, z_min, z_max, 
                  f"{title} - 3D View")
    
    # XY view (top-down, looking along Z-axis)
    ax2 = fig.add_subplot(222)
    _draw_orthogonal_view(ax2, raytracer, intersected_cells, 'xy', x_min, x_max, y_min, y_max, 
                          f"{title} - XY View (Top)")
    
    # XZ view (side view, looking along Y-axis)
    ax3 = fig.add_subplot(223)
    _draw_orthogonal_view(ax3, raytracer, intersected_cells, 'xz', x_min, x_max, z_min, z_max, 
                          f"{title} - XZ View (Side)")
    
    # YZ view (front view, looking along X-axis)
    ax4 = fig.add_subplot(224)
    _draw_orthogonal_view(ax4, raytracer, intersected_cells, 'yz', y_min, y_max, z_min, z_max, 
                          f"{title} - YZ View (Front)")
    
    plt.tight_layout()
    plt.show()
    
    return intersected_cells

def _draw_3d_view(ax, raytracer, intersected_cells, x_min, x_max, y_min, y_max, z_min, z_max, title):
    """Draw the 3D perspective view."""
    # Draw intersected cells as semi-transparent cubes
    for cell in intersected_cells:
        if len(cell) == 3:  # Ensure it's 3D
            x, y, z = int(cell[0]), int(cell[1]), int(cell[2])  # Ensure integer coordinates
            _draw_cube(ax, x, y, z, color='lightblue', alpha=0.4)
    
    # Draw the ray line
    start_x, start_y, start_z = raytracer.start_coords[0], raytracer.start_coords[1], raytracer.start_coords[2]
    end_x, end_y, end_z = raytracer.end_coords[0], raytracer.end_coords[1], raytracer.end_coords[2]
    
    ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 
            'r-', linewidth=4, label='Ray Path', alpha=0.9)
    
    # Mark start and end points
    ax.scatter(start_x, start_y, start_z, color='green', s=150, label='Start', alpha=1.0)
    ax.scatter(end_x, end_y, end_z, color='red', s=150, label='End', alpha=1.0)
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_aspect('equal')  
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    
    # Force integer ticks only
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_yticks(range(int(y_min), int(y_max) + 1))
    ax.set_zticks(range(int(z_min), int(z_max) + 1))

def _draw_orthogonal_view(ax, raytracer, intersected_cells, view_type, min1, max1, min2, max2, title):
    """Draw an orthogonal 2D projection view."""
    # Determine which coordinates to use based on view type
    if view_type == 'xy':
        coord1_idx, coord2_idx = 0, 1  # X, Y
        xlabel, ylabel = 'X', 'Y'
    elif view_type == 'xz':
        coord1_idx, coord2_idx = 0, 2  # X, Z
        xlabel, ylabel = 'X', 'Z'
    else:  # yz
        coord1_idx, coord2_idx = 1, 2  # Y, Z
        xlabel, ylabel = 'Y', 'Z'
    
    # Draw grid lines
    for i in range(int(min1), int(max1) + 1):
        ax.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.7)
    for i in range(int(min2), int(max2) + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Color intersected cells (project to 2D)
    projected_cells = set()
    for cell in intersected_cells:
        if len(cell) == 3:
            coord1, coord2 = int(cell[coord1_idx]), int(cell[coord2_idx])
            projected_cells.add((coord1, coord2))
    
    for coord1, coord2 in projected_cells:
        rect = patches.Rectangle((coord1, coord2), 1, 1, 
                               linewidth=1.5, edgecolor='blue', 
                               facecolor='lightblue', alpha=0.6)
        ax.add_patch(rect)
    
    # Draw the ray line (projected)
    start_coord1 = raytracer.start_coords[coord1_idx]
    start_coord2 = raytracer.start_coords[coord2_idx]
    end_coord1 = raytracer.end_coords[coord1_idx]
    end_coord2 = raytracer.end_coords[coord2_idx]
    
    ax.plot([start_coord1, end_coord1], [start_coord2, end_coord2], 
            'r-', linewidth=3, label='Ray Path', alpha=0.8)
    
    # Mark start and end points
    ax.plot(start_coord1, start_coord2, 'go', markersize=10, label='Start', zorder=5)
    ax.plot(end_coord1, end_coord2, 'ro', markersize=10, label='End', zorder=5)
    
    # Set axis properties
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Force integer ticks only
    ax.set_xticks(range(int(min1), int(max1) + 1))
    ax.set_yticks(range(int(min2), int(max2) + 1))
    
    # Add coordinate info
    info_text = f'Cells: {len(projected_cells)}\n'
    info_text += f'Start: ({start_coord1:.1f}, {start_coord2:.1f})\n'
    info_text += f'End: ({end_coord1:.1f}, {end_coord2:.1f})'
    
    ax.text(0.02, 0.02, info_text, 
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def _draw_cube(ax, x, y, z, color='blue', alpha=0.3):
    """Draw a 3D cube at the given grid position."""
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
