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
            x, y = cell[0], cell[1]
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
    
    # Add text showing number of intersected cells
    ax.text(0.02, 0.98, f'Intersected cells: {len(intersected_cells)}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return intersected_cells

def _visualize_3d(raytracer, intersected_cells, title):
    """Visualize 3D raytracer."""
    if raytracer.grid is None:
        print("Raytracer initialization failed")
        return intersected_cells
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
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
    
    # Draw intersected cells as semi-transparent cubes
    for cell in intersected_cells:
        if len(cell) == 3:  # Ensure it's 3D
            x, y, z = cell[0], cell[1], cell[2]
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
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    
    # Add text showing number of intersected cells
    ax.text2D(0.02, 0.98, f'Intersected cells: {len(intersected_cells)}', 
              transform=ax.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return intersected_cells

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
