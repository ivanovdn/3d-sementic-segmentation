"""
Opening Detection using Gap Analysis

Detects doors and windows in walls by analyzing gaps in point clouds.

Approach:
1. Project wall points onto 2D plane (wall direction × height)
2. Create density grid
3. Find rectangular gaps (low density regions)
4. Classify: door (touches floor) vs window (floating)
5. Measure dimensions

"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import ndimage
from scipy.ndimage import label, binary_opening, binary_closing
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def project_wall_to_2d(wall: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Project 3D wall points onto 2D plane (along wall × height).
    
    Parameters:
    -----------
    wall : dict
        Wall with 'points' (N, 3) and 'plane_model' or 'normal'
        
    Returns:
    --------
    points_2d : np.ndarray (N, 2)
        Points projected to (along_wall, height) coordinates
    transform : dict
        Transform info to convert back to 3D
    """
    
    points = wall['points']
    
    # Get wall normal (horizontal direction)
    if 'normal' in wall:
        normal = wall['normal'][:2]  # Take X, Y components
    else:
        plane = wall['plane_model']
        normal = np.array([plane[0], plane[1]])
    
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    
    # Wall direction is perpendicular to normal in XY plane
    wall_direction = np.array([-normal[1], normal[0]])
    
    # Project points
    points_xy = points[:, :2]
    centroid_xy = points_xy.mean(axis=0)
    
    # Along-wall coordinate (horizontal along wall)
    along_wall = np.dot(points_xy - centroid_xy, wall_direction)
    
    # Height coordinate (Z)
    height = points[:, 2]
    
    points_2d = np.column_stack([along_wall, height])
    
    transform = {
        'centroid_xy': centroid_xy,
        'wall_direction': wall_direction,
        'normal': normal,
        'z_min': height.min(),
        'z_max': height.max()
    }
    
    return points_2d, transform


def create_density_grid(points_2d: np.ndarray,
                        cell_size: float = 0.05,
                        min_x: float = None,
                        max_x: float = None,
                        min_z: float = None,
                        max_z: float = None) -> Tuple[np.ndarray, Dict]:
    """
    Create 2D density grid from projected points.
    
    Parameters:
    -----------
    points_2d : np.ndarray (N, 2)
        Points in (along_wall, height) coordinates
    cell_size : float
        Grid cell size in meters
    min_x, max_x, min_z, max_z : float
        Grid bounds (optional, computed from data if not provided)
        
    Returns:
    --------
    grid : np.ndarray (H, W)
        Density grid (point counts per cell)
    grid_info : dict
        Grid metadata for coordinate conversion
    """
    
    x = points_2d[:, 0]  # Along wall
    z = points_2d[:, 1]  # Height
    
    # Grid bounds
    if min_x is None:
        min_x = x.min() - cell_size
    if max_x is None:
        max_x = x.max() + cell_size
    if min_z is None:
        min_z = z.min() - cell_size
    if max_z is None:
        max_z = z.max() + cell_size
    
    # Grid dimensions
    n_cols = int(np.ceil((max_x - min_x) / cell_size))
    n_rows = int(np.ceil((max_z - min_z) / cell_size))
    
    # Create grid
    grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    # Fill grid with point counts
    col_idx = ((x - min_x) / cell_size).astype(int)
    row_idx = ((z - min_z) / cell_size).astype(int)
    
    # Clip to valid range
    col_idx = np.clip(col_idx, 0, n_cols - 1)
    row_idx = np.clip(row_idx, 0, n_rows - 1)
    
    # Note: row 0 is at min_z (bottom), so we don't need to flip
    for r, c in zip(row_idx, col_idx):
        grid[r, c] += 1
    
    grid_info = {
        'cell_size': cell_size,
        'min_x': min_x,
        'max_x': max_x,
        'min_z': min_z,
        'max_z': max_z,
        'n_rows': n_rows,
        'n_cols': n_cols
    }
    
    return grid, grid_info


def find_gaps(grid: np.ndarray,
              grid_info: Dict,
              density_threshold: float = 0.1,
              min_gap_width: float = 0.5,
              min_gap_height: float = 0.5,
              morphology_size: int = 3) -> List[Dict]:
    """
    Find rectangular gaps in density grid.
    
    Parameters:
    -----------
    grid : np.ndarray
        Density grid
    grid_info : dict
        Grid metadata
    density_threshold : float
        Cells with density below this fraction of max are considered "empty"
    min_gap_width : float
        Minimum gap width in meters
    min_gap_height : float
        Minimum gap height in meters
        
    Returns:
    --------
    gaps : list of dict
        Each gap has bounding box and properties
    """
    
    cell_size = grid_info['cell_size']
    min_x = grid_info['min_x']
    min_z = grid_info['min_z']
    
    # Normalize grid
    max_density = grid.max()
    if max_density == 0:
        return []
    
    normalized = grid / max_density
    
    # Create binary mask: 1 = has points, 0 = empty
    has_points = (normalized > density_threshold).astype(np.uint8)
    
    # Apply morphological operations to clean up noise
    # Close small holes in walls
    struct = np.ones((morphology_size, morphology_size))
    has_points_clean = binary_closing(has_points, structure=struct, iterations=2)
    
    # Find empty regions (potential openings)
    empty_mask = ~has_points_clean
    
    # Open to remove small noise gaps
    empty_mask = binary_opening(empty_mask, structure=struct, iterations=1)
    
    # Label connected components
    labeled, n_components = label(empty_mask)
    
    gaps = []
    
    for i in range(1, n_components + 1):
        component = (labeled == i)
        
        # Find bounding box
        rows, cols = np.where(component)
        if len(rows) == 0:
            continue
        
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        
        # Convert to real coordinates
        x_min = min_x + col_min * cell_size
        x_max = min_x + (col_max + 1) * cell_size
        z_min = min_z + row_min * cell_size
        z_max = min_z + (row_max + 1) * cell_size
        
        width = x_max - x_min
        height = z_max - z_min
        
        # Filter by size
        if width < min_gap_width or height < min_gap_height:
            continue
        
        # Check if gap touches edges (might be boundary, not opening)
        touches_left = col_min <= 1
        touches_right = col_max >= grid_info['n_cols'] - 2
        touches_bottom = row_min <= 1
        touches_top = row_max >= grid_info['n_rows'] - 2
        
        # Skip gaps that touch left/right edges (wall boundaries)
        if touches_left or touches_right:
            continue
        
        gaps.append({
            'x_min': x_min,
            'x_max': x_max,
            'z_min': z_min,
            'z_max': z_max,
            'width': width,
            'height': height,
            'area': width * height,
            'touches_bottom': touches_bottom,
            'touches_top': touches_top,
            'grid_bbox': (row_min, row_max, col_min, col_max),
            'component_mask': component
        })
    
    return gaps


def classify_openings(gaps: List[Dict],
                      floor_height: float,
                      ceiling_height: float,
                      floor_tolerance: float = 0.15,
                      min_door_height: float = 1.8,
                      max_door_height: float = 2.5,
                      min_door_width: float = 0.6,
                      max_door_width: float = 1.5,
                      min_window_height: float = 0.5,
                      min_window_width: float = 0.4) -> List[Dict]:
    """
    Classify gaps as doors or windows.
    
    Parameters:
    -----------
    gaps : list of dict
        Detected gaps
    floor_height : float
        Z coordinate of floor
    ceiling_height : float
        Z coordinate of ceiling
    floor_tolerance : float
        How close to floor to be considered "touching floor"
    min_door_height, max_door_height : float
        Door height range
    min_door_width, max_door_width : float
        Door width range
    min_window_height, min_window_width : float
        Minimum window dimensions
        
    Returns:
    --------
    openings : list of dict
        Classified openings with type and dimensions
    """
    
    openings = []
    
    for gap in gaps:
        # Check if touches floor
        near_floor = abs(gap['z_min'] - floor_height) < floor_tolerance
        
        # Check if touches ceiling
        near_ceiling = abs(gap['z_max'] - ceiling_height) < floor_tolerance
        
        width = gap['width']
        height = gap['height']
        
        opening = {
            'x_min': gap['x_min'],
            'x_max': gap['x_max'],
            'z_min': gap['z_min'],
            'z_max': gap['z_max'],
            'width': width,
            'height': height,
            'near_floor': near_floor,
            'near_ceiling': near_ceiling
        }
        
        # Classify
        if near_floor and not near_ceiling:
            # Touches floor but not ceiling = likely door
            if (min_door_width <= width <= max_door_width and
                min_door_height <= height <= max_door_height):
                opening['type'] = 'door'
                opening['confidence'] = 'high'
            elif min_door_height <= height:
                opening['type'] = 'door'
                opening['confidence'] = 'medium'
            else:
                opening['type'] = 'unknown'
                opening['confidence'] = 'low'
        
        elif not near_floor and not near_ceiling:
            # Floating = likely window
            if (min_window_width <= width and min_window_height <= height):
                opening['type'] = 'window'
                opening['confidence'] = 'high'
            else:
                opening['type'] = 'window'
                opening['confidence'] = 'medium'
        
        elif near_floor and near_ceiling:
            # Floor to ceiling = could be doorway or full opening
            opening['type'] = 'full_opening'
            opening['confidence'] = 'medium'
        
        else:
            opening['type'] = 'unknown'
            opening['confidence'] = 'low'
        
        openings.append(opening)
    
    return openings


def detect_openings_in_wall(wall: Dict,
                            floor_height: float,
                            ceiling_height: float,
                            cell_size: float = 0.05,
                            density_threshold: float = 0.1,
                            min_gap_width: float = 0.5,
                            min_gap_height: float = 0.5,
                            verbose: bool = True) -> List[Dict]:
    """
    Detect openings (doors, windows) in a single wall.
    
    Parameters:
    -----------
    wall : dict
        Wall segment with 'points'
    floor_height : float
        Z coordinate of floor
    ceiling_height : float
        Z coordinate of ceiling
    cell_size : float
        Grid cell size for density analysis
    density_threshold : float
        Threshold for empty cells
    min_gap_width : float
        Minimum opening width
    min_gap_height : float
        Minimum opening height
    verbose : bool
        Print progress
        
    Returns:
    --------
    openings : list of dict
        Detected openings with type, position, and dimensions
    """
    
    if verbose:
        print(f"\n  Analyzing wall with {len(wall['points'])} points...")
    
    # Step 1: Project to 2D
    points_2d, transform = project_wall_to_2d(wall)
    
    if verbose:
        print(f"    Projected to 2D: x=[{points_2d[:, 0].min():.2f}, {points_2d[:, 0].max():.2f}], "
              f"z=[{points_2d[:, 1].min():.2f}, {points_2d[:, 1].max():.2f}]")
    
    # Step 2: Create density grid
    grid, grid_info = create_density_grid(
        points_2d, 
        cell_size=cell_size,
        min_z=floor_height,
        max_z=ceiling_height
    )
    
    if verbose:
        print(f"    Grid: {grid_info['n_rows']}×{grid_info['n_cols']} cells")
    
    # Step 3: Find gaps
    gaps = find_gaps(
        grid, grid_info,
        density_threshold=density_threshold,
        min_gap_width=min_gap_width,
        min_gap_height=min_gap_height
    )
    
    if verbose:
        print(f"    Found {len(gaps)} gaps")
    
    # Step 4: Classify openings
    openings = classify_openings(
        gaps,
        floor_height=floor_height,
        ceiling_height=ceiling_height
    )
    
    # Add wall reference and transform info
    for opening in openings:
        opening['wall_transform'] = transform
        opening['grid_info'] = grid_info
    
    if verbose:
        for i, op in enumerate(openings):
            print(f"    Opening {i}: {op['type']} ({op['confidence']}) - "
                  f"W={op['width']:.2f}m × H={op['height']:.2f}m")
    
    return openings


def detect_all_openings(walls: List[Dict],
                        floor_height: float,
                        ceiling_height: float,
                        # Grid params
                        cell_size: float = 0.05,
                        density_threshold: float = 0.1,
                        # Gap detection params
                        min_gap_width: float = 0.5,
                        min_gap_height: float = 0.5,
                        morphology_size: int = 3,
                        # Classification params
                        floor_tolerance: float = 0.15,
                        min_door_height: float = 1.8,
                        max_door_height: float = 2.5,
                        min_door_width: float = 0.6,
                        max_door_width: float = 1.5,
                        min_window_height: float = 0.5,
                        min_window_width: float = 0.4,
                        verbose: bool = True) -> List[Dict]:
    """
    Detect openings in all walls.
    
    Parameters:
    -----------
    walls : list of dict
        Wall segments
    floor_height : float
        Z coordinate of floor
    ceiling_height : float
        Z coordinate of ceiling
    cell_size : float
        Grid cell size for density analysis
    density_threshold : float
        Threshold for empty cells (fraction of max density)
    min_gap_width : float
        Minimum opening width in meters
    min_gap_height : float
        Minimum opening height in meters
    morphology_size : int
        Kernel size for morphological operations
    floor_tolerance : float
        Distance from floor to be considered "touching floor"
    min_door_height, max_door_height : float
        Door height range for classification
    min_door_width, max_door_width : float
        Door width range for classification
    min_window_height, min_window_width : float
        Minimum window dimensions
    verbose : bool
        Print progress
        
    Returns:
    --------
    all_openings : list of dict
        All detected openings with wall index
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"OPENING DETECTION")
        print(f"{'='*70}")
        print(f"  Walls: {len(walls)}")
        print(f"  Floor: {floor_height:.2f}m, Ceiling: {ceiling_height:.2f}m")
        print(f"  Grid cell size: {cell_size}m")
        print(f"  Min gap size: {min_gap_width}m × {min_gap_height}m")
    
    all_openings = []
    
    for i, wall in enumerate(walls):
        if verbose:
            print(f"\n  Wall {i}: {len(wall['points'])} points")
        
        # Project to 2D
        points_2d, transform = project_wall_to_2d(wall)
        
        # Create density grid
        grid, grid_info = create_density_grid(
            points_2d,
            cell_size=cell_size,
            min_z=floor_height,
            max_z=ceiling_height
        )
        
        # Find gaps
        gaps = find_gaps(
            grid, grid_info,
            density_threshold=density_threshold,
            min_gap_width=min_gap_width,
            min_gap_height=min_gap_height,
            morphology_size=morphology_size
        )
        
        # Classify openings
        openings = classify_openings(
            gaps,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            floor_tolerance=floor_tolerance,
            min_door_height=min_door_height,
            max_door_height=max_door_height,
            min_door_width=min_door_width,
            max_door_width=max_door_width,
            min_window_height=min_window_height,
            min_window_width=min_window_width
        )
        
        # Add metadata
        for op in openings:
            op['wall_index'] = i
            op['wall_transform'] = transform
            op['grid_info'] = grid_info
            all_openings.append(op)
            
            if verbose:
                print(f"    → {op['type']} ({op['confidence']}): "
                      f"{op['width']:.2f}m × {op['height']:.2f}m")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TOTAL OPENINGS: {len(all_openings)}")
        doors = [o for o in all_openings if o['type'] == 'door']
        windows = [o for o in all_openings if o['type'] == 'window']
        print(f"  Doors: {len(doors)}")
        print(f"  Windows: {len(windows)}")
        print(f"{'='*70}")
    
    return all_openings


def visualize_wall_openings(wall: Dict,
                            openings: List[Dict],
                            floor_height: float,
                            ceiling_height: float,
                            cell_size: float = 0.05,
                            title: str = "Wall Opening Detection",
                            save_path: str = None):
    """
    Visualize wall with detected openings.
    """
    
    # Project wall to 2D
    points_2d, transform = project_wall_to_2d(wall)
    
    # Create density grid
    grid, grid_info = create_density_grid(
        points_2d,
        cell_size=cell_size,
        min_z=floor_height,
        max_z=ceiling_height
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Point cloud
    ax1 = axes[0]
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=1, alpha=0.5, c='blue')
    ax1.set_xlabel('Along Wall (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Wall Points (2D projection)')
    ax1.set_aspect('equal')
    ax1.axhline(y=floor_height, color='brown', linestyle='--', label='Floor')
    ax1.axhline(y=ceiling_height, color='gray', linestyle='--', label='Ceiling')
    
    # Draw openings
    colors = {'door': 'red', 'window': 'green', 'full_opening': 'orange', 'unknown': 'gray'}
    for op in openings:
        color = colors.get(op['type'], 'gray')
        rect = patches.Rectangle(
            (op['x_min'], op['z_min']),
            op['width'], op['height'],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
        )
        ax1.add_patch(rect)
        ax1.text(
            (op['x_min'] + op['x_max']) / 2,
            (op['z_min'] + op['z_max']) / 2,
            f"{op['type']}\n{op['width']:.2f}×{op['height']:.2f}m",
            ha='center', va='center', fontsize=9, fontweight='bold'
        )
    
    ax1.legend()
    
    # Right: Density grid
    ax2 = axes[1]
    extent = [grid_info['min_x'], grid_info['max_x'], 
              grid_info['min_z'], grid_info['max_z']]
    im = ax2.imshow(grid, origin='lower', extent=extent, 
                    cmap='hot', aspect='auto')
    ax2.set_xlabel('Along Wall (m)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Density Grid')
    plt.colorbar(im, ax=ax2, label='Point count')
    
    # Draw opening boxes on density grid
    for op in openings:
        color = colors.get(op['type'], 'gray')
        rect = patches.Rectangle(
            (op['x_min'], op['z_min']),
            op['width'], op['height'],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Opening Detection Module")
    print("=" * 50)
    
    # Create synthetic wall with door and window
    np.random.seed(42)
    
    wall_width = 4.0
    wall_height = 2.5
    floor_z = 0.0
    ceiling_z = 2.5
    n_points = 5000
    
    # Generate wall points
    x = np.random.uniform(0, wall_width, n_points)
    z = np.random.uniform(floor_z, ceiling_z, n_points)
    y = np.random.normal(0, 0.02, n_points)  # Wall at y=0 with noise
    
    # Remove points for door (x: 0.5-1.3, z: 0-2.1)
    door_mask = (x > 0.5) & (x < 1.3) & (z < 2.1)
    
    # Remove points for window (x: 2.5-3.5, z: 0.8-1.8)
    window_mask = (x > 2.5) & (x < 3.5) & (z > 0.8) & (z < 1.8)
    
    keep_mask = ~door_mask & ~window_mask
    
    points = np.column_stack([x[keep_mask], y[keep_mask], z[keep_mask]])
    
    print(f"\nSynthetic wall: {len(points)} points")
    print(f"  Door: 0.8m × 2.1m at x=0.5-1.3")
    print(f"  Window: 1.0m × 1.0m at x=2.5-3.5, z=0.8-1.8")
    
    # Create wall dict
    wall = {
        'points': points,
        'normal': np.array([0, 1, 0])  # Normal pointing in Y direction
    }
    
    # Detect openings
    openings = detect_openings_in_wall(
        wall,
        floor_height=floor_z,
        ceiling_height=ceiling_z,
        cell_size=0.05,
        verbose=True
    )
    
    print(f"\n✓ Detected {len(openings)} openings")
    for i, op in enumerate(openings):
        print(f"\n  Opening {i}: {op['type']}")
        print(f"    Width: {op['width']:.2f}m")
        print(f"    Height: {op['height']:.2f}m")
        print(f"    Position: x=[{op['x_min']:.2f}, {op['x_max']:.2f}], z=[{op['z_min']:.2f}, {op['z_max']:.2f}]")
