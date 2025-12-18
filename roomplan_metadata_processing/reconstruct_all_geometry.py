"""
Functions to reconstruct RoomPlan geometry (walls, windows, openings) from transform + dimensions
Since they don't have polygonCorners, we create them programmatically
"""

import numpy as np


def reconstruct_rectangle_from_transform(transform_flat, dimensions):
    """
    Reconstruct a rectangle (wall/window/opening) from its transform and dimensions
    
    Args:
        transform_flat: 16-element list from room.json  
        dimensions: [width, height, thickness] in meters
        
    Returns:
        corners_zup: numpy array (4, 3) - rectangle corners in Z-up space
        center_zup: numpy array (3,) - rectangle center in Z-up space
        normal_zup: numpy array (3,) - normal vector (perpendicular to rectangle)
    """
    width, height, thickness = dimensions[0], dimensions[1], dimensions[2]
    
    # Create 4 corners in element's LOCAL coordinate system
    # Rectangle is in X-Y plane, centered at origin
    # X-axis = width, Y-axis = height, Z-axis = normal (perpendicular)
    half_w, half_h = width / 2, height / 2
    
    corners_local = np.array([
        [-half_w, -half_h, 0, 1],  # Bottom-left
        [ half_w, -half_h, 0, 1],  # Bottom-right
        [ half_w,  half_h, 0, 1],  # Top-right
        [-half_w,  half_h, 0, 1],  # Top-left
    ])
    
    # Parse transform (column-major format)
    T_element = np.array(transform_flat).reshape(4, 4).T
    
    # Transform to ARKit World (Y-up)
    corners_yup = (T_element @ corners_local.T).T[:, :3]
    
    # Get center and normal in Y-up
    center_yup = T_element[:3, 3]
    normal_local = np.array([0, 0, 1])  # Normal points along Z in local coords
    normal_yup = T_element[:3, :3] @ normal_local
    
    # Transform to Z-up
    R_yup_to_zup = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    
    corners_zup = corners_yup @ R_yup_to_zup.T
    center_zup = R_yup_to_zup @ center_yup
    normal_zup = R_yup_to_zup @ normal_yup
    
    return corners_zup, center_zup, normal_zup


def get_all_walls_zup(room_data):
    """
    Reconstruct all walls as rectangles in Z-up space
    
    Args:
        room_data: dict from room.json
        
    Returns:
        walls: list of dicts, each containing:
            - corners: (4, 3) numpy array of corner positions
            - center: (3,) numpy array of center position
            - normal: (3,) numpy array of normal vector
            - dimensions: [width, height, thickness]
            - identifier: wall ID string
    """
    walls = []
    
    for wall_data in room_data.get('walls', []):
        corners, center, normal = reconstruct_rectangle_from_transform(
            wall_data['transform'],
            wall_data['dimensions']
        )
        
        walls.append({
            'corners': corners,
            'center': center,
            'normal': normal,
            'dimensions': wall_data['dimensions'],
            'identifier': wall_data.get('identifier', 'unknown')
        })
    
    return walls


def get_all_windows_zup(room_data):
    """
    Reconstruct all windows as rectangles in Z-up space
    
    Args:
        room_data: dict from room.json
        
    Returns:
        windows: list of dicts with same structure as get_all_walls_zup()
    """
    windows = []
    
    for window_data in room_data.get('windows', []):
        corners, center, normal = reconstruct_rectangle_from_transform(
            window_data['transform'],
            window_data['dimensions']
        )
        
        windows.append({
            'corners': corners,
            'center': center,
            'normal': normal,
            'dimensions': window_data['dimensions'],
            'identifier': window_data.get('identifier', 'unknown')
        })
    
    return windows


def get_all_openings_zup(room_data):
    """
    Reconstruct all openings (doors/passages) as rectangles in Z-up space
    
    Args:
        room_data: dict from room.json
        
    Returns:
        openings: list of dicts with same structure as get_all_walls_zup()
    """
    openings = []
    
    for opening_data in room_data.get('openings', []):
        corners, center, normal = reconstruct_rectangle_from_transform(
            opening_data['transform'],
            opening_data['dimensions']
        )
        
        openings.append({
            'corners': corners,
            'center': center,
            'normal': normal,
            'dimensions': opening_data['dimensions'],
            'identifier': opening_data.get('identifier', 'unknown')
        })
    
    return openings


def get_all_roomplan_geometry_zup(room_json_path):
    """
    Get ALL RoomPlan geometry (floor, walls, windows, openings) in Z-up space
    
    This is the main function you should use!
    
    Args:
        room_json_path: path to room.json
        
    Returns:
        geometry: dict containing:
            - floor_corners: (4, 3) numpy array
            - floor_dimensions: [width, height, 0]
            - walls: list of wall dicts
            - windows: list of window dicts  
            - openings: list of opening dicts
    """
    import json
    
    with open(room_json_path, 'r') as f:
        room_data = json.load(f)
    
    # Floor (has polygonCorners)
    from roomplan_alignment_functions import get_roomplan_transform_matrix, get_floor_corners_zup
    T_combined, _ = get_roomplan_transform_matrix(room_json_path)
    floor_corners, floor_dims = get_floor_corners_zup(room_data, T_combined)
    
    # Walls, Windows, Openings (reconstruct from transform + dimensions)
    walls = get_all_walls_zup(room_data)
    windows = get_all_windows_zup(room_data)
    openings = get_all_openings_zup(room_data)
    
    return {
        'floor_corners': floor_corners,
        'floor_dimensions': floor_dims,
        'walls': walls,
        'windows': windows,
        'openings': openings
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import json
    
    room_json_path = "/mnt/user-data/uploads/room.json"
    
    print("="*80)
    print("RECONSTRUCTING ALL ROOMPLAN GEOMETRY")
    print("="*80)
    print()
    
    # Get everything
    geometry = get_all_roomplan_geometry_zup(room_json_path)
    
    print("FLOOR:")
    print("-" * 80)
    print(f"Dimensions: {geometry['floor_dimensions'][0]:.3f}m × {geometry['floor_dimensions'][1]:.3f}m")
    print(f"Corners: {len(geometry['floor_corners'])} points")
    for i, corner in enumerate(geometry['floor_corners']):
        print(f"  C{i}: [{corner[0]:7.4f}, {corner[1]:7.4f}, {corner[2]:7.4f}]")
    print()
    
    print("WALLS:")
    print("-" * 80)
    print(f"Total: {len(geometry['walls'])} walls")
    for i, wall in enumerate(geometry['walls']):
        dims = wall['dimensions']
        print(f"Wall {i}:")
        print(f"  Dimensions: {dims[0]:.3f}m × {dims[1]:.3f}m")
        print(f"  Center: [{wall['center'][0]:7.4f}, {wall['center'][1]:7.4f}, {wall['center'][2]:7.4f}]")
        print(f"  Normal: [{wall['normal'][0]:6.3f}, {wall['normal'][1]:6.3f}, {wall['normal'][2]:6.3f}]")
        print(f"  Corners:")
        for j, corner in enumerate(wall['corners']):
            print(f"    C{j}: [{corner[0]:7.4f}, {corner[1]:7.4f}, {corner[2]:7.4f}]")
        print()
    
    print("WINDOWS:")
    print("-" * 80)
    print(f"Total: {len(geometry['windows'])} windows")
    for i, window in enumerate(geometry['windows']):
        dims = window['dimensions']
        print(f"Window {i}:")
        print(f"  Dimensions: {dims[0]:.3f}m × {dims[1]:.3f}m")
        print(f"  Center: [{window['center'][0]:7.4f}, {window['center'][1]:7.4f}, {window['center'][2]:7.4f}]")
        print()
    
    print("OPENINGS:")
    print("-" * 80)
    print(f"Total: {len(geometry['openings'])} openings")
    for i, opening in enumerate(geometry['openings']):
        dims = opening['dimensions']
        print(f"Opening {i}:")
        print(f"  Dimensions: {dims[0]:.3f}m × {dims[1]:.3f}m")
        print(f"  Center: [{opening['center'][0]:7.4f}, {opening['center'][1]:7.4f}, {opening['center'][2]:7.4f}]")
        print()
    
    print("="*80)
    print("NOW YOU HAVE ALL GEOMETRY IN Z-UP SPACE!")
    print("="*80)
    print()
    print("You can:")
    print("  • Visualize walls as rectangles with point cloud")
    print("  • Measure wall dimensions from point cloud")
    print("  • Verify window/door positions")
    print("  • Compare RoomPlan vs actual geometry")
