"""
Clean functions for aligning RoomPlan geometry with point cloud
Use these functions one by one to transform your data
"""

import numpy as np
import json


# ============================================================================
# FUNCTION 1: Load point cloud and convert Y-up to Z-up
# ============================================================================

def load_pointcloud_yup_to_zup(ply_path, voxel_size=0.02):
    """
    Load point cloud from PLY (ARKit Y-up) and convert to Z-up
    
    Args:
        ply_path: Path to PLY file
        voxel_size: Voxel size for downsampling (meters), None to skip
        
    Returns:
        points_zup: numpy array (N, 3) - points in Z-up coordinates
        colors: numpy array (N, 3) - RGB colors [0-1]
    """
    import open3d as o3d
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Optional downsampling
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    
    # Get data
    points_yup = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Apply Y-up to Z-up rotation
    R_yup_to_zup = np.array([
        [1, 0, 0],  # X stays X
        [0, 0, 1],  # ARKit Z becomes Y
        [0, 1, 0]   # ARKit Y becomes Z
    ])
    
    points_zup = points_yup @ R_yup_to_zup.T
    
    return points_zup, colors


# ============================================================================
# FUNCTION 2: Get transformation matrix for RoomPlan geometry
# ============================================================================

def get_roomplan_transform_matrix(room_json_path):
    """
    Get the transformation matrix to convert RoomPlan geometry to Z-up space
    
    This creates a combined matrix that transforms:
    Floor/Wall Local Coords → ARKit World (Y-up) → Z-up
    
    Args:
        room_json_path: Path to room.json file
        
    Returns:
        T_combined: 4x4 transformation matrix
        room_data: dict with full room data
    """
    with open(room_json_path, 'r') as f:
        room_data = json.load(f)
    
    # Get floor transform (stored in column-major format)
    floor_transform_flat = room_data['floors'][0]['transform']
    T_floor = np.array(floor_transform_flat).reshape(4, 4).T
    
    # Create Y-up to Z-up transform
    R_yup_to_zup = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    T_yup_to_zup = np.eye(4)
    T_yup_to_zup[:3, :3] = R_yup_to_zup
    
    # Combine: Local → ARKit (Y-up) → Z-up
    # NOTE: We skip the referenceOriginTransform!
    T_combined = T_yup_to_zup @ T_floor
    
    return T_combined, room_data


# ============================================================================
# FUNCTION 3: Transform RoomPlan floor corners to Z-up space
# ============================================================================

def get_floor_corners_zup(room_data, T_combined):
    """
    Transform RoomPlan floor corners to Z-up point cloud space
    
    Args:
        room_data: dict from room.json
        T_combined: 4x4 transformation matrix from get_roomplan_transform_matrix()
        
    Returns:
        corners_zup: numpy array (4, 3) - floor corners in Z-up space
        dimensions: [width, height, 0] in meters
    """
    # Get floor corners in local coordinates
    floor_corners_local = np.array(room_data['floors'][0]['polygonCorners'])
    floor_dimensions = room_data['floors'][0]['dimensions']
    
    # Transform each corner
    corners_zup = []
    for corner in floor_corners_local:
        corner_homogeneous = np.append(corner, 1)  # Add 1 for 4x4 matrix multiplication
        corner_transformed = (T_combined @ corner_homogeneous)[:3]
        corners_zup.append(corner_transformed)
    
    corners_zup = np.array(corners_zup)
    
    return corners_zup, floor_dimensions


# ============================================================================
# FUNCTION 4: Transform RoomPlan walls to Z-up space
# ============================================================================

def get_walls_zup(room_data, T_floor_to_zup):
    """
    Transform RoomPlan walls to Z-up point cloud space
    
    Args:
        room_data: dict from room.json
        T_floor_to_zup: 4x4 transformation matrix (from get_roomplan_transform_matrix)
        
    Returns:
        walls: list of dicts, each containing:
            - center_zup: (3,) wall center position
            - dimensions: [length, height, thickness]
            - normal_zup: (3,) wall normal vector
            - identifier: wall ID
    """
    walls_transformed = []
    
    # Y-up to Z-up rotation (3x3)
    R_yup_to_zup = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    
    for wall in room_data['walls']:
        # Get wall transform (column-major)
        wall_transform_flat = wall['transform']
        T_wall = np.array(wall_transform_flat).reshape(4, 4).T
        
        # Transform: Wall Local → ARKit (Y-up) → Z-up
        T_wall_zup = T_floor_to_zup @ np.linalg.inv(T_floor_to_zup) @ (R_yup_to_zup @ T_wall[:3, :3])
        
        # Simpler: just apply Y-up to Z-up to the wall transform
        T_wall_zup_full = np.eye(4)
        T_wall_zup_full[:3, :3] = R_yup_to_zup
        T_wall_final = T_wall_zup_full @ T_wall
        
        # Get wall center in Z-up
        wall_center_yup = T_wall[:3, 3]
        wall_center_zup = R_yup_to_zup @ wall_center_yup
        
        # Get wall normal (assuming wall faces along local X-axis)
        wall_normal_local = np.array([1, 0, 0])
        wall_rotation = T_wall[:3, :3]
        wall_normal_yup = wall_rotation @ wall_normal_local
        wall_normal_zup = R_yup_to_zup @ wall_normal_yup
        
        walls_transformed.append({
            'center_zup': wall_center_zup,
            'dimensions': wall['dimensions'],
            'normal_zup': wall_normal_zup,
            'identifier': wall['identifier'],
            'transform_zup': T_wall_final
        })
    
    return walls_transformed


# ============================================================================
# FUNCTION 5: Transform any RoomPlan geometry (generic)
# ============================================================================

def transform_roomplan_geometry_to_zup(geometry_transform_flat):
    """
    Transform any RoomPlan geometry (wall, window, door, object) to Z-up space
    
    Args:
        geometry_transform_flat: 16-element list (transform from room.json)
        
    Returns:
        T_geometry_zup: 4x4 transformation matrix in Z-up space
        center_zup: (3,) center position in Z-up
    """
    # Parse transform (column-major)
    T_geometry_yup = np.array(geometry_transform_flat).reshape(4, 4).T
    
    # Y-up to Z-up transform
    R_yup_to_zup = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    T_yup_to_zup = np.eye(4)
    T_yup_to_zup[:3, :3] = R_yup_to_zup
    
    # Transform to Z-up
    T_geometry_zup = T_yup_to_zup @ T_geometry_yup
    center_zup = T_geometry_zup[:3, 3]
    
    return T_geometry_zup, center_zup


# ============================================================================
# HELPER: Create Y-up to Z-up rotation matrix
# ============================================================================

def get_yup_to_zup_rotation():
    """
    Get the rotation matrix to convert Y-up (ARKit) to Z-up coordinates
    
    Returns:
        R: 3x3 rotation matrix
    """
    return np.array([
        [1, 0, 0],  # X stays X
        [0, 0, 1],  # Z becomes Y
        [0, 1, 0]   # Y becomes Z
    ])


# ============================================================================
# HELPER: Parse RoomPlan transform matrix
# ============================================================================

def parse_roomplan_transform(transform_flat):
    """
    Parse a RoomPlan transform from column-major flat list to numpy matrix
    
    Args:
        transform_flat: 16-element list from room.json
        
    Returns:
        T: 4x4 numpy transformation matrix
        R: 3x3 rotation part
        t: (3,) translation part
    """
    # iOS/ARKit uses column-major format
    T = np.array(transform_flat).reshape(4, 4).T
    R = T[:3, :3]
    t = T[:3, 3]
    
    return T, R, t


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EXAMPLE USAGE - Step by Step")
    print("="*80)
    print()
    
    # Example paths
    ply_path = "/mnt/user-data/uploads/alex1.ply"
    room_json_path = "/mnt/user-data/uploads/room.json"
    
    print("STEP 1: Load and transform point cloud")
    print("-" * 80)
    points_zup, colors = load_pointcloud_yup_to_zup(ply_path, voxel_size=0.02)
    print(f"Loaded {len(points_zup)} points in Z-up coordinates")
    print(f"Bounds: X[{points_zup[:,0].min():.2f}, {points_zup[:,0].max():.2f}], "
          f"Y[{points_zup[:,1].min():.2f}, {points_zup[:,1].max():.2f}], "
          f"Z[{points_zup[:,2].min():.2f}, {points_zup[:,2].max():.2f}]")
    print()
    
    print("STEP 2: Get RoomPlan transformation matrix")
    print("-" * 80)
    T_combined, room_data = get_roomplan_transform_matrix(room_json_path)
    print("Transformation matrix (4x4):")
    print(T_combined)
    print()
    
    print("STEP 3: Transform floor corners")
    print("-" * 80)
    floor_corners_zup, floor_dims = get_floor_corners_zup(room_data, T_combined)
    print(f"Floor dimensions: {floor_dims[0]:.3f}m × {floor_dims[1]:.3f}m")
    print("Floor corners in Z-up space:")
    for i, corner in enumerate(floor_corners_zup):
        print(f"  Corner {i}: [{corner[0]:7.4f}, {corner[1]:7.4f}, {corner[2]:7.4f}]")
    print()
    
    print("STEP 4: Transform windows")
    print("-" * 80)
    if 'windows' in room_data and len(room_data['windows']) > 0:
        for i, window in enumerate(room_data['windows']):
            T_window_zup, center_zup = transform_roomplan_geometry_to_zup(window['transform'])
            dims = window['dimensions']
            print(f"Window {i}:")
            print(f"  Center: [{center_zup[0]:7.4f}, {center_zup[1]:7.4f}, {center_zup[2]:7.4f}]")
            print(f"  Dimensions: {dims[0]:.3f}m × {dims[1]:.3f}m")
    else:
        print("No windows found")
    print()
    
    print("="*80)
    print("Now points_zup and floor_corners_zup are in the same coordinate system!")
    print("You can visualize them together or measure dimensions.")
    print("="*80)
