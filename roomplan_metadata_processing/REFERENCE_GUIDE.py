"""
QUICK REFERENCE GUIDE
How to align RoomPlan with Point Cloud - Function by Function
"""

# ============================================================================
# FUNCTION 1: Load Point Cloud (Y-up → Z-up)
# ============================================================================

"""
PURPOSE: Load your ARKit point cloud and convert from Y-up to Z-up

INPUT:
  - ply_path: path to your .ply file
  - voxel_size: downsample size in meters (optional, use None to skip)

OUTPUT:
  - points_zup: numpy array (N, 3) of XYZ coordinates in Z-up
  - colors: numpy array (N, 3) of RGB values [0-1]

USAGE:
"""
from roomplan_alignment_functions import load_pointcloud_yup_to_zup

points_zup, colors = load_pointcloud_yup_to_zup(
    ply_path="path/to/scan.ply",
    voxel_size=0.02  # 2cm downsampling
)

print(f"Loaded {len(points_zup)} points")
# points_zup is now in Z-up coordinates (X-right, Y-forward, Z-up)


# ============================================================================
# FUNCTION 2: Get RoomPlan Transform Matrix
# ============================================================================

"""
PURPOSE: Get the transformation matrix that converts RoomPlan geometry to Z-up

INPUT:
  - room_json_path: path to room.json from RoomPlan

OUTPUT:
  - T_combined: 4×4 transformation matrix (Local → Z-up)
  - room_data: complete dict with all RoomPlan data

USAGE:
"""
from roomplan_alignment_functions import get_roomplan_transform_matrix

T_combined, room_data = get_roomplan_transform_matrix("path/to/room.json")

# T_combined is a 4×4 matrix that transforms:
# RoomPlan Local Coordinates → ARKit World (Y-up) → Z-up
# This skips the problematic referenceOriginTransform!


# ============================================================================
# FUNCTION 3: Get Floor Corners in Z-up
# ============================================================================

"""
PURPOSE: Transform the floor boundary corners to Z-up coordinates

INPUT:
  - room_data: dict from function 2
  - T_combined: transform matrix from function 2

OUTPUT:
  - corners_zup: numpy array (4, 3) with floor corner coordinates
  - dimensions: [width, height, 0] in meters

USAGE:
"""
from roomplan_alignment_functions import get_floor_corners_zup

floor_corners_zup, floor_dims = get_floor_corners_zup(room_data, T_combined)

print(f"Floor: {floor_dims[0]:.2f}m × {floor_dims[1]:.2f}m")
print(f"Corner 0: {floor_corners_zup[0]}")
# floor_corners_zup now matches your point cloud coordinate system!


# ============================================================================
# FUNCTION 4: Transform Any RoomPlan Geometry
# ============================================================================

"""
PURPOSE: Transform any RoomPlan element (wall, window, door, object) to Z-up

INPUT:
  - geometry_transform_flat: 16-element list from room.json

OUTPUT:
  - T_geometry_zup: 4×4 transformation matrix in Z-up
  - center_zup: (3,) center position in Z-up

USAGE:
"""
from roomplan_alignment_functions import transform_roomplan_geometry_to_zup

# Example: Transform all windows
for window in room_data['windows']:
    T_window, center = transform_roomplan_geometry_to_zup(window['transform'])
    dims = window['dimensions']
    print(f"Window at: {center}, size: {dims[0]:.2f}m × {dims[1]:.2f}m")

# Example: Transform all walls
for wall in room_data['walls']:
    T_wall, center = transform_roomplan_geometry_to_zup(wall['transform'])
    dims = wall['dimensions']
    print(f"Wall at: {center}, size: {dims[0]:.2f}m × {dims[1]:.2f}m")

# Example: Transform all doors
for door in room_data.get('doors', []):
    T_door, center = transform_roomplan_geometry_to_zup(door['transform'])
    dims = door['dimensions']
    print(f"Door at: {center}, size: {dims[0]:.2f}m × {dims[1]:.2f}m")


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

"""
COMPLETE EXAMPLE: Load everything and verify alignment
"""

import numpy as np
from roomplan_alignment_functions import (
    load_pointcloud_yup_to_zup,
    get_roomplan_transform_matrix,
    get_floor_corners_zup,
    transform_roomplan_geometry_to_zup
)

# Step 1: Load point cloud
print("Loading point cloud...")
points_zup, colors = load_pointcloud_yup_to_zup(
    "scan.ply",
    voxel_size=0.02
)

# Step 2: Get RoomPlan transform
print("Loading RoomPlan data...")
T_combined, room_data = get_roomplan_transform_matrix("room.json")

# Step 3: Get floor
print("Transforming floor...")
floor_corners, floor_dims = get_floor_corners_zup(room_data, T_combined)

# Step 4: Get all other geometry
print("Transforming windows...")
windows_zup = []
for window in room_data.get('windows', []):
    T_win, center = transform_roomplan_geometry_to_zup(window['transform'])
    windows_zup.append({
        'center': center,
        'dimensions': window['dimensions'],
        'transform': T_win
    })

print("Transforming walls...")
walls_zup = []
for wall in room_data.get('walls', []):
    T_wall, center = transform_roomplan_geometry_to_zup(wall['transform'])
    walls_zup.append({
        'center': center,
        'dimensions': wall['dimensions'],
        'transform': T_wall
    })

# Now everything is in the same Z-up coordinate system!
print("\nAlignment complete!")
print(f"Point cloud: {len(points_zup)} points")
print(f"Floor corners: {len(floor_corners)} corners")
print(f"Windows: {len(windows_zup)}")
print(f"Walls: {len(walls_zup)}")

# You can now:
# - Visualize them together
# - Measure dimensions from point cloud
# - Compare with RoomPlan measurements
# - Extract room geometry


# ============================================================================
# KEY POINTS TO REMEMBER
# ============================================================================

"""
1. COORDINATE SYSTEMS:
   - ARKit uses Y-up (Y=vertical)
   - We convert to Z-up (Z=vertical) for easier processing
   
2. TRANSFORMATION ORDER:
   - Point cloud: ARKit (Y-up) → Z-up
   - RoomPlan: Local → ARKit (Y-up) → Z-up
   - Both end up in the same Z-up space!
   
3. WHAT WE SKIP:
   - We DO NOT use referenceOriginTransform
   - It's RoomPlan's internal standardization
   - floor_transform already gives ARKit world coordinates
   
4. MATRIX FORMAT:
   - RoomPlan uses column-major format (iOS convention)
   - We transpose to row-major for numpy operations
   - This is handled automatically in the functions

5. HOMOGENEOUS COORDINATES:
   - 4×4 matrices work with [x, y, z, 1]
   - Append 1 to points before multiplying
   - Drop the 4th coordinate after transformation
"""
