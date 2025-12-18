"""
Interactive 3D Visualization: Complete Room + Point Cloud
Using Open3D viewer - you can rotate, zoom, and inspect
"""

import numpy as np
import open3d as o3d

from reconstruct_all_geometry import get_all_roomplan_geometry_zup
from roomplan_alignment_functions import load_pointcloud_yup_to_zup

print("=" * 80)
print("CREATING 3D INTERACTIVE VISUALIZATION")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load data
# ============================================================================

print("Loading point cloud...")
points_zup, colors = load_pointcloud_yup_to_zup("./data/alex1.ply", voxel_size=0.02)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_zup)
pcd.colors = o3d.utility.Vector3dVector(colors)
print(f"  {len(points_zup)} points")
print()

print("Loading RoomPlan geometry...")
geometry = get_all_roomplan_geometry_zup(
    "./data/scan_2025-11-28_13-03-52/model/room.json"
)
print(f"  Floor: {len(geometry['floor_corners'])} corners")
print(f"  Walls: {len(geometry['walls'])} walls")
print(f"  Windows: {len(geometry['windows'])} windows")
print(f"  Openings: {len(geometry['openings'])} openings")
print()

# ============================================================================
# STEP 2: Create 3D geometries
# ============================================================================

geometries_to_visualize = [pcd]

print("Creating 3D geometries...")
print()

# ============================================================================
# FLOOR - as mesh (filled rectangle)
# ============================================================================

print("Creating floor mesh...")
floor_corners = geometry["floor_corners"]

# Create triangles for floor (2 triangles make a rectangle)
floor_vertices = floor_corners
floor_triangles = np.array([[0, 1, 2], [0, 2, 3]])  # First triangle  # Second triangle

floor_mesh = o3d.geometry.TriangleMesh()
floor_mesh.vertices = o3d.utility.Vector3dVector(floor_vertices)
floor_mesh.triangles = o3d.utility.Vector3iVector(floor_triangles)
floor_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red
floor_mesh.compute_vertex_normals()

# Make floor semi-transparent
geometries_to_visualize.append(floor_mesh)

# Floor boundary as thick line
floor_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
floor_line_set = o3d.geometry.LineSet()
floor_line_set.points = o3d.utility.Vector3dVector(floor_corners)
floor_line_set.lines = o3d.utility.Vector2iVector(floor_lines)
floor_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in floor_lines])
geometries_to_visualize.append(floor_line_set)

# Floor corner spheres
for i, corner in enumerate(floor_corners):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    sphere.translate(corner)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.compute_vertex_normals()
    geometries_to_visualize.append(sphere)

print(f"  Floor mesh created (red)")
print()

# ============================================================================
# WALLS - as meshes (rectangles)
# ============================================================================

print("Creating wall meshes...")
for i, wall in enumerate(geometry["walls"]):
    wall_corners = wall["corners"]

    # Create triangles for wall (2 triangles make a rectangle)
    wall_vertices = wall_corners
    wall_triangles = np.array(
        [[0, 1, 2], [0, 2, 3]]  # First triangle  # Second triangle
    )

    wall_mesh = o3d.geometry.TriangleMesh()
    wall_mesh.vertices = o3d.utility.Vector3dVector(wall_vertices)
    wall_mesh.triangles = o3d.utility.Vector3iVector(wall_triangles)
    wall_mesh.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
    wall_mesh.compute_vertex_normals()
    geometries_to_visualize.append(wall_mesh)

    # Wall edges as lines
    wall_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    wall_line_set = o3d.geometry.LineSet()
    wall_line_set.points = o3d.utility.Vector3dVector(wall_corners)
    wall_line_set.lines = o3d.utility.Vector2iVector(wall_lines)
    wall_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in wall_lines])
    geometries_to_visualize.append(wall_line_set)

    # Wall center marker
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    center_sphere.translate(wall["center"])
    center_sphere.paint_uniform_color([0.0, 0.0, 1.0])
    center_sphere.compute_vertex_normals()
    geometries_to_visualize.append(center_sphere)

    # Wall normal vector (arrow showing direction)
    normal_start = wall["center"]
    normal_end = wall["center"] + wall["normal"] * 0.5

    # Create arrow as cylinder + cone
    arrow_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=0.4)
    arrow_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.15)

    # Position and orient arrow
    arrow_direction = wall["normal"] / np.linalg.norm(wall["normal"])
    arrow_cylinder.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
    arrow_cone.paint_uniform_color([0.0, 1.0, 1.0])

    # Rotate to align with normal
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, arrow_direction)
    rotation_angle = np.arccos(np.dot(z_axis, arrow_direction))

    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle
        )
        arrow_cylinder.rotate(R, center=[0, 0, 0])
        arrow_cone.rotate(R, center=[0, 0, 0])

    arrow_cylinder.translate(normal_start + arrow_direction * 0.2)
    arrow_cone.translate(normal_start + arrow_direction * 0.45)

    geometries_to_visualize.append(arrow_cylinder)
    geometries_to_visualize.append(arrow_cone)

print(f"  {len(geometry['walls'])} wall meshes created (blue)")
print()

# ============================================================================
# WINDOWS - as meshes
# ============================================================================

print("Creating window meshes...")
for i, window in enumerate(geometry["windows"]):
    window_corners = window["corners"]

    # Create window mesh
    window_vertices = window_corners
    window_triangles = np.array([[0, 1, 2], [0, 2, 3]])

    window_mesh = o3d.geometry.TriangleMesh()
    window_mesh.vertices = o3d.utility.Vector3dVector(window_vertices)
    window_mesh.triangles = o3d.utility.Vector3iVector(window_triangles)
    window_mesh.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    window_mesh.compute_vertex_normals()
    geometries_to_visualize.append(window_mesh)

    # Window edges
    window_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    window_line_set = o3d.geometry.LineSet()
    window_line_set.points = o3d.utility.Vector3dVector(window_corners)
    window_line_set.lines = o3d.utility.Vector2iVector(window_lines)
    window_line_set.colors = o3d.utility.Vector3dVector(
        [[1, 1, 0] for _ in window_lines]
    )
    geometries_to_visualize.append(window_line_set)

    # Window center marker
    window_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    window_sphere.translate(window["center"])
    window_sphere.paint_uniform_color([1.0, 1.0, 0.0])
    window_sphere.compute_vertex_normals()
    geometries_to_visualize.append(window_sphere)

print(f"  {len(geometry['windows'])} window meshes created (yellow)")
print()

# ============================================================================
# OPENINGS - as meshes
# ============================================================================

print("Creating opening meshes...")
for i, opening in enumerate(geometry["openings"]):
    opening_corners = opening["corners"]

    # Create opening mesh
    opening_vertices = opening_corners
    opening_triangles = np.array([[0, 1, 2], [0, 2, 3]])

    opening_mesh = o3d.geometry.TriangleMesh()
    opening_mesh.vertices = o3d.utility.Vector3dVector(opening_vertices)
    opening_mesh.triangles = o3d.utility.Vector3iVector(opening_triangles)
    opening_mesh.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    opening_mesh.compute_vertex_normals()
    geometries_to_visualize.append(opening_mesh)

    # Opening edges
    opening_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    opening_line_set = o3d.geometry.LineSet()
    opening_line_set.points = o3d.utility.Vector3dVector(opening_corners)
    opening_line_set.lines = o3d.utility.Vector2iVector(opening_lines)
    opening_line_set.colors = o3d.utility.Vector3dVector(
        [[0, 1, 0] for _ in opening_lines]
    )
    geometries_to_visualize.append(opening_line_set)

    # Opening center marker
    opening_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    opening_sphere.translate(opening["center"])
    opening_sphere.paint_uniform_color([0.0, 1.0, 0.0])
    opening_sphere.compute_vertex_normals()
    geometries_to_visualize.append(opening_sphere)

print(f"  {len(geometry['openings'])} opening meshes created (green)")
print()

# ============================================================================
# COORDINATE FRAMES
# ============================================================================

print("Creating coordinate frames...")

# World origin (0, 0, 0)
coord_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)
geometries_to_visualize.append(coord_origin)

# Floor center
floor_center = floor_corners.mean(axis=0)
coord_floor = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.4, origin=floor_center
)
geometries_to_visualize.append(coord_floor)

print(f"  Coordinate frames created")
print()

# ============================================================================
# STEP 3: Visualize
# ============================================================================

print("=" * 80)
print("OPENING 3D VIEWER")
print("=" * 80)
print()
print("Geometry legend:")
print("  🔴 Red mesh & lines:    Floor")
print("  🔵 Blue meshes & lines:  Walls")
print("  🟡 Yellow mesh:          Window")
print("  🟢 Green mesh:           Opening/Door")
print("  🔵 Blue spheres:         Wall centers")
print("  🔵 Cyan arrows:          Wall normals (facing direction)")
print("  📐 RGB axes at origin:  World origin (scan start)")
print("  📐 RGB axes at center:  Floor center")
print()
print("Controls:")
print("  • Left mouse drag:      Rotate view")
print("  • Right mouse drag:     Pan view")
print("  • Scroll wheel:         Zoom in/out")
print("  • Press 'H':            Show help menu")
print("  • Press 'R':            Reset view")
print()
print("Opening viewer...")
print()

# Visualize with custom settings
o3d.visualization.draw_geometries(
    geometries_to_visualize,
    window_name="Complete Room: Floor + 4 Walls + Windows + Openings (3D)",
    width=1920,
    height=1080,
    left=50,
    top=50,
    point_show_normal=False,
    mesh_show_wireframe=False,
    mesh_show_back_face=True,
)

print()
print("=" * 80)
print("3D VISUALIZATION CLOSED")
print("=" * 80)
print()
print("Summary:")
print(f"  • Point cloud: {len(points_zup)} points")
print(f"  • Floor: 1 mesh (4 corners)")
print(f"  • Walls: {len(geometry['walls'])} meshes")
print(f"  • Windows: {len(geometry['windows'])} meshes")
print(f"  • Openings: {len(geometry['openings'])} meshes")
print()
print("All geometry is aligned in Z-up coordinate system!")
print("=" * 80)
print("=" * 80)
