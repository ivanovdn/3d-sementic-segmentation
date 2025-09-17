import numpy as np
import open3d as o3d


def floor_segmentation(points, room_height, coef=0.3, distance_threshold=0.05):
    """Segment floor using RANSAC"""

    z_min = np.min(points[:, 2])
    z_threshold = z_min + coef * room_height
    bottom_mask = points[:, 2] <= z_threshold

    # Create point cloud for bottom region
    bottom_pcd = o3d.geometry.PointCloud()
    bottom_pcd.points = o3d.utility.Vector3dVector(points[:, :3][bottom_mask])
    bottom_pcd.colors = o3d.utility.Vector3dVector(points[:, 3:][bottom_mask])

    # RANSAC for horizontal plane
    plane_model, inliers = bottom_pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000  # 5 cm
    )

    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    print(normal)

    if abs(normal[2]) > 0.8:  # Normal is mostly vertical (horizontal plane)
        # Convert back to original indices
        bottom_indices = np.where(bottom_mask)[0]
        floor_indices = bottom_indices[inliers]

        if len(floor_indices) > 0.05 * len(points):  # At least 5% of points
            return floor_indices, bottom_pcd.select_by_index(inliers)

    return None, None


def ceiling_segmentation(
    points, room_height, floor_indices, coef=0.3, distance_threshold=0.05
):
    """Segment ceil using RANSAC"""

    z_max = np.max(points[:, 2])
    z_threshold = z_max - coef * room_height
    top_mask = points[:, 2] >= z_threshold

    # Exclude floor points
    if floor_indices is not None:
        top_mask[floor_indices] = False

    if np.sum(top_mask) < 500:
        print("Not enough points in top region for ceiling detection")
        return None, None

    # Create point cloud for top region
    top_pcd = o3d.geometry.PointCloud()
    top_pcd.points = o3d.utility.Vector3dVector(points[:, :3][top_mask])
    top_pcd.colors = o3d.utility.Vector3dVector(points[:, 3:][top_mask])

    # RANSAC for horizontal plane
    plane_model, inliers = top_pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
    )

    # Check if plane is horizontal
    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    print(normal)

    if abs(normal[2]) > 0.8:  # Horizontal plane
        # Convert back to original indices
        top_indices = np.where(top_mask)[0]
        ceiling_indices = top_indices[inliers]

        # Check size
        if len(ceiling_indices) > 0.03 * len(points):  # At least 3% of points
            return ceiling_indices, top_pcd.select_by_index(inliers)

    return None, None


def segment_walls(
    points, floor_indices, ceiling_indices, room_height, distance_threshold=0.03
):
    """Segment walls using RANSAC"""

    # Focus on middle region (exclude floor and ceiling)
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    middle_mask = (points[:, 2] > z_min + 0.1 * room_height) & (
        points[:, 2] < z_max - 0.1 * room_height
    )

    # Exclude already segmented points
    if floor_indices is not None:
        middle_mask[floor_indices] = False
    if ceiling_indices is not None:
        middle_mask[ceiling_indices] = False

    if np.sum(middle_mask) < 1000:
        return []

    middle_points = points[:, :3][middle_mask]
    middle_colors = points[:, 3:][middle_mask]
    middle_indices = np.where(middle_mask)[0]

    wall_segments = []
    remaining_mask = np.ones(len(middle_points), dtype=bool)

    # Iteratively find wall planes
    for iteration in range(4):  # Find up to 4 walls
        if np.sum(remaining_mask) < 500:
            break

        # Create point cloud for remaining points
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(middle_points[remaining_mask])
        remaining_pcd.colors = o3d.utility.Vector3dVector(middle_colors[remaining_mask])

        # RANSAC for plane
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,  # to fine-tune
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inliers) < 300:  # Not enough points for a wall
            break

        # Check if plane is vertical (normal should be horizontal)
        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)

        if abs(normal[2]) < 0.2:  # Normal is mostly horizontal (vertical plane)
            # Convert back to original indices
            remaining_indices = np.where(remaining_mask)[0]
            wall_local_indices = remaining_indices[inliers]
            wall_global_indices = middle_indices[wall_local_indices]

            wall_pcd = remaining_pcd.select_by_index(inliers)

            wall_segments.append((wall_global_indices, wall_pcd))

            # Remove these points from remaining
            remaining_mask[wall_local_indices] = False

            print(f"Found wall {iteration+1}: {len(wall_global_indices)} points")
        else:
            break

    return wall_segments


def visualize_plane_segmentation(points, plane_labels):
    """Visualize plane segmentation results
    points: numpy array of shape (N, 6) containing [x, y, z, r, g, b]
    """

    # Create colored point cloud for planes
    plane_colors = np.array(
        [
            [0.8, 0.8, 0.8],  # ceiling - light gray
            [0.4, 0.2, 0.1],  # floor - brown
            [0.6, 0.6, 0.5],  # wall - beige
        ]
    )

    colored_points = np.zeros((len(points), 3))
    for i, label in enumerate(plane_labels):
        if 0 <= label <= 2:
            colored_points[i] = plane_colors[label]
        else:
            colored_points[i] = [1.0, 0.0, 0.0]  # red for remaining points

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colored_points)

    print(
        "Showing plane segmentation: Gray=ceiling, Brown=floor, Beige=walls, Red=remaining"
    )
    o3d.visualization.draw_geometries([pcd], window_name="Plane Segmentation Results")


def prepare_points_to_pointnet(pcd):
    """
    Returns:
        points: numpy array of shape (N, 6) containing [x, y, z, r, g, b]
    """

    points_xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Combine xyz and rgb
    points = np.concatenate([points_xyz, colors], axis=1)
    return points
