import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

###### FIND PARALLEL WALLS


def find_parallel_wall_pairs(wall_planes, angle_threshold=5.0, distance_threshold=0.20):
    """
    Find walls that are parallel and close together
    """

    pairs = []

    for i in range(len(wall_planes)):
        for j in range(i + 1, len(wall_planes)):

            plane_i = wall_planes[i]
            plane_j = wall_planes[j]

            # Check 1: Are normals parallel?
            normal_i = plane_i["normal"]
            normal_j = plane_j["normal"]

            # Angle between normals
            cos_angle = np.dot(normal_i, normal_j)
            angle = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1)))

            if angle > angle_threshold:
                continue  # Not parallel

            # Check 2: Are they close together?
            #  centroid of each wall as representative point
            point_i = plane_i["points"].mean(axis=0)
            point_j = plane_j["points"].mean(axis=0)

            # Distance between parallel planes
            # Project vector between centroids onto normal
            distance = np.abs(np.dot(point_j - point_i, normal_i))

            if distance > distance_threshold:
                continue  # Too far apart

            # Found a pair!
            pairs.append((i, j))
            print(
                f"Found pair: Wall {i} ↔ Wall {j}, distance={distance*100:.1f}cm, angle={angle:.2f}°"
            )

    return pairs


#### APPROACHES TO REMOVE DOUBLE WALLS

### APPROACH 1 - CLOSER TO CENTER


def select_inner_walls(wall_planes, pairs, room_center):
    """
    From each pair, keep only the wall closer to room center
    """

    walls_to_remove = set()

    for i, j in pairs:
        plane_i = wall_planes[i]
        plane_j = wall_planes[j]

        # Calculate distance from each wall to room center
        centroid_i = plane_i["points"].mean(axis=0)
        centroid_j = plane_j["points"].mean(axis=0)

        dist_i = np.linalg.norm(centroid_i - room_center)
        dist_j = np.linalg.norm(centroid_j - room_center)

        if dist_i < dist_j:
            # Wall i is inner, remove wall j
            keep_idx = i
            remove_idx = j
        else:
            # Wall j is inner, remove wall i
            keep_idx = j
            remove_idx = i

        walls_to_remove.add(remove_idx)

        print(
            f"Pair {i}↔{j}: Keeping wall {keep_idx} (inner), removing wall {remove_idx} (outer)"
        )

    # Create filtered list
    merged_walls = [
        wall for idx, wall in enumerate(wall_planes) if idx not in walls_to_remove
    ]

    print(f"\nWalls: {len(wall_planes)} → {len(merged_walls)}")

    return merged_walls, walls_to_remove


### APPROACH 2 - MERGE WALLS TO MIDDLE PLANE


def merge_to_middle_plane(walls, pairs):
    """
    Project paired walls to middle plane

    Result: all points on middle plane
    """

    print(f"\n{'='*70}")
    print(f"APPROACH B: MERGE TO MIDDLE PLANE")
    print(f"{'='*70}")

    walls_to_remove = set()
    merged_walls_list = []

    for i, j in pairs:
        wall_i = walls[i]
        wall_j = walls[j]

        walls_to_remove.add(i)
        walls_to_remove.add(j)

        # Get data
        points_i = wall_i["points"]
        points_j = wall_j["points"]
        normal_i = wall_i["normal"]
        normal_j = wall_j["normal"]

        print(
            f"\nMerging Wall {i} ({len(points_i):,} pts) + Wall {j} ({len(points_j):,} pts)"
        )

        # Calculate middle plane
        centroid_i = points_i.mean(axis=0)
        centroid_j = points_j.mean(axis=0)
        middle_point = (centroid_i + centroid_j) / 2

        middle_normal = (normal_i + normal_j) / 2
        middle_normal = middle_normal / np.linalg.norm(middle_normal)

        print(
            f"  Middle point: [{middle_point[0]:.3f}, {middle_point[1]:.3f}, {middle_point[2]:.3f}]"
        )

        # Combine all points
        all_points = np.vstack([points_i, points_j])

        # Project ALL points onto middle plane
        vectors = all_points - middle_point
        distances = np.dot(vectors, middle_normal)[:, np.newaxis]
        projected_points = all_points - distances * middle_normal

        print(f"  Projected {len(all_points):,} points to middle plane")
        print(
            f"  Distance range before: [{distances.min():.3f}, {distances.max():.3f}]"
        )
        print(f"  Distance after: 0 (all on plane)")

        merged_wall = {
            "points": projected_points,
            "normal": middle_normal,
            "merged_from": [i, j],
            "plane_point": middle_point,
            "approach": "B_middle_plane",
        }

        merged_walls_list.append(merged_wall)

    # Build new wall list
    new_walls = []
    for idx in range(len(walls)):
        if idx not in walls_to_remove:
            new_walls.append(walls[idx])
        elif idx == min([i for i, j in pairs if idx in (i, j)]):
            merged_wall = next(m for m in merged_walls_list if idx in m["merged_from"])
            new_walls.append(merged_wall)

    print(f"\n{'='*70}")
    print(f"RESULT: {len(walls)} → {len(new_walls)} walls")
    print(f"Gap eliminated by projection to middle plane")
    print(f"{'='*70}")

    return new_walls


def extract_inner_surface_perpendicular(
    wall_points, plane_normal, plane_point, room_center, depth_threshold=0.05
):  # 5cm from inner surface
    """
    Extract inner surface - CORRECTED VERSION

    Strategy:
    1. Determine inner direction (toward room center)
    2. Keep only points on inner side of fitted plane
    3. Keep only points CLOSE to that side (within depth_threshold)

    This keeps points as a FLAT surface, not a curved blob!

    Parameters:
    -----------
    depth_threshold : float
        How far from inner surface to include points (meters)
        e.g., 0.05 = include points within 5cm of inner surface
    """

    print(f"\n{'='*70}")
    print(f"EXTRACTING INNER SURFACE (CORRECTED)")
    print(f"{'='*70}")
    print(f"Input points: {len(wall_points):,}")
    print(f"Depth threshold: {depth_threshold*100:.1f}cm")

    # ========================================================================
    # Step 1: Determine inner direction
    # ========================================================================

    plane_to_center = room_center - plane_point
    dot_product = np.dot(plane_to_center, plane_normal)

    if dot_product > 0:
        inner_normal = plane_normal.copy()
        flip = False
    else:
        inner_normal = -plane_normal
        flip = True

    print(f"\nStep 1: Inner direction")
    print(f"  Normal flipped: {flip}")

    # ========================================================================
    # Step 2: Calculate perpendicular signed distances
    # ========================================================================

    vectors = wall_points - plane_point
    signed_distances = np.dot(vectors, inner_normal)

    thickness = signed_distances.max() - signed_distances.min()
    print(f"\nStep 2: Perpendicular distances")
    print(
        f"  Range: [{signed_distances.min()*100:.1f}cm, {signed_distances.max()*100:.1f}cm]"
    )
    print(f"  Thickness: {thickness*100:.1f}cm")

    # ========================================================================
    # Step 3: Find inner surface position
    # ========================================================================

    # The innermost points are at the MAXIMUM perpendicular distance
    # (on the side toward room center)

    inner_surface_distance = np.percentile(
        signed_distances, 95
    )  # 95th percentile = near inner edge

    print(f"\nStep 3: Inner surface location")
    print(f"  Inner surface at: {inner_surface_distance*100:.1f}cm from fitted plane")

    # ========================================================================
    # Step 4: Keep points near inner surface
    # ========================================================================

    # Keep points within depth_threshold of inner surface
    lower_bound = inner_surface_distance - depth_threshold
    upper_bound = inner_surface_distance

    mask = (signed_distances >= lower_bound) & (signed_distances <= upper_bound)
    inner_points = wall_points[mask]

    print(f"\nStep 4: Filter to inner surface")
    print(f"  Keep points in range: [{lower_bound*100:.1f}cm, {upper_bound*100:.1f}cm]")
    print(f"  Points kept: {mask.sum():,} ({mask.sum()/len(wall_points)*100:.1f}%)")

    # ========================================================================
    # Statistics
    # ========================================================================

    statistics = {
        "thickness": thickness,
        "inner_surface_distance": inner_surface_distance,
        "depth_threshold": depth_threshold,
        "kept_count": len(inner_points),
        "filter_ratio": len(inner_points) / len(wall_points),
    }

    print(f"\n{'='*70}")
    print(f"RESULT: {len(wall_points):,} → {len(inner_points):,} points")
    print(f"{'='*70}\n")

    return inner_points, statistics


def extract_inner_boundary_only(
    wall_points, plane_normal, plane_point, room_center, boundary_thickness=0.02
):
    """
    Extract ONLY the inner boundary - a thin layer

    This gives you the actual inner surface, not a thick slab

    Parameters:
    -----------
    boundary_thickness : float
        Thickness of inner surface layer to extract (default 1cm = 0.01m)
    """

    print(f"\n{'='*70}")
    print(f"EXTRACTING INNER BOUNDARY (THIN LAYER)")
    print(f"{'='*70}")
    print(f"Input points: {len(wall_points):,}")
    print(f"Boundary thickness: {boundary_thickness*100:.1f}cm")

    # ========================================================================
    # Step 1: Determine inner direction
    # ========================================================================

    plane_to_center = room_center - plane_point
    if np.dot(plane_to_center, plane_normal) < 0:
        inner_normal = -plane_normal
        flip = True
    else:
        inner_normal = plane_normal.copy()
        flip = False

    print(f"  Normal flipped: {flip}")

    # ========================================================================
    # Step 2: Calculate perpendicular distances
    # ========================================================================

    vectors = wall_points - plane_point
    signed_distances = np.dot(vectors, inner_normal)

    thickness = signed_distances.max() - signed_distances.min()
    print(f"  Wall thickness: {thickness*100:.1f}cm")
    print(
        f"  Distance range: [{signed_distances.min()*100:.1f}cm, {signed_distances.max()*100:.1f}cm]"
    )

    # ========================================================================
    # Step 3: Find innermost surface location
    # ========================================================================

    # The innermost points are at MAXIMUM perpendicular distance
    max_distance = signed_distances.max()

    print(f"  Innermost surface at: {max_distance*100:.1f}cm from fitted plane")

    # ========================================================================
    # Step 4: Extract thin layer at innermost position
    # ========================================================================

    # Keep only points within boundary_thickness of the innermost position
    lower_bound = max_distance - boundary_thickness
    upper_bound = max_distance

    mask = (signed_distances >= lower_bound) & (signed_distances <= upper_bound)
    inner_boundary_points = wall_points[mask]

    print(f"  Keep points in range: [{lower_bound*100:.1f}cm, {upper_bound*100:.1f}cm]")
    print(f"  Points kept: {mask.sum():,} ({mask.sum()/len(wall_points)*100:.1f}%)")

    # ========================================================================
    # Verify we have enough points
    # ========================================================================

    if len(inner_boundary_points) < 100:
        print(f"\n⚠️  WARNING: Very few points! Increasing threshold...")
        # Fallback: use 2cm if 1cm gives too few points
        boundary_thickness = 0.02
        lower_bound = max_distance - boundary_thickness
        mask = (signed_distances >= lower_bound) & (signed_distances <= upper_bound)
        inner_boundary_points = wall_points[mask]
        print(f"  Using {boundary_thickness*100:.1f}cm threshold instead")
        print(f"  Points kept: {len(inner_boundary_points):,}")

    statistics = {
        "thickness": thickness,
        "max_distance": max_distance,
        "boundary_thickness": boundary_thickness,
        "kept_count": len(inner_boundary_points),
        "filter_ratio": len(inner_boundary_points) / len(wall_points),
    }

    print(f"\n{'='*70}")
    print(f"RESULT: {len(wall_points):,} → {len(inner_boundary_points):,} points")
    print(f"Reduction: {(1 - statistics['filter_ratio'])*100:.1f}%")
    print(f"{'='*70}\n")

    return inner_boundary_points, statistics


def visualize_each_wall_independetly_3d(walls):
    wall_segments = {}
    for i, wall in enumerate(walls):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(wall["points"])
        colors = plt.get_cmap("tab20")(i)
        pcd.paint_uniform_color(list(colors[:3]))
        wall_segments[i] = pcd
    o3d.visualization.draw_geometries(
        [wall_segment for wall_segment in wall_segments.values()]
    )


def distribute_small_segments_proportional_local(dimensions, min_wall_length=0.25):
    """
    Distribute small segments to adjacent neighbors, but proportionally

    This is a middle ground:
    - Only affects immediate neighbors (preserves shape)
    - Distribution is proportional to neighbor lengths (fairer than 50/50)

    Parameters:
    -----------
    dimensions : list of dict
        Wall dimensions
    min_wall_length : float
        Segments shorter than this are distributed

    Returns:
    --------
    new_dimensions : list of dict
        Adjusted dimensions
    """

    import numpy as np

    print(f"\n{'='*70}")
    print(f"PROPORTIONAL LOCAL DISTRIBUTION")
    print(f"{'='*70}")
    print(f"Input walls: {len(dimensions)}")
    print(f"Min wall length: {min_wall_length}m")

    # Step 1: Identify small segments
    small_indices = []
    for i, wall in enumerate(dimensions):
        if wall["length"] < min_wall_length:
            small_indices.append(i)

    if len(small_indices) == 0:
        print(f"No small segments found")
        return dimensions

    print(f"\nFound {len(small_indices)} small segment(s):")
    for idx in small_indices:
        wall = dimensions[idx]
        prev_idx = (idx - 1) % len(dimensions)
        next_idx = (idx + 1) % len(dimensions)

        prev_wall = dimensions[prev_idx]
        next_wall = dimensions[next_idx]

        # Calculate proportional split
        total_neighbor_length = prev_wall["length"] + next_wall["length"]
        ratio_to_prev = (
            prev_wall["length"] / total_neighbor_length
            if total_neighbor_length > 0
            else 0.5
        )
        ratio_to_next = (
            next_wall["length"] / total_neighbor_length
            if total_neighbor_length > 0
            else 0.5
        )

        print(f"\n  Wall {wall['wall_id']}: {wall['length']:.3f}m")
        print(f"    Prev wall {prev_wall['wall_id']}: {prev_wall['length']:.3f}m")
        print(f"    Next wall {next_wall['wall_id']}: {next_wall['length']:.3f}m")
        print(f"    Split ratio: {ratio_to_prev:.1%} / {ratio_to_next:.1%}")

    # Step 2: Calculate new corner positions
    new_corners = {}

    for idx in small_indices:
        wall = dimensions[idx]
        prev_idx = (idx - 1) % len(dimensions)
        next_idx = (idx + 1) % len(dimensions)

        prev_wall = dimensions[prev_idx]
        next_wall = dimensions[next_idx]

        # Proportional split
        total_neighbor_length = prev_wall["length"] + next_wall["length"]
        ratio_to_prev = (
            prev_wall["length"] / total_neighbor_length
            if total_neighbor_length > 0
            else 0.5
        )

        # New corner position (interpolate along small segment)
        new_corner = wall["start_point"] + ratio_to_prev * (
            wall["end_point"] - wall["start_point"]
        )
        new_corners[idx] = new_corner

        amount_to_prev = wall["length"] * ratio_to_prev
        amount_to_next = wall["length"] * ratio_to_next

        print(f"\n  Distributing Wall {wall['wall_id']}:")
        print(f"    To prev wall {prev_wall['wall_id']}: {amount_to_prev:.3f}m")
        print(f"    To next wall {next_wall['wall_id']}: {amount_to_next:.3f}m")

    # Step 3: Build new dimensions
    new_dimensions = []

    for i, wall in enumerate(dimensions):
        if i in small_indices:
            continue

        new_wall = wall.copy()

        # Check if previous wall was small
        prev_idx = (i - 1) % len(dimensions)
        if prev_idx in small_indices:
            new_wall["start_point"] = new_corners[prev_idx]

        # Check if next wall is small
        next_idx = (i + 1) % len(dimensions)
        if next_idx in small_indices:
            new_wall["end_point"] = new_corners[next_idx]

        # Recalculate
        vec = new_wall["end_point"] - new_wall["start_point"]
        old_length = new_wall["length"]
        new_wall["length"] = np.linalg.norm(vec)
        new_wall["angle"] = np.degrees(np.arctan2(vec[1], vec[0])) % 360

        if abs(new_wall["length"] - old_length) > 0.001:
            change = new_wall["length"] - old_length
            print(
                f"    Wall {wall['wall_id']}: {old_length:.3f}m → {new_wall['length']:.3f}m "
                f"({change:+.3f}m)"
            )

        new_dimensions.append(new_wall)

    # Renumber
    for idx, wall in enumerate(new_dimensions, 1):
        wall["wall_id"] = idx
        wall["corner_start"] = idx
        wall["corner_end"] = (idx % len(new_dimensions)) + 1

    print(f"\n✓ Result: {len(dimensions)} → {len(new_dimensions)} walls")

    return new_dimensions
