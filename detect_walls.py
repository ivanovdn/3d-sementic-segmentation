import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def extract_wall_points(
    points, floor_height, ceiling_height, floor_margin=0.1, ceiling_margin=0.1
):
    """
    Keep only points between floor and ceiling (potential walls)

    Args:
        points: Nx3 array
        floor_height: float
        ceiling_height: float
        floor_margin: Distance above floor to include (m)
        ceiling_margin: Distance below ceiling to include (m)

    Returns:
        wall_points: Filtered points
        wall_mask: Boolean mask
    """
    z_values = points[:, 2]

    wall_mask = (z_values > floor_height + floor_margin) & (
        z_values < ceiling_height - ceiling_margin
    )

    wall_points = points[wall_mask]

    # print(f"Original points: {len(points)}")
    # print(f"Wall points: {len(wall_points)} ({100*len(wall_points)/len(points):.1f}%)")

    return wall_points, wall_mask


def compute_normals(points, radius=0.1, max_nn=30):
    """
    Compute normal vectors for each point

    Args:
        points: Nx3 array
        radius: Search radius for normal estimation
        max_nn: Max number of neighbors

    Returns:
        normals: Nx3 array of normal vectors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=15)

    normals = np.asarray(pcd.normals)

    # print(f"Computed normals for {len(normals)} points")

    return normals, pcd


def filter_vertical_points(points, normals, vertical_threshold=0.1):
    """
    Keep only points with vertical normals (walls)

    Args:
        points: Nx3 array
        normals: Nx3 array
        vertical_threshold: Max abs(normal_z) for vertical surfaces

    Returns:
        vertical_points: Filtered points
        vertical_normals: Corresponding normals
        vertical_mask: Boolean mask
    """
    # Vertical surfaces have normals with small Z component
    normal_z = np.abs(normals[:, 2])
    vertical_mask = normal_z < vertical_threshold

    vertical_points = points[vertical_mask]
    vertical_normals = normals[vertical_mask]

    # print(f"Wall points: {len(points)}")
    # print(
    #     f"Vertical wall points: {len(vertical_points)} "
    #     f"({100*len(vertical_points)/len(points):.1f}%)"
    # )

    return vertical_points, vertical_normals, vertical_mask


def extract_walls_complete_pipeline(
    points,
    floor_height,
    ceiling_height,
    floor_margin=0.10,
    ceiling_margin=0.10,
    normal_radius=0.10,
    max_nn=30,
    vertical_threshold=0.15,
):
    """
    Complete 3-step wall extraction pipeline

    Step 1: Height-based filtering (remove floor/ceiling + margins)
    Step 2: Normal computation
    Step 3: Vertical surface filtering

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Full point cloud
    floor_height : float
        Floor Z coordinate
    ceiling_height : float
        Ceiling Z coordinate
    floor_margin : float
        Distance above floor to exclude (m)
    ceiling_margin : float
        Distance below ceiling to exclude (m)
    normal_radius : float
        Radius for normal estimation (m)
    max_nn : int
        Max neighbors for normal estimation
    vertical_threshold : float
        Max |normal_z| for vertical surfaces

    Returns:
    --------
    result : dict
        {
            'wall_points': np.ndarray (M, 3) - final wall points,
            'wall_normals': np.ndarray (M, 3) - wall normals,
            'wall_indices': np.ndarray (M,) - original indices,
            'step1_points': np.ndarray - after height filtering,
            'step2_normals': np.ndarray - all computed normals,
            'pcd': o3d.PointCloud - with normals
        }
    """

    # print(f"\n{'='*70}")
    # print(f"COMPLETE WALL EXTRACTION PIPELINE")
    # print(f"{'='*70}")
    # print(f"Input points: {len(points):,}")
    # print(f"Floor height: {floor_height:.3f}m")
    # print(f"Ceiling height: {ceiling_height:.3f}m")
    # print(f"Safety margins: floor={floor_margin}m, ceiling={ceiling_margin}m")

    # STEP 1: Height-based filtering
    # print(f"\n{'─'*70}")
    # print(f"STEP 1: Height-based filtering")
    # print(f"{'─'*70}")

    wall_points_step1, wall_mask_step1 = extract_wall_points(
        points,
        floor_height,
        ceiling_height,
        floor_margin=floor_margin,
        ceiling_margin=ceiling_margin,
    )

    step1_indices = np.where(wall_mask_step1)[0]

    print(
        f"  Excluded: {len(points) - len(wall_points_step1):,} points "
        f"({100*(len(points) - len(wall_points_step1))/len(points):.1f}%)"
    )

    # STEP 2: Normal computation
    # print(f"\n{'─'*70}")
    # print(f"STEP 2: Normal computation")
    # print(f"{'─'*70}")
    # print(f"  Normal radius: {normal_radius}m")
    # print(f"  Max neighbors: {max_nn}")

    normals_step2, pcd = compute_normals(
        wall_points_step1, radius=normal_radius, max_nn=max_nn
    )

    # STEP 3: Vertical surface filtering
    # print(f"\n{'─'*70}")
    # print(f"STEP 3: Vertical surface filtering")
    # print(f"{'─'*70}")
    # print(f"  Vertical threshold: |normal_z| < {vertical_threshold}")

    vertical_points, vertical_normals, vertical_mask = filter_vertical_points(
        wall_points_step1, normals_step2, vertical_threshold=vertical_threshold
    )

    # Map back to original indices
    vertical_indices = step1_indices[vertical_mask]

    # print(
    #     f"  Non-vertical removed: {len(wall_points_step1) - len(vertical_points):,} points "
    #     f"({100*(len(wall_points_step1) - len(vertical_points))/len(wall_points_step1):.1f}%)"
    # )

    # Summary
    # print(f"\n{'='*70}")
    # print(f"PIPELINE SUMMARY")
    # print(f"{'='*70}")
    # print(f"Input points:                 {len(points):,}")
    # print(
    #     f"After Step 1 (height filter): {len(wall_points_step1):,} "
    #     f"({100*len(wall_points_step1)/len(points):.1f}%)"
    # )
    # print(f"After Step 2 (normals):       {len(wall_points_step1):,} (same)")
    # print(
    #     f"After Step 3 (vertical):      {len(vertical_points):,} "
    #     f"({100*len(vertical_points)/len(points):.1f}%)"
    # )
    # print(f"\n✓ Final wall points: {len(vertical_points):,}")
    # print(
    #     f"  Total reduction: {len(points) - len(vertical_points):,} "
    #     f"({100*(len(points) - len(vertical_points))/len(points):.1f}%)"
    # )
    # print(f"{'='*70}")

    return {
        "wall_points": vertical_points,
        "wall_normals": vertical_normals,
        "wall_indices": vertical_indices,
        "step1_points": wall_points_step1,
        "step2_normals": normals_step2,
        "pcd": pcd,
    }


def visualize_wall_pipeline(points, floor_height, ceiling_height, result):
    """
    Visualize the 3-step wall extraction process
    """

    fig = plt.figure(figsize=(20, 10))

    # Subsample for visualization
    subsample = 30000

    # Plot 1: Original point cloud
    ax1 = fig.add_subplot(231, projection="3d")
    if len(points) > subsample:
        idx = np.random.choice(len(points), subsample, replace=False)
        points_vis = points[idx]
    else:
        points_vis = points
    ax1.scatter(
        points_vis[:, 0],
        points_vis[:, 1],
        points_vis[:, 2],
        c=points_vis[:, 2],
        cmap="viridis",
        s=1,
        alpha=0.5,
    )
    ax1.set_title(f"Step 0: Original\n{len(points):,} points", fontweight="bold")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Plot 2: After height filtering
    ax2 = fig.add_subplot(232, projection="3d")
    step1_pts = result["step1_points"]
    if len(step1_pts) > subsample:
        idx = np.random.choice(len(step1_pts), subsample, replace=False)
        step1_vis = step1_pts[idx]
    else:
        step1_vis = step1_pts
    ax2.scatter(
        step1_vis[:, 0], step1_vis[:, 1], step1_vis[:, 2], c="orange", s=1, alpha=0.6
    )
    ax2.axhline(y=0, z=floor_height, color="green", linewidth=2, label="Floor")
    ax2.axhline(y=0, z=ceiling_height, color="red", linewidth=2, label="Ceiling")
    ax2.set_title(
        f"Step 1: Height Filter\n{len(step1_pts):,} points", fontweight="bold"
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Plot 3: After vertical filtering
    ax3 = fig.add_subplot(233, projection="3d")
    wall_pts = result["wall_points"]
    if len(wall_pts) > subsample:
        idx = np.random.choice(len(wall_pts), subsample, replace=False)
        wall_vis = wall_pts[idx]
    else:
        wall_vis = wall_pts
    ax3.scatter(
        wall_vis[:, 0], wall_vis[:, 1], wall_vis[:, 2], c="blue", s=1, alpha=0.6
    )
    ax3.set_title(f"Step 3: Vertical Only\n{len(wall_pts):,} points", fontweight="bold")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    # Plot 4: Top view - Step 1
    ax4 = fig.add_subplot(234)
    ax4.scatter(step1_pts[:, 0], step1_pts[:, 1], c="orange", s=0.5, alpha=0.3)
    ax4.set_title("Step 1: Top View", fontweight="bold")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Top view - Step 3 (walls)
    ax5 = fig.add_subplot(235)
    ax5.scatter(wall_pts[:, 0], wall_pts[:, 1], c="blue", s=0.5, alpha=0.5)
    ax5.set_title("Step 3: Walls Top View", fontweight="bold")
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    ax5.set_aspect("equal")
    ax5.grid(True, alpha=0.3)

    # Plot 6: Normal distribution
    ax6 = fig.add_subplot(236)
    normal_z = result["step2_normals"][:, 2]
    ax6.hist(np.abs(normal_z), bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    ax6.axvline(
        0.15, color="red", linestyle="--", linewidth=2, label="Vertical threshold"
    )
    ax6.set_xlabel("|Normal Z|", fontweight="bold")
    ax6.set_ylabel("Count", fontweight="bold")
    ax6.set_title("Normal Z Distribution", fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compute_wall_geometry_robust(wall_points, room_height):
    """
    Robust wall geometry computation with fallback methods

    Returns:
        dict with length, height, thickness, area
    """

    # Height is always reliable
    z_values = wall_points[:, 2]
    wall_height = z_values.max() - z_values.min()

    # 2D points
    points_2d = wall_points[:, :2]

    if len(points_2d) < 3:
        return None

    # Method 1: Try PCA (best when it works)
    try:
        from sklearn.decomposition import PCA

        # Check if points are not all identical
        if np.std(points_2d) < 1e-6:
            raise ValueError("Points are identical")

        pca = PCA(n_components=2)
        pca.fit(points_2d)

        # Check if PCA is valid
        if np.any(np.isnan(pca.components_)) or np.any(np.isinf(pca.components_)):
            raise ValueError("PCA produced invalid results")

        transformed = pca.transform(points_2d)

        # Check transformed values
        if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
            raise ValueError("Transform produced invalid results")

        length = transformed[:, 0].max() - transformed[:, 0].min()
        thickness = transformed[:, 1].max() - transformed[:, 1].min()

        wall_length = max(length, thickness)
        wall_thickness = min(length, thickness)

        # Sanity check
        if wall_length < 0.01 or wall_thickness < 0:
            raise ValueError("Invalid dimensions from PCA")

    except Exception as e:
        # Method 2: Fallback - use bounding box
        print(f"    PCA failed ({str(e)[:50]}), using bounding box")

        min_x, min_y = points_2d.min(axis=0)
        max_x, max_y = points_2d.max(axis=0)

        length_x = max_x - min_x
        length_y = max_y - min_y

        wall_length = max(length_x, length_y)
        wall_thickness = min(length_x, length_y)

    wall_area = wall_length * wall_height
    height_ratio = wall_height / room_height

    return {
        "length": wall_length,
        "height": wall_height,
        "thickness": wall_thickness,
        "area": wall_area,
        "height_ratio": height_ratio,
    }


def detect_walls_ransac_clustering(
    vertical_points,
    floor_height,
    ceiling_height,
    distance_threshold=0.03,
    cluster_eps=0.10,
    min_cluster_points=100,
    min_wall_height=1.5,
    min_wall_length=0.80,
    max_wall_thickness=0.25,
    min_height_ratio=0.50,
    max_iterations=30,
):
    """
    Detect walls using RANSAC + DBSCAN clustering (ROBUST VERSION)
    """

    # print(f"\n{'='*70}")
    # print(f"WALL DETECTION: RANSAC + DBSCAN (ROBUST)")
    # print(f"{'='*70}")
    # print(f"Total vertical points: {len(vertical_points):,}")

    room_height = ceiling_height - floor_height
    # print(f"Room height: {room_height:.2f}m")
    # print(f"Min wall length: {min_wall_length}m")
    # print(f"Min height ratio: {min_height_ratio*100:.0f}%")

    walls = []
    # remaining_points = vertical_points.copy()
    remaining_points = vertical_points.copy()
    remaining_idx = np.arange(len(vertical_points))

    for iteration in range(max_iterations):
        if len(remaining_points) < min_cluster_points:
            # print(f"\n✓ Stopping: Only {len(remaining_points)} points remaining")
            break

        # print(f"\n{'-'*70}")
        # print(f"Iteration {iteration + 1}/{max_iterations}")
        # print(f"Remaining: {len(remaining_points):,} points")

        # STEP 1: RANSAC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining_points)

        plane_model, inliers_idx = pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=5000
        )
        inlier_global_idx = remaining_idx[inliers_idx]
        inlier_global_idx = np.array(inlier_global_idx, dtype=int)

        if len(inliers_idx) < min_cluster_points:
            # print(f"  ⚠️  Plane too small: {len(inliers_idx)} points")
            break

        [a, b, c, d] = plane_model
        # print(f"  Plane: {len(inliers_idx)} inliers, n=[{a:.3f}, {b:.3f}, {c:.3f}]")

        # Check verticality
        if abs(c) > 0.3:
            # print(f"  ⚠️  Not vertical (|nz|={abs(c):.2f})")
            remaining_points = np.delete(remaining_points, inliers_idx, axis=0)
            remaining_idx = np.delete(remaining_idx, inliers_idx, axis=0)
            continue

        inlier_points = remaining_points[inliers_idx]

        # STEP 2: DBSCAN
        pcd_inliers = o3d.geometry.PointCloud()
        pcd_inliers.points = o3d.utility.Vector3dVector(inlier_points)

        labels = np.array(
            pcd_inliers.cluster_dbscan(eps=cluster_eps, min_points=min_cluster_points)
        )

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        if len(unique_labels) == 0:
            # print(f"  ⚠️  No valid clusters")
            remaining_points = np.delete(remaining_points, inliers_idx, axis=0)
            remaining_idx = np.delete(remaining_idx, inliers_idx, axis=0)
            continue

        # Find largest cluster
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        largest_idx = unique_labels[np.argmax(cluster_sizes)]
        largest_size = max(cluster_sizes)

        # print(f"  Clusters: {len(unique_labels)}, largest: {largest_size:,} pts")

        # Extract largest cluster
        wall_mask = labels == largest_idx
        wall_points = inlier_points[wall_mask]
        wall_indices = inlier_global_idx[wall_mask]

        # STEP 3: Compute geometry (ROBUST)
        geometry = compute_wall_geometry_robust(wall_points, room_height)

        if geometry is None:
            # print(f"  ⚠️  Geometry computation failed")
            remaining_points = np.delete(remaining_points, inliers_idx, axis=0)
            remaining_idx = np.delete(remaining_idx, inliers_idx, axis=0)
            continue

        wall_length = geometry["length"]
        wall_height = geometry["height"]
        wall_thickness = geometry["thickness"]
        wall_area = geometry["area"]
        height_ratio = geometry["height_ratio"]

        # STEP 4: Validate
        is_valid = (
            wall_height >= min_wall_height
            and wall_length >= min_wall_length
            and wall_thickness <= max_wall_thickness
            and height_ratio >= min_height_ratio  # NEW!
            and largest_size >= min_cluster_points
        )

        # print(
        #     f"  Geo: L={wall_length:.2f}m, H={wall_height:.2f}m ({height_ratio*100:.0f}%), "
        #     f"T={wall_thickness:.2f}m"
        # )

        if not is_valid:
            # print(f"  ❌ Rejected")
            remaining_points = np.delete(remaining_points, inliers_idx, axis=0)
            remaining_idx = np.delete(remaining_idx, inliers_idx, axis=0)
            continue

        # STEP 5: Store
        center = wall_points.mean(axis=0)
        normal = np.array([a, b, c])

        wall_info = {
            "wall_id": len(walls) + 1,
            "plane_model": plane_model,
            "normal": normal,
            "center": center,
            "wall_indices": wall_indices,
            "points": wall_points,
            "num_points": largest_size,
            "length": wall_length,
            "height": wall_height,
            "thickness": wall_thickness,
            "area": wall_area,
            "height_ratio": height_ratio,
        }

        walls.append(wall_info)

        # print(
        #     f"  ✅ Wall {len(walls)}: {largest_size:,} pts, "
        #     f"{wall_length:.2f}m × {wall_height:.2f}m"
        # )

        remaining_points = np.delete(remaining_points, inliers_idx, axis=0)
        remaining_idx = np.delete(remaining_idx, inliers_idx, axis=0)

    # print(f"\n{'='*70}")
    # print(f"COMPLETE: {len(walls)} walls detected")
    # print(f"{'='*70}")

    return walls


def ensemble_ransac_wall_detection(
    vertical_points, floor_height, ceiling_height, n_runs=5, config=None
):
    """
    Run RANSAC multiple times, keep points that appear in majority of runs

    Strategy: Consensus voting
    - Run RANSAC n_runs times
    - Each point gets a "vote" each time it's classified as wall
    - Keep points with votes >= threshold (e.g., 3 out of 5)
    """

    # print(f"\n{'='*70}")
    # print(f"ENSEMBLE RANSAC WALL DETECTION ({n_runs} runs)")
    # print(f"{'='*70}")

    # Track votes for each point
    point_votes = np.zeros(len(vertical_points), dtype=int)

    all_walls_list = []
    all_wall_indices_list = []

    # Run RANSAC multiple times
    for run_idx in tqdm(range(n_runs)):
        # print(f"\nRun {run_idx + 1}/{n_runs}...")

        # Detect walls (RANSAC is stochastic, gives different results)
        walls = detect_walls_ransac_clustering(
            vertical_points,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            **config,
        )

        # Get indices of all points classified as walls
        wall_indices = set()
        for wall in walls:
            # Assuming you have indices stored in wall dict
            if "wall_indices" in wall:
                wall_indices.update(wall["wall_indices"])
            else:
                # Find indices by matching points
                for pt in wall["points"]:
                    idx = np.where((vertical_points == pt).all(axis=1))[0]
                    if len(idx) > 0:
                        wall_indices.add(idx[0])

        # Vote for these points
        wall_indices = np.array(list(wall_indices))
        point_votes[wall_indices] += 1

        all_walls_list.append(walls)
        all_wall_indices_list.append(wall_indices)

        # print(f"  Found {len(walls)} walls, {len(wall_indices):,} wall points")

    # Determine consensus threshold
    consensus_threshold = (n_runs + 1) // 2  # Majority vote (e.g., 3 out of 5)

    # print(f"\n{'='*70}")
    # print(f"CONSENSUS RESULTS")
    # print(f"{'='*70}")
    # print(f"Consensus threshold: {consensus_threshold}/{n_runs} votes")

    # Get consensus wall points
    consensus_mask = point_votes >= consensus_threshold
    consensus_wall_points = vertical_points[consensus_mask]

    print(f"\nVoting distribution:")
    for votes in range(n_runs + 1):
        count = np.sum(point_votes == votes)
        if count > 0:
            print(
                f"  {votes} votes: {count:,} points ({100*count/len(vertical_points):.1f}%)"
            )

    # print(f"\nConsensus wall points: {len(consensus_wall_points):,}")

    return {
        "consensus_points": consensus_wall_points,
        "consensus_indices": np.where(consensus_mask)[0],
        "point_votes": point_votes,
        "all_runs": all_walls_list,
    }


def get_room_wall_points(walls):
    """
    Extract all wall points for a specific room

    Args:
        walls: Dict from detect_walls_per_room_all_clusters
               Format: {room_id: [wall1, wall2, ...]}
               Each wall has: {'inliers': Nx3 array, 'plane_model': [a,b,c,d], ...}
        room_id: Room ID (integer)

    Returns:
        points_3d: Nx3 numpy array of all wall points in this room
                   Returns empty array if room not found or has no walls
    """

    room_walls = walls.copy()

    # Collect all point clouds from all wall segments
    all_points = []

    for i, wall in enumerate(room_walls):
        wall_points = wall["points"]  # Nx3 array
        all_points.append(wall_points)

    # Stack all points into single array
    combined_points = np.vstack(all_points)

    return combined_points


def project_points_to_2d(points_3d):
    """
    Project 3D wall points to 2D (X-Y plane)

    Args:
        points_3d: Nx3 numpy array (X, Y, Z coordinates)

    Returns:
        points_2d: Nx2 numpy array (X, Y coordinates)
    """
    if len(points_3d) == 0:
        return np.array([])

    # Simply take X and Y columns (ignore Z)
    points_2d = points_3d[:, :2]

    # print(f"Projected {len(points_3d)} points to 2D")
    # print(f"2D bounds:")
    # print(
    #     f"  X: [{points_2d[:, 0].min():.3f}, {points_2d[:, 0].max():.3f}] "
    #     f"(range: {points_2d[:, 0].max() - points_2d[:, 0].min():.3f}m)"
    # )
    # print(
    #     f"  Y: [{points_2d[:, 1].min():.3f}, {points_2d[:, 1].max():.3f}] "
    #     f"(range: {points_2d[:, 1].max() - points_2d[:, 1].min():.3f}m)"
    # )

    return points_2d


def visualize_2d_points(points_2d, title="2D Points", downsample=1):
    """
    Visualize 2D points with matplotlib

    Args:
        points_2d: Nx2 numpy array
        title: Plot title
        downsample: Plot every Nth point (for speed with large datasets)
    """
    if len(points_2d) == 0:
        print("No points to visualize")
        return

    # Downsample for visualization
    if downsample > 1:
        points_to_plot = points_2d[::downsample]
        print(
            f"Plotting {len(points_to_plot)} points (downsampled from {len(points_2d)})"
        )
    else:
        points_to_plot = points_2d

    plt.figure(figsize=(10, 6))
    plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], s=1, c="blue", alpha=0.5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_wall_points_3d(wall_points, wall_normals):
    """Visualize extracted wall points"""
    pcd_walls = o3d.geometry.PointCloud()
    pcd_walls.points = o3d.utility.Vector3dVector(wall_points)
    pcd_walls.normals = o3d.utility.Vector3dVector(wall_normals)
    pcd_walls.paint_uniform_color([0, 1, 0])  # green
    o3d.visualization.draw_geometries([pcd_walls], window_name="Wall Points")
