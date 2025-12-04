import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN


def segment_floor_ceiling_walls(
    points,
    ceiling_indices=None,
    floor_method="histogram_ransac",
    bottom_slice_height=0.30,
    floor_ransac_threshold=0.02,
):
    """
    Complete segmentation: ceiling, floor, and walls

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Full point cloud
    ceiling_indices : np.ndarray, optional
        Indices of ceiling points (from previous detection)
        If None, will detect ceiling first
    floor_method : str
        'histogram_ransac', 'histogram' or 'ransac'
    bottom_slice_height : float
        Height of bottom slice for histogram+RANSAC (meters)
    floor_ransac_threshold : float
        RANSAC distance threshold for floor detection

    Returns:
    --------
    segmentation : dict
        {
            'ceiling_indices': np.ndarray,
            'floor_indices': np.ndarray,
            'wall_indices': np.ndarray,
            'ceiling_points': np.ndarray,
            'floor_points': np.ndarray,
            'wall_points': np.ndarray,
            'floor_height': float,
            'ceiling_height': float (if available)
        }
    """

    # print(f"\n{'='*70}")
    # print(f"COMPLETE FLOOR/CEILING/WALL SEGMENTATION")
    # print(f"{'='*70}")
    # print(f"Total points: {len(points):,}")

    # Step 1: Get ceiling indices (if not provided)
    # if ceiling_indices is None:
    #     print(f"\nStep 1: Detecting ceiling...")
    #     # Use your existing ceiling detection
    #     # from ceiling_detection import detect_and_refine_ceiling
    #     # ceiling_indices = detect_and_refine_ceiling(points)
    # else:
    #     print(f"\nStep 1: Using provided ceiling indices")
    #     print(f"  Ceiling points: {len(ceiling_indices):,}")

    # Step 2: Remove ceiling, detect floor
    # print(f"\nStep 2: Detecting floor...")

    # Create mask for non-ceiling points
    non_ceiling_mask = np.ones(len(points), dtype=bool)
    non_ceiling_mask[ceiling_indices] = False
    non_ceiling_points = points[non_ceiling_mask]
    non_ceiling_indices = np.where(non_ceiling_mask)[0]

    # print(f"  Remaining after ceiling removal: {len(non_ceiling_points):,}")

    # Detect floor
    if floor_method == "histogram_ransac":
        floor_indices_local, floor_height = detect_floor_histogram_ransac(
            non_ceiling_points,
            bottom_slice_height=bottom_slice_height,
            distance_threshold=floor_ransac_threshold,
        )
    elif floor_method == "histogram":
        floor_indices_local = detect_floor_histogram(non_ceiling_points)
        floor_height = (
            non_ceiling_points[floor_indices_local, 2].mean()
            if len(floor_indices_local) > 0
            else None
        )
    else:  # ransac
        floor_indices_local = detect_floor_ransac(
            non_ceiling_points, floor_ransac_threshold
        )
        floor_height = (
            non_ceiling_points[floor_indices_local, 2].mean()
            if len(floor_indices_local) > 0
            else None
        )

    # Map back to original indices
    floor_indices = non_ceiling_indices[floor_indices_local]

    # print(f"  Floor points detected: {len(floor_indices):,}")

    # Step 3: Remaining points = walls
    # print(f"\nStep 3: Extracting walls...")

    all_structural = np.zeros(len(points), dtype=bool)
    all_structural[ceiling_indices] = True
    all_structural[floor_indices] = True

    wall_mask = ~all_structural
    wall_indices = np.where(wall_mask)[0]

    # print(f"  Wall points: {len(wall_indices):,}")

    # Get ceiling height
    ceiling_height = (
        points[ceiling_indices, 2].mean() if len(ceiling_indices) > 0 else None
    )

    # Summary
    # print(f"\n{'='*70}")
    # print(f"SEGMENTATION SUMMARY")
    # print(f"{'='*70}")
    # print(f"Total points:   {len(points):,}")
    # print(
    #     f"Ceiling:        {len(ceiling_indices):,} ({100*len(ceiling_indices)/len(points):.1f}%)"
    # )
    # print(
    #     f"Floor:          {len(floor_indices):,} ({100*len(floor_indices)/len(points):.1f}%)"
    # )
    # print(
    #     f"Walls:          {len(wall_indices):,} ({100*len(wall_indices)/len(points):.1f}%)"
    # )
    if floor_height and ceiling_height:
        room_height = ceiling_height - floor_height
        print(
            f"\nRoom height:    {room_height:.3f}m (floor: {floor_height:.3f}m, ceiling: {ceiling_height:.3f}m)"
        )
    # print(f"{'='*70}")

    return {
        "ceiling_indices": ceiling_indices,
        "floor_indices": floor_indices,
        "wall_indices": wall_indices,
        "ceiling_points": points[ceiling_indices],
        "floor_points": points[floor_indices],
        "wall_points": points[wall_indices],
        "floor_height": floor_height,
        "ceiling_height": ceiling_height,
    }


def detect_floor_histogram_ransac(
    points,
    bottom_slice_height=0.30,
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000,
):
    """
    Detect floor using HISTOGRAM + RANSAC (same strategy as ceiling)

    Step 1: Find lowest height cluster in histogram
    Step 2: Extract bottom slice around that cluster
    Step 3: Run RANSAC only on bottom slice (NOT all points!)

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Points (with ceiling already removed)
    bottom_slice_height : float
        Height of bottom slice to analyze (meters)
        e.g., 0.30m = analyze bottom 30cm around floor peak
    distance_threshold : float
        RANSAC inlier threshold (meters)

    Returns:
    --------
    floor_indices : np.ndarray
        Indices of floor points (in input array)
    floor_height : float
        Floor height (Z coordinate)
    """

    # print(f"  Floor detection method: Histogram + RANSAC")
    # print(f"    Bottom slice height: {bottom_slice_height}m")
    # print(f"    RANSAC threshold: {distance_threshold}m")

    z_coords = points[:, 2]

    # print(f"\n  Z range: [{z_coords.min():.3f}m, {z_coords.max():.3f}m]")

    # Step 1: Height histogram to find floor peak
    num_bins = 100
    hist, bin_edges = np.histogram(z_coords, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Focus on BOTTOM 30% of height range
    z_range = z_coords.max() - z_coords.min()
    bottom_threshold = z_coords.min() + z_range * 0.3

    bottom_mask = bin_centers <= bottom_threshold
    bottom_hist = hist.copy()
    bottom_hist[~bottom_mask] = 0  # Zero out upper bins

    # Find peaks
    peaks, properties = find_peaks(
        bottom_hist,
        prominence=len(points) * 0.01,  # At least 1% of points
        distance=5,  # At least 5 bins apart
    )

    if len(peaks) == 0:
        print(f"    ❌ ERROR: No peaks found in bottom histogram")
        print(f"       Bottom threshold: {bottom_threshold:.3f}m")
        return np.array([], dtype=int), None

    # Get LOWEST peak (floor is at bottom!)
    peak_heights = bin_centers[peaks]
    floor_peak_idx = peaks[np.argmin(peak_heights)]
    floor_height_estimate = bin_centers[floor_peak_idx]

    # print(f"\n  Histogram analysis:")
    # print(f"    Found {len(peaks)} peaks in bottom 30%")
    # print(f"    Peak heights: {', '.join([f'{h:.3f}m' for h in sorted(peak_heights)])}")
    # print(
    #     f"    Lowest peak (floor): {floor_height_estimate:.3f}m ({hist[floor_peak_idx]:,} points)"
    # )

    # Step 2: Extract bottom slice around floor peak
    slice_min = floor_height_estimate - bottom_slice_height / 2
    slice_max = floor_height_estimate + bottom_slice_height / 2

    slice_mask = (z_coords >= slice_min) & (z_coords <= slice_max)
    slice_points = points[slice_mask]
    slice_indices = np.where(slice_mask)[0]

    # print(f"\n  Bottom slice [{slice_min:.3f}m, {slice_max:.3f}m]:")
    # print(
    #     f"    Points in slice: {len(slice_points):,} ({100*len(slice_points)/len(points):.1f}% of remaining)"
    # )

    if len(slice_points) < 100:
        print(f"    ❌ ERROR: Too few points in bottom slice")
        return np.array([], dtype=int), None

    # Step 3: RANSAC ONLY on bottom slice
    print(f"\n  Running RANSAC on bottom slice only...")

    pcd_slice = o3d.geometry.PointCloud()
    pcd_slice.points = o3d.utility.Vector3dVector(slice_points)

    plane_model, inliers_local = pcd_slice.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    [a, b, c, d] = plane_model

    # print(f"\n  RANSAC results:")
    # print(f"    Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    # print(f"    Normal: [{a:.3f}, {b:.3f}, {c:.3f}]")
    # print(f"    |Normal Z|: {abs(c):.3f}")
    # print(
    #     f"    Inliers: {len(inliers_local):,} / {len(slice_points):,} ({100*len(inliers_local)/len(slice_points):.1f}% of slice)"
    # )

    # Check if horizontal
    if abs(c) < 0.9:
        print(f"    ⚠️  WARNING: Plane not horizontal! |nz|={abs(c):.3f}")
        print(f"       This might not be a valid floor")
    else:
        print(f"    ✓ Plane is horizontal")

    # Map back to original indices
    floor_indices = slice_indices[inliers_local]

    # Actual floor height (from plane equation: z = -d/c for horizontal plane)
    floor_height = -d / c if abs(c) > 0.01 else floor_height_estimate

    distances = np.abs(
        a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    ) / np.sqrt(a**2 + b**2 + c**2)

    # Expand to all points within thickness
    final_thickness = 0.10  # 10cm tolerance
    expanded_mask = distances <= final_thickness
    expanded_indices = np.where(expanded_mask)[0]

    print(f"    RANSAC inliers in slice: {len(floor_indices):,}")
    print(
        f"    Expanded to all points (±{final_thickness}m): {len(expanded_indices):,}"
    )
    print(
        f"    Additional floor points found: {len(expanded_indices) - len(floor_indices):,}"
    )

    # Return expanded indices instead
    return expanded_indices, floor_height
    # print(f"\n  ✓ Floor detection complete:")
    # print(f"    Floor height: {floor_height:.3f}m")
    # print(f"    Floor points: {len(floor_indices):,}")

    # return floor_indices, floor_height


def detect_floor_histogram(
    points, bottom_slice_thickness=0.30, height_tolerance=0.05, min_points=1000
):
    """
    Detect floor using histogram only (no RANSAC)

    Faster but less accurate than histogram + RANSAC

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Points (with ceiling already removed)
    bottom_slice_thickness : float
        Thickness of bottom slice to analyze (meters)
    height_tolerance : float
        Clustering tolerance for height grouping (meters)
    min_points : int
        Minimum points to consider valid floor

    Returns:
    --------
    floor_indices : np.ndarray
        Indices of floor points (in input array)
    """

    print(f"  Floor detection method: Histogram only")

    z_coords = points[:, 2]
    min_z = z_coords.min()
    max_z_for_floor = min_z + bottom_slice_thickness

    print(f"    Min height: {min_z:.2f}m")
    print(f"    Floor slice: {min_z:.2f}m to {max_z_for_floor:.2f}m")

    # Extract bottom slice
    floor_mask = z_coords <= max_z_for_floor
    floor_slice = points[floor_mask]
    floor_slice_indices = np.where(floor_mask)[0]

    print(f"    Points in bottom slice: {len(floor_slice):,}")

    if len(floor_slice) < min_points:
        print(f"    ⚠️  Too few points in floor slice!")
        return np.array([], dtype=int)

    # Cluster by height using DBSCAN

    z_values = floor_slice[:, 2].reshape(-1, 1)
    clustering = DBSCAN(eps=height_tolerance, min_samples=50)
    labels = clustering.fit_predict(z_values)

    # Find largest cluster (main floor)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise

    if len(unique_labels) == 0:
        print(f"    ⚠️  No floor clusters found!")
        return np.array([], dtype=int)

    # Get largest cluster
    cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
    largest_cluster = max(cluster_sizes, key=lambda x: x[1])
    floor_cluster_label = largest_cluster[0]
    floor_cluster_size = largest_cluster[1]

    print(f"    Floor cluster: {floor_cluster_size:,} points")

    # Get indices of floor points
    floor_cluster_mask = labels == floor_cluster_label
    floor_indices = floor_slice_indices[floor_cluster_mask]

    return floor_indices


def detect_floor_ransac(
    points, distance_threshold=0.02, ransac_n=3, num_iterations=1000
):
    """
    Detect floor using RANSAC only (on ALL points)

    ⚠️  WARNING: This can detect walls if they're the largest plane!
    Use histogram_ransac instead for robustness.

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Points (with ceiling already removed)
    distance_threshold : float
        RANSAC inlier threshold (meters)

    Returns:
    --------
    floor_indices : np.ndarray
        Indices of floor points (in input array)
    """

    print(f"  Floor detection method: RANSAC (all points)")
    print(f"    ⚠️  WARNING: May detect walls!")
    print(f"    Distance threshold: {distance_threshold}m")

    # Convert to Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # RANSAC plane fitting
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    [a, b, c, d] = plane_model

    print(f"    Floor plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(
        f"    Inliers: {len(inliers):,} / {len(points):,} ({100*len(inliers)/len(points):.1f}%)"
    )

    # Check if plane is roughly horizontal (floor should be)
    normal_z = abs(c)
    if normal_z < 0.9:
        print(f"    ⚠️  WARNING: Plane not horizontal! Normal Z: {normal_z:.3f}")
        print(f"       Likely detected a WALL instead of floor!")

    floor_indices = np.array(inliers)

    return floor_indices


def visualize_segmentation(points, segmentation, subsample=None):
    """
    Visualize ceiling/floor/wall segmentation

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Full point cloud
    segmentation : dict
        Output from segment_floor_ceiling_walls
    subsample : int, optional
        Subsample points for faster visualization
    """

    fig = plt.figure(figsize=(18, 6))

    # Subsample if needed
    if subsample and len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points_vis = points[indices]

        # Create mapping
        ceiling_vis = np.isin(indices, segmentation["ceiling_indices"])
        floor_vis = np.isin(indices, segmentation["floor_indices"])
        wall_vis = np.isin(indices, segmentation["wall_indices"])
    else:
        points_vis = points
        ceiling_vis = np.isin(np.arange(len(points)), segmentation["ceiling_indices"])
        floor_vis = np.isin(np.arange(len(points)), segmentation["floor_indices"])
        wall_vis = np.isin(np.arange(len(points)), segmentation["wall_indices"])

    # Plot 1: All together
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(
        points_vis[ceiling_vis, 0],
        points_vis[ceiling_vis, 1],
        points_vis[ceiling_vis, 2],
        c="red",
        s=1,
        alpha=0.5,
        label="Ceiling",
    )
    ax1.scatter(
        points_vis[floor_vis, 0],
        points_vis[floor_vis, 1],
        points_vis[floor_vis, 2],
        c="green",
        s=1,
        alpha=0.5,
        label="Floor",
    )
    ax1.scatter(
        points_vis[wall_vis, 0],
        points_vis[wall_vis, 1],
        points_vis[wall_vis, 2],
        c="blue",
        s=1,
        alpha=0.5,
        label="Walls",
    )
    ax1.set_title("Complete Segmentation", fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend()

    # Plot 2: Walls only (side view)
    ax2 = fig.add_subplot(132, projection="3d")
    wall_points = segmentation["wall_points"]
    if subsample and len(wall_points) > subsample // 2:
        wall_sample = wall_points[
            np.random.choice(len(wall_points), subsample // 2, replace=False)
        ]
    else:
        wall_sample = wall_points
    ax2.scatter(
        wall_sample[:, 0],
        wall_sample[:, 1],
        wall_sample[:, 2],
        c="blue",
        s=2,
        alpha=0.6,
    )
    ax2.set_title(f"Walls Only ({len(wall_points):,} points)", fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")

    # Plot 3: Top-down view (walls)
    ax3 = fig.add_subplot(133)
    ax3.scatter(wall_points[:, 0], wall_points[:, 1], c="blue", s=1, alpha=0.3)
    ax3.set_title("Walls (Top View)", fontweight="bold")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
