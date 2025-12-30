"""
Improved Floor/Ceiling Detection

Key improvements:
1. Use percentile of actual points for floor height (not RANSAC plane equation)
2. Use percentile for ceiling height (not mean)
3. Better handling of noise

"""

import numpy as np
import open3d as o3d
from scipy.signal import find_peaks


def detect_ceiling_with_ransac(
    points,
    top_percentile=95,
    thickness=0.15,
    ransac_threshold=0.02,
    ransac_n=3,
    num_iterations=1000,
    height_percentile=95.0,  # Use 95th percentile (highest points)
):
    """
    Detect ceiling using HEIGHT filtering + RANSAC.
    
    Key: Use the HIGHEST percentile of ceiling points, not the plane center.
    
    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Point cloud
    top_percentile : float
        Look at top X% of points by height
    thickness : float
        Thickness for initial ceiling extraction (meters)
    ransac_threshold : float
        RANSAC distance threshold
    height_percentile : float
        Percentile for ceiling height (95 = use 95th percentile, i.e., highest points)
        
    Returns:
    --------
    ceiling_indices : np.ndarray
        Indices of ceiling points
    ceiling_height : float
        Ceiling height (Z coordinate from highest percentile)
    """

    z_coords = points[:, 2]

    print(f"\n{'='*70}")
    print(f"CEILING DETECTION (HEIGHT + RANSAC)")
    print(f"{'='*70}")
    print(f"Z range: [{z_coords.min():.3f}m, {z_coords.max():.3f}m]")

    # Step 1: Get top percentile
    z_threshold = np.percentile(z_coords, top_percentile)
    print(f"\nTop {100-top_percentile}% height threshold: {z_threshold:.3f}m")

    # Step 2: Extract top points
    top_mask = z_coords >= z_threshold
    top_points = points[top_mask]
    top_indices = np.where(top_mask)[0]

    print(f"Points above threshold: {len(top_points):,}")

    if len(top_points) < 100:
        print(f"❌ Too few top points!")
        return np.array([], dtype=int), None

    # Step 3: Find peak in top region
    top_z = top_points[:, 2]
    hist, bin_edges = np.histogram(top_z, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(hist, prominence=len(top_points) * 0.05)

    if len(peaks) == 0:
        ceiling_height_estimate = np.median(top_z)
        print(f"No clear peak, using median: {ceiling_height_estimate:.3f}m")
    else:
        peak_heights = bin_centers[peaks]
        ceiling_height_estimate = peak_heights.max()
        print(f"Found {len(peaks)} peaks in top region")
        print(f"Highest peak at: {ceiling_height_estimate:.3f}m")

    # Step 4: Extract slice around ceiling peak
    slice_min = ceiling_height_estimate - thickness
    slice_max = ceiling_height_estimate + thickness
    
    slice_mask = (z_coords >= slice_min) & (z_coords <= slice_max)
    slice_points = points[slice_mask]
    slice_indices = np.where(slice_mask)[0]
    
    print(f"\nCeiling slice [{slice_min:.3f}m, {slice_max:.3f}m]:")
    print(f"  Points in slice: {len(slice_points):,}")

    if len(slice_points) < 100:
        print(f"❌ Too few points in ceiling slice")
        return np.array([], dtype=int), None

    # Step 5: RANSAC on ceiling slice
    print(f"\nRunning RANSAC on ceiling slice...")
    
    pcd_slice = o3d.geometry.PointCloud()
    pcd_slice.points = o3d.utility.Vector3dVector(slice_points)

    plane_model, inliers_local = pcd_slice.segment_plane(
        distance_threshold=ransac_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    [a, b, c, d] = plane_model

    print(f"  Plane normal: [{a:.3f}, {b:.3f}, {c:.3f}]")
    print(f"  |Normal Z|: {abs(c):.3f}")

    if abs(c) < 0.9:
        print(f"  ⚠️  WARNING: Plane not horizontal!")
    else:
        print(f"  ✓ Plane is horizontal")

    print(f"  RANSAC inliers: {len(inliers_local):,} / {len(slice_points):,}")

    # Step 6: Expand to all points within thickness of plane
    distances = np.abs(
        a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    ) / np.sqrt(a**2 + b**2 + c**2)

    final_thickness = 0.10  # 10cm tolerance
    expanded_mask = distances <= final_thickness
    ceiling_indices = np.where(expanded_mask)[0]

    print(f"  Expanded to ±{final_thickness}m: {len(ceiling_indices):,} points")

    # Step 7: Calculate ceiling height
    # Use 95th percentile - balances accuracy vs outlier rejection
    ceiling_points_z = points[ceiling_indices, 2]
    
    ceiling_height_from_plane = -d / c if abs(c) > 0.01 else ceiling_height_estimate
    ceiling_height_percentile = np.percentile(ceiling_points_z, height_percentile)
    ceiling_height_max = ceiling_points_z.max()
    
    print(f"\nCeiling height calculation:")
    print(f"  From RANSAC plane equation: {ceiling_height_from_plane:.3f}m")
    print(f"  From {height_percentile}th percentile: {ceiling_height_percentile:.3f}m")
    print(f"  From max of ceiling points: {ceiling_height_max:.3f}m")
    print(f"  Points Z range: [{ceiling_points_z.min():.3f}m, {ceiling_points_z.max():.3f}m]")
    
    # Use percentile - best balance of accuracy and robustness
    ceiling_height = ceiling_height_percentile
    
    print(f"\n✓ Using ceiling height: {ceiling_height:.3f}m ({height_percentile}th percentile)")
    print(f"Ceiling points: {len(ceiling_indices):,} ({100*len(ceiling_indices)/len(points):.1f}%)")

    return ceiling_indices, ceiling_height


def detect_ceiling_height_based(
    points,
    top_percentile=95,
    thickness=0.10,
    height_percentile=50.0,  # Use median of ceiling points (more robust)
):
    """
    Detect ceiling by looking at HIGHEST points.
    
    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Point cloud
    top_percentile : float
        Look at top X% of points by height
    thickness : float
        Thickness for ceiling extraction (meters)
    height_percentile : float
        Percentile for ceiling height calculation (50 = median)
        
    Returns:
    --------
    ceiling_indices : np.ndarray
        Indices of ceiling points
    ceiling_height : float
        Ceiling height (Z coordinate)
    """

    z_coords = points[:, 2]

    print(f"\n{'='*70}")
    print(f"CEILING DETECTION (HEIGHT-BASED)")
    print(f"{'='*70}")
    print(f"Z range: [{z_coords.min():.3f}m, {z_coords.max():.3f}m]")

    # Step 1: Get top percentile
    z_threshold = np.percentile(z_coords, top_percentile)

    print(f"\nTop {100-top_percentile}% height threshold: {z_threshold:.3f}m")

    # Step 2: Extract top points
    top_mask = z_coords >= z_threshold
    top_points = points[top_mask]

    print(f"Points above threshold: {len(top_points):,}")

    if len(top_points) < 100:
        print(f"❌ Too few top points!")
        return np.array([], dtype=int), None

    # Step 3: Find densest cluster in top points using histogram
    top_z = top_points[:, 2]
    hist, bin_edges = np.histogram(top_z, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peak in top region
    peaks, _ = find_peaks(hist, prominence=len(top_points) * 0.05)

    if len(peaks) == 0:
        # No clear peak, use median of top points
        ceiling_height_estimate = np.median(top_z)
        print(f"No clear peak, using median: {ceiling_height_estimate:.3f}m")
    else:
        # Get HIGHEST peak (ceiling is at top!)
        peak_heights = bin_centers[peaks]
        ceiling_height_estimate = peak_heights.max()
        print(f"Found {len(peaks)} peaks in top region")
        print(f"Highest peak at: {ceiling_height_estimate:.3f}m")

    # Step 4: Extract ceiling points (within thickness of peak)
    ceiling_mask = np.abs(z_coords - ceiling_height_estimate) <= thickness
    ceiling_indices = np.where(ceiling_mask)[0]
    
    # Step 5: Calculate actual ceiling height from points
    # Use median (50th percentile) - most robust to outliers
    ceiling_points_z = z_coords[ceiling_indices]
    ceiling_height = np.percentile(ceiling_points_z, height_percentile)

    print(f"\nCeiling height (from {height_percentile}th percentile): {ceiling_height:.3f}m")
    print(f"  (min={ceiling_points_z.min():.3f}m, max={ceiling_points_z.max():.3f}m)")
    print(f"Ceiling points: {len(ceiling_indices):,} ({100*len(ceiling_indices)/len(points):.1f}%)")

    return ceiling_indices, ceiling_height


def detect_floor_histogram_ransac_improved(
    points,
    bottom_slice_height=0.30,
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000,
    height_percentile=5.0,  # Use 5th percentile (lowest points)
):
    """
    Detect floor using HISTOGRAM + RANSAC with improved height calculation.
    
    Key: Use the LOWEST percentile of floor points, not the plane center.
    This captures the true floor even if most points are slightly above it.

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Points (with ceiling already removed)
    bottom_slice_height : float
        Height of bottom slice to analyze (meters)
    distance_threshold : float
        RANSAC inlier threshold (meters)
    height_percentile : float
        Percentile for floor height (5 = use 5th percentile, i.e., lowest points)

    Returns:
    --------
    floor_indices : np.ndarray
        Indices of floor points (in input array)
    floor_height : float
        Floor height (Z coordinate)
    """

    print(f"\n{'='*70}")
    print(f"FLOOR DETECTION (HISTOGRAM + RANSAC)")
    print(f"{'='*70}")

    z_coords = points[:, 2]

    print(f"Z range: [{z_coords.min():.3f}m, {z_coords.max():.3f}m]")
    print(f"Bottom slice height: {bottom_slice_height}m")
    print(f"RANSAC threshold: {distance_threshold}m")

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
        print(f"❌ ERROR: No peaks found in bottom histogram")
        return np.array([], dtype=int), None

    # Get LOWEST peak (floor is at bottom!)
    peak_heights = bin_centers[peaks]
    floor_peak_idx = peaks[np.argmin(peak_heights)]
    floor_height_estimate = bin_centers[floor_peak_idx]

    print(f"\nHistogram analysis:")
    print(f"  Found {len(peaks)} peaks in bottom 30%")
    print(f"  Lowest peak (floor estimate): {floor_height_estimate:.3f}m")

    # Step 2: Extract bottom slice around floor peak
    slice_min = floor_height_estimate - bottom_slice_height / 2
    slice_max = floor_height_estimate + bottom_slice_height / 2

    slice_mask = (z_coords >= slice_min) & (z_coords <= slice_max)
    slice_points = points[slice_mask]
    slice_indices = np.where(slice_mask)[0]

    print(f"\nBottom slice [{slice_min:.3f}m, {slice_max:.3f}m]:")
    print(f"  Points in slice: {len(slice_points):,}")

    if len(slice_points) < 100:
        print(f"❌ ERROR: Too few points in bottom slice")
        return np.array([], dtype=int), None

    # Step 3: RANSAC ONLY on bottom slice
    print(f"\nRunning RANSAC on bottom slice...")

    pcd_slice = o3d.geometry.PointCloud()
    pcd_slice.points = o3d.utility.Vector3dVector(slice_points)

    plane_model, inliers_local = pcd_slice.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    [a, b, c, d] = plane_model

    print(f"  Plane normal: [{a:.3f}, {b:.3f}, {c:.3f}]")
    print(f"  |Normal Z|: {abs(c):.3f}")

    # Check if horizontal
    if abs(c) < 0.9:
        print(f"  ⚠️  WARNING: Plane not horizontal!")
    else:
        print(f"  ✓ Plane is horizontal")

    print(f"  RANSAC inliers: {len(inliers_local):,} / {len(slice_points):,}")

    # Expand to all points within thickness
    distances = np.abs(
        a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    ) / np.sqrt(a**2 + b**2 + c**2)

    final_thickness = 0.10  # 10cm tolerance
    expanded_mask = distances <= final_thickness
    floor_indices = np.where(expanded_mask)[0]

    print(f"  Expanded to ±{final_thickness}m: {len(floor_indices):,} points")

    # Step 4: Calculate floor height
    # Use 5th percentile - balances accuracy vs outlier rejection
    floor_points_z = points[floor_indices, 2]
    
    floor_height_from_plane = -d / c if abs(c) > 0.01 else floor_height_estimate
    floor_height_percentile = np.percentile(floor_points_z, height_percentile)
    floor_height_min = floor_points_z.min()
    
    print(f"\nFloor height calculation:")
    print(f"  From RANSAC plane equation: {floor_height_from_plane:.3f}m")
    print(f"  From {height_percentile}th percentile: {floor_height_percentile:.3f}m")
    print(f"  From min of floor points: {floor_height_min:.3f}m")
    print(f"  Points Z range: [{floor_points_z.min():.3f}m, {floor_points_z.max():.3f}m]")
    
    # Use percentile - best balance of accuracy and robustness
    floor_height = floor_height_percentile
    
    print(f"\n✓ Using floor height: {floor_height:.3f}m ({height_percentile}th percentile)")

    return floor_indices, floor_height


def segment_floor_ceiling_walls_improved(
    points,
    ceiling_indices=None,
    ceiling_height=None,
    floor_method="histogram_ransac",
    bottom_slice_height=0.30,
    floor_ransac_threshold=0.02,
    ceiling_ransac_threshold=0.02,
    use_ransac_ceiling=True,  # Use RANSAC for ceiling
):
    """
    Complete segmentation with RANSAC for both floor and ceiling.

    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Full point cloud
    ceiling_indices : np.ndarray, optional
        Indices of ceiling points (from previous detection)
    ceiling_height : float, optional
        Pre-calculated ceiling height
    floor_method : str
        'histogram_ransac'
    bottom_slice_height : float
        Height of bottom slice for histogram+RANSAC (meters)
    floor_ransac_threshold : float
        RANSAC distance threshold for floor detection
    ceiling_ransac_threshold : float
        RANSAC distance threshold for ceiling detection
    use_ransac_ceiling : bool
        If True, use RANSAC for ceiling height (recommended)

    Returns:
    --------
    segmentation : dict
    """

    print(f"\n{'='*70}")
    print(f"COMPLETE FLOOR/CEILING/WALL SEGMENTATION (RANSAC)")
    print(f"{'='*70}")
    print(f"Total points: {len(points):,}")
    print(f"Use RANSAC for ceiling: {use_ransac_ceiling}")

    # Step 1: Detect ceiling with RANSAC
    if ceiling_indices is None or ceiling_height is None:
        print(f"\nStep 1: Detecting ceiling...")
        if use_ransac_ceiling:
            ceiling_indices, ceiling_height = detect_ceiling_with_ransac(
                points,
                ransac_threshold=ceiling_ransac_threshold
            )
        else:
            ceiling_indices, ceiling_height = detect_ceiling_height_based(
                points
            )
    else:
        print(f"\nStep 1: Using provided ceiling (height={ceiling_height:.3f}m)")

    # Step 2: Remove ceiling, detect floor
    print(f"\nStep 2: Detecting floor...")

    # Create mask for non-ceiling points
    non_ceiling_mask = np.ones(len(points), dtype=bool)
    non_ceiling_mask[ceiling_indices] = False
    non_ceiling_points = points[non_ceiling_mask]
    non_ceiling_indices = np.where(non_ceiling_mask)[0]

    print(f"  Points after ceiling removal: {len(non_ceiling_points):,}")

    # Detect floor with RANSAC
    if floor_method == "histogram_ransac":
        floor_indices_local, floor_height = detect_floor_histogram_ransac_improved(
            non_ceiling_points,
            bottom_slice_height=bottom_slice_height,
            distance_threshold=floor_ransac_threshold,
        )
    else:
        raise ValueError(f"Unknown floor method: {floor_method}")

    # Map back to original indices
    floor_indices = non_ceiling_indices[floor_indices_local]

    # Step 3: Remaining points = walls
    print(f"\nStep 3: Extracting walls...")

    all_structural = np.zeros(len(points), dtype=bool)
    all_structural[ceiling_indices] = True
    all_structural[floor_indices] = True

    wall_mask = ~all_structural
    wall_indices = np.where(wall_mask)[0]

    print(f"  Wall points: {len(wall_indices):,}")

    # Summary
    room_height = ceiling_height - floor_height
    
    print(f"\n{'='*70}")
    print(f"SEGMENTATION SUMMARY")
    print(f"{'='*70}")
    print(f"Ceiling height: {ceiling_height:.3f}m (RANSAC plane)")
    print(f"Floor height:   {floor_height:.3f}m (RANSAC plane)")
    print(f"Room height:    {room_height:.3f}m")
    print(f"{'='*70}")

    return {
        "ceiling_indices": ceiling_indices,
        "floor_indices": floor_indices,
        "wall_indices": wall_indices,
        "ceiling_points": points[ceiling_indices],
        "floor_points": points[floor_indices],
        "wall_points": points[wall_indices],
        "floor_height": floor_height,
        "ceiling_height": ceiling_height,
        "room_height": room_height,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Improved Floor/Ceiling Detection")
    print("=" * 50)
    
    # Create synthetic room
    np.random.seed(42)
    
    # Room: 3.01m height, floor at -1.88m, ceiling at 1.13m
    actual_floor = -1.88
    actual_ceiling = 1.13
    actual_height = actual_ceiling - actual_floor  # 3.01m
    
    print(f"\nSynthetic room:")
    print(f"  Actual floor: {actual_floor}m")
    print(f"  Actual ceiling: {actual_ceiling}m")
    print(f"  Actual height: {actual_height}m")
    
    n_floor = 5000
    n_ceiling = 5000
    n_walls = 10000
    
    # Floor points (with some noise)
    floor_xy = np.random.uniform(-2, 2, (n_floor, 2))
    floor_z = actual_floor + np.random.normal(0, 0.02, n_floor)  # 2cm noise
    # Add some outliers below floor
    floor_z[:100] = actual_floor - np.random.uniform(0.05, 0.15, 100)
    floor_points = np.column_stack([floor_xy, floor_z])
    
    # Ceiling points (with some noise)
    ceiling_xy = np.random.uniform(-2, 2, (n_ceiling, 2))
    ceiling_z = actual_ceiling + np.random.normal(0, 0.02, n_ceiling)  # 2cm noise
    # Add some outliers above ceiling
    ceiling_z[:100] = actual_ceiling + np.random.uniform(0.05, 0.15, 100)
    ceiling_points = np.column_stack([ceiling_xy, ceiling_z])
    
    # Wall points
    wall_z = np.random.uniform(actual_floor + 0.1, actual_ceiling - 0.1, n_walls)
    wall_xy = np.random.uniform(-2, 2, (n_walls, 2))
    wall_points = np.column_stack([wall_xy, wall_z])
    
    # Combine
    points = np.vstack([floor_points, ceiling_points, wall_points])
    np.random.shuffle(points)
    
    print(f"\nTotal points: {len(points)}")
    
    # Run segmentation
    result = segment_floor_ceiling_walls_improved(
        points,
        use_ransac_ceiling=True,
    )
    
    print(f"\n✓ Results:")
    print(f"  Detected floor: {result['floor_height']:.3f}m (actual: {actual_floor}m, error: {abs(result['floor_height'] - actual_floor)*100:.1f}cm)")
    print(f"  Detected ceiling: {result['ceiling_height']:.3f}m (actual: {actual_ceiling}m, error: {abs(result['ceiling_height'] - actual_ceiling)*100:.1f}cm)")
    print(f"  Detected height: {result['room_height']:.3f}m (actual: {actual_height}m, error: {abs(result['room_height'] - actual_height)*100:.1f}cm)")
