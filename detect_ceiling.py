import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.signal import find_peaks


def detect_ceiling_peaks(points, show_top_n=10):
    """
    Analyze height distribution and identify ceiling peaks.

    Returns peaks.
    """

    print(f"\n{'='*70}")
    print(f"DETECT CEILING PEAKS")
    print(f"{'='*70}")

    z_coords = points[:, 2]

    # Create histogram
    bin_size = 0.02
    z_min, z_max = z_coords.min(), z_coords.max()
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    hist, bin_edges = np.histogram(z_coords, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find all peaks
    peaks, properties = find_peaks(hist, prominence=0.02 * hist.max(), distance=5)

    # Sort by height (highest first)
    peak_heights = bin_centers[peaks]
    peak_counts = hist[peaks]
    sorted_indices = np.argsort(peak_heights)[::-1]

    print(f"Found {len(peaks)} peaks")
    print(f"\nTop {show_top_n} peaks by height:")
    print(f"{'ID':<5} {'Height (m)':<12} {'Points':<12} {'Type':<15}")
    print(f"{'─'*50}")

    for i, idx in enumerate(sorted_indices[:show_top_n]):
        h = peak_heights[idx]
        count = peak_counts[idx]

        # Classify peak
        if h > 0.5:
            peak_type = "CEILING ✓"
        elif h < -0.5:
            peak_type = "FLOOR"
        else:
            peak_type = "MID-HEIGHT"

        print(f"{i:<5} {h:<12.3f} {count:<12,} {peak_type:<15}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        bin_centers,
        hist,
        width=bin_size * 0.8,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
    )
    ax.plot(bin_centers[peaks], hist[peaks], "r*", markersize=12, label="All peaks")

    # Highlight top peaks
    for i, idx in enumerate(sorted_indices[:show_top_n]):
        h = peak_heights[idx]
        c = peak_counts[idx]
        ax.annotate(
            f"{i}: {h:.2f}m",
            xy=(h, c),
            xytext=(h, c + hist.max() * 0.05),
            ha="center",
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

    ax.set_xlabel("Height (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Point count", fontsize=12, fontweight="bold")
    ax.set_title(
        "Height Distribution - Select Ceiling Peak IDs", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Return data for manual selection
    result = {
        "peak_heights": peak_heights,
        "peak_counts": peak_counts,
        "sorted_indices": sorted_indices,
        "z_coords": z_coords,
        "points": points,
    }

    print(f"\n✓ Inspect the plot and choose ceiling peak IDs")
    print(f"  Example: If peaks 0 and 1 are ceilings, use: [0, 1]")

    return result


def extract_ceiling_levels(step1_result, selected_peak_ids, thickness=0.10):
    """
    Extract ceiling points for selected peak IDs

    Parameters:
    -----------
    step1_result : dict
        Output from step1_detect_ceiling_peaks
    selected_peak_ids : list of int
        Which peak IDs to use as ceilings (e.g., [0, 1])
    thickness : float
        Thickness around each peak (meters)

    Returns:
    --------
    ceiling_levels : list of dict
        Each dict contains:
        - 'height': float
        - 'points': np.ndarray (M, 3)
        - 'indices': np.ndarray (M,) - indices into ORIGINAL point cloud
        - 'level_id': int
    """

    print(f"\n{'='*70}")
    print(f"EXTRACT CEILING LEVELS")
    print(f"{'='*70}")

    peak_heights = step1_result["peak_heights"]
    sorted_indices = step1_result["sorted_indices"]
    z_coords = step1_result["z_coords"]
    points = step1_result["points"]

    ceiling_levels = []

    for peak_id in selected_peak_ids:
        if peak_id >= len(sorted_indices):
            print(f"⚠️  Peak ID {peak_id} out of range!")
            continue

        idx = sorted_indices[peak_id]
        height = peak_heights[idx]

        # Extract points at this height
        mask = np.abs(z_coords - height) < thickness
        level_points = points[mask]
        level_indices = np.where(mask)[0]  # track indices

        ceiling_levels.append(
            {
                "height": height,
                "points": level_points,
                "indices": level_indices,
                "level_id": peak_id,
            }
        )

        print(
            f"✓ Ceiling level {peak_id}: {height:.3f}m with {len(level_points):,} points"
        )

    # Visualize (unchanged)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(ceiling_levels)))

    # Top view
    for i, level in enumerate(ceiling_levels):
        pts = level["points"]
        ax1.scatter(
            pts[:, 0],
            pts[:, 1],
            c=[colors[i]],
            s=1,
            alpha=0.7,
            label=f"Level {level['level_id']} ({level['height']:.2f}m)",
        )

    ax1.set_aspect("equal")
    ax1.set_xlabel("X (m)", fontsize=12)
    ax1.set_ylabel("Y (m)", fontsize=12)
    ax1.set_title("Ceiling Levels (Top View)", fontsize=14, fontweight="bold")
    ax1.legend(markerscale=5)
    ax1.grid(True, alpha=0.3)

    # 3D view
    ax2 = plt.subplot(122, projection="3d")
    for i, level in enumerate(ceiling_levels):
        pts = level["points"]
        if len(pts) > 10000:
            vis_indices = np.random.choice(len(pts), 10000, replace=False)
            pts = pts[vis_indices]
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[colors[i]], s=1, alpha=0.6)

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title("Ceiling Levels (3D)", fontweight="bold")

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Extracted {len(ceiling_levels)} ceiling level(s)")

    return ceiling_levels


def ransac_refine_ceiling_level(
    ceiling_points,
    ceiling_indices,
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000,
):
    """
    RANSAC plane fitting with index tracking

    Parameters:
    -----------
    ceiling_points : np.ndarray (N, 3)
        Points from one ceiling level
    ceiling_indices : np.ndarray (N,)
        Indices of these points in ORIGINAL point cloud
    distance_threshold : float
        RANSAC inlier threshold (meters)
    ransac_n : int
        Minimum points to fit plane
    num_iterations : int
        RANSAC iterations

    Returns:
    --------
    inlier_points : np.ndarray (M, 3)
        Clean ceiling points (outliers removed)
    inlier_indices : np.ndarray (M,)
        Indices of inliers in ORIGINAL point cloud
    plane_model : tuple (a, b, c, d)
        Plane equation: ax + by + cz + d = 0
    inlier_ratio : float
        Percentage of points kept
    """

    print(f"\n{'='*70}")
    print(f"RANSAC PLANE REFINEMENT")
    print(f"{'='*70}")
    print(f"Input points: {len(ceiling_points):,}")
    print(f"Distance threshold: {distance_threshold}m")

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ceiling_points)

    # RANSAC plane fitting
    plane_model, inlier_mask = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    [a, b, c, d] = plane_model

    # Extract inliers AND their original indices
    inlier_points = ceiling_points[inlier_mask]
    inlier_indices = ceiling_indices[inlier_mask]  # ← Track through RANSAC!

    inlier_ratio = len(inlier_mask) / len(ceiling_points)

    print(f"\n✓ Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(
        f"✓ Inliers: {len(inlier_mask):,} / {len(ceiling_points):,} ({inlier_ratio*100:.1f}%)"
    )
    print(
        f"✓ Outliers removed: {len(ceiling_points) - len(inlier_mask):,} ({(1-inlier_ratio)*100:.1f}%)"
    )

    # Check if plane is roughly horizontal
    normal_z = abs(c)
    if normal_z < 0.9:
        print(f"⚠️  WARNING: Plane is not horizontal! Normal Z: {normal_z:.3f}")
        print(f"   Expected ~1.0 for ceiling. Check your data.")

    return inlier_points, inlier_indices, plane_model, inlier_ratio


def refine_all_ceiling_levels(ceiling_levels, distance_threshold=0.02):
    """
    Apply RANSAC to all ceiling levels with index tracking

    Parameters:
    -----------
    ceiling_levels : list of dict
        Output from step2_extract_ceiling_levels
    distance_threshold : float
        RANSAC threshold

    Returns:
    --------
    refined_levels : list of dict
        Each dict contains:
        - 'height': float
        - 'points': np.ndarray (M, 3) - clean points
        - 'indices': np.ndarray (M,) - original indices
        - 'level_id': int
        - 'plane_equation': tuple
        - 'inlier_ratio': float
        - 'original_count': int
        - 'cleaned_count': int
    """

    print(f"\n{'='*70}")
    print(f"REFINING ALL CEILING LEVELS")
    print(f"{'='*70}")

    refined_levels = []

    for i, level in enumerate(ceiling_levels):
        print(f"\n--- Level {i} (Height: {level['height']:.3f}m) ---")

        # RANSAC refinement with index tracking
        clean_points, clean_indices, plane, ratio = ransac_refine_ceiling_level(
            level["points"],
            level["indices"],  # ← Pass indices
            distance_threshold=distance_threshold,
        )

        # Store refined version
        refined_levels.append(
            {
                "height": level["height"],
                "points": clean_points,
                "indices": clean_indices,  # ← Preserve indices!
                "level_id": level["level_id"],
                "plane_equation": plane,
                "inlier_ratio": ratio,
                "original_count": len(level["points"]),
                "cleaned_count": len(clean_points),
            }
        )

    # Summary
    print(f"\n{'='*70}")
    print(f"REFINEMENT SUMMARY")
    print(f"{'='*70}")

    total_original = 0
    total_cleaned = 0

    for level in refined_levels:
        removed = level["original_count"] - level["cleaned_count"]
        print(f"Level {level['level_id']} @ {level['height']:.2f}m:")
        print(f"  Original: {level['original_count']:,} points")
        print(f"  Cleaned:  {level['cleaned_count']:,} points")
        print(f"  Removed:  {removed:,} ({removed/level['original_count']*100:.1f}%)")

        total_original += level["original_count"]
        total_cleaned += level["cleaned_count"]

    print(f"\nTotal across all levels:")
    print(f"  Original: {total_original:,} points")
    print(f"  Cleaned:  {total_cleaned:,} points")
    print(
        f"  Removed:  {total_original - total_cleaned:,} ({(total_original-total_cleaned)/total_original*100:.1f}%)"
    )

    return refined_levels
