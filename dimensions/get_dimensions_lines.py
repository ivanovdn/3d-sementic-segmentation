import matplotlib.pyplot as plt
import numpy as np


def extract_line_direction_hough(
    points_2d,
    resolution_rho=0.01,
    resolution_theta=np.pi / 180,
    threshold_ratio=0.1,
    use_opencv=True,
):
    """
    Extract line direction using Hough Transform (ROBUST VERSION)

    Parameters:
    -----------
    points_2d : np.ndarray (N, 2)
        2D points
    resolution_rho : float
        Distance resolution in meters (default 1cm)
    resolution_theta : float
        Angle resolution in radians (default 1 degree)
    threshold_ratio : float
        Threshold for line detection (fraction of points)
    use_opencv : bool
        Try OpenCV first, fall back to custom if fails

    Returns:
    --------
    direction : np.ndarray (2,)
        Unit vector along detected line
    """

    # Need at least some points
    if len(points_2d) < 10:
        print("  ⚠️  Too few points for Hough Transform, using PCA")
        return fallback_to_pca(points_2d)

    print(f"\n  Hough Transform:")
    print(f"    Points: {len(points_2d):,}")
    print(f"    Threshold ratio: {threshold_ratio:.1%}")

    # ========================================================================
    # Try OpenCV first (if available)
    # ========================================================================

    if use_opencv:
        try:
            import cv2

            direction = hough_opencv(
                points_2d, resolution_rho, resolution_theta, threshold_ratio
            )
            if direction is not None:
                return direction
            print("  ⚠️  OpenCV Hough failed, trying custom implementation")
        except ImportError:
            print("  ⚠️  OpenCV not available, using custom implementation")

    # ========================================================================
    # Use custom implementation
    # ========================================================================

    return hough_custom(points_2d)


def hough_opencv(points_2d, resolution_rho, resolution_theta, threshold_ratio):
    """
    Hough Transform using OpenCV
    """
    import cv2

    # Normalize points to image coordinates
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)

    # Calculate image size
    width = x_max - x_min
    height = y_max - y_min

    # Image dimensions (use smaller resolution for better performance)
    pixels_per_meter = 100  # 1cm per pixel
    image_width = int(width * pixels_per_meter) + 1
    image_height = int(height * pixels_per_meter) + 1

    # Limit size
    max_dim = 2000
    if max(image_width, image_height) > max_dim:
        scale = max_dim / max(image_width, image_height)
        image_width = int(image_width * scale)
        image_height = int(image_height * scale)

    print(f"    Image size: {image_width}x{image_height}")

    # Create image
    image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Map points to pixels
    x_pixels = ((points_2d[:, 0] - x_min) / width * (image_width - 1)).astype(int)
    y_pixels = ((points_2d[:, 1] - y_min) / height * (image_height - 1)).astype(int)

    x_pixels = np.clip(x_pixels, 0, image_width - 1)
    y_pixels = np.clip(y_pixels, 0, image_height - 1)

    # Set pixels
    image[y_pixels, x_pixels] = 255

    # Dilate to connect nearby points (IMPORTANT for sparse clouds!)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    print(f"    Non-zero pixels: {np.count_nonzero(image)}")

    # Calculate threshold
    threshold = int(threshold_ratio * len(points_2d))
    threshold = max(10, threshold)

    print(f"    Threshold: {threshold} votes")

    # Try with progressively lower thresholds
    for attempt, factor in enumerate([1.0, 0.5, 0.3, 0.1]):
        current_threshold = int(threshold * factor)

        lines = cv2.HoughLines(image, 1, np.pi / 180, current_threshold)

        if lines is not None and len(lines) > 0:
            print(f"    ✓ Found {len(lines)} lines (threshold={current_threshold})")

            # Take the strongest line
            rho, theta = lines[0][0]

            # Direction perpendicular to normal
            direction = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])
            direction = direction / np.linalg.norm(direction)

            angle = np.degrees(theta)
            print(f"    Angle: {angle:.1f}°, Rho: {rho:.2f}")

            return direction
        else:
            if attempt < 3:
                print(
                    f"    ⚠️  No lines at threshold={current_threshold}, trying lower..."
                )

    print(f"    ✗ No lines found even with low threshold")
    return None


def hough_custom(points_2d, num_theta=360, num_rho=500):
    """
    Custom Hough Transform (no OpenCV dependency)

    This is more forgiving than OpenCV implementation
    """

    print(f"    Using custom Hough Transform")
    print(f"    Theta bins: {num_theta}, Rho bins: {num_rho}")

    # Theta range: 0° to 180° (full range)
    thetas = np.linspace(0, np.pi, num_theta)

    # Calculate max possible rho
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)

    # Center points (Hough works better with centered data)
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    points_centered = points_2d - center

    max_rho = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    rhos = np.linspace(-max_rho, max_rho, num_rho)

    # Accumulator
    accumulator = np.zeros((num_rho, num_theta), dtype=int)

    # Precompute sin/cos
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Vote
    for point in points_centered:
        x, y = point

        # Calculate rho for each theta
        rho_values = x * cos_thetas + y * sin_thetas

        # Find closest rho bin
        rho_indices = np.searchsorted(rhos, rho_values) - 1
        rho_indices = np.clip(rho_indices, 0, num_rho - 1)

        # Vote
        accumulator[rho_indices, np.arange(num_theta)] += 1

    # Find peak
    max_votes = accumulator.max()
    peak_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    rho_idx, theta_idx = peak_idx

    rho = rhos[rho_idx]
    theta = thetas[theta_idx]

    print(f"    ✓ Peak found: {max_votes} votes")
    print(f"    Theta: {np.degrees(theta):.1f}°, Rho: {rho:.3f}m")

    # Direction perpendicular to normal [cos(theta), sin(theta)]
    direction = np.array([-np.sin(theta), np.cos(theta)])
    direction = direction / np.linalg.norm(direction)

    return direction


def fallback_to_pca(points_2d):
    """
    Fallback to PCA when Hough fails
    """
    centroid = points_2d.mean(axis=0)
    centered = points_2d - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal_idx = np.argmax(eigenvalues)
    direction = eigenvectors[:, principal_idx].real
    direction = direction / np.linalg.norm(direction)

    return direction


def extract_line_direction_hough_simple(points_2d):
    """
    Simplified Hough Transform that always works

    Uses angle histogram voting - very robust!
    """

    print(f"\n  Simple Hough (angle voting):")

    # Calculate pairwise angles between points
    num_samples = min(1000, len(points_2d))

    if len(points_2d) > num_samples:
        # Sample points for speed
        indices = np.random.choice(len(points_2d), num_samples, replace=False)
        sample_points = points_2d[indices]
    else:
        sample_points = points_2d

    # Vote for angles
    angles = []

    # Sample pairs of points
    for i in range(min(500, len(sample_points))):
        for j in range(i + 1, min(i + 20, len(sample_points))):
            p1 = sample_points[i]
            p2 = sample_points[j]

            # Angle of line through these points
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = np.arctan2(dy, dx)
                angles.append(angle)

    if len(angles) == 0:
        print(f"    ⚠️  No angles computed, using PCA")
        return fallback_to_pca(points_2d)

    angles = np.array(angles)

    # Normalize to [0, π] (lines are same in both directions)
    angles = angles % np.pi

    # Histogram
    num_bins = 180
    hist, bin_edges = np.histogram(angles, bins=num_bins, range=(0, np.pi))

    # Find peak
    peak_bin = np.argmax(hist)
    peak_angle = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2

    print(f"    ✓ Peak angle: {np.degrees(peak_angle):.1f}° ({hist[peak_bin]} votes)")

    # Direction vector
    direction = np.array([np.cos(peak_angle), np.sin(peak_angle)])

    return direction


def extract_wall_line_2d(wall_segment, method="plane_normal"):
    """
    Extract 2D line segment from wall point cloud

    Args:
        wall_segment: Dict with 'points' and 'plane_model'
        method:
            - 'plane_normal': Use plane normal (fastest)
            - 'pca': Use PCA (data-driven)
            - 'hough': Use Hough Transform (most robust, requires OpenCV)
            - 'hough_simple': Simple angle voting (always works, no OpenCV)
    """
    points_3d = wall_segment["points"]
    plane_model = wall_segment["plane_model"]
    a, b, c, d = plane_model

    points_2d = points_3d[:, :2]

    if method == "plane_normal":
        wall_direction = np.array([-b, a])
        norm = np.linalg.norm(wall_direction)
        if norm < 1e-6:
            wall_direction = np.array([1, 0])
        else:
            wall_direction = wall_direction / norm

    elif method == "pca":
        wall_direction = fallback_to_pca(points_2d)

    elif method == "hough":
        wall_direction = extract_line_direction_hough(points_2d)

    elif method == "hough_simple":
        wall_direction = extract_line_direction_hough_simple(points_2d)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute extent
    centroid = points_2d.mean(axis=0)
    centered = points_2d - centroid
    distances = np.dot(centered, wall_direction)

    min_dist = distances.min()
    max_dist = distances.max()

    start = centroid + min_dist * wall_direction
    end = centroid + max_dist * wall_direction
    length = np.linalg.norm(end - start)

    return {
        "start": start,
        "end": end,
        "direction": wall_direction,
        "length": length,
        "centroid": centroid,
        "plane_model": plane_model,
        "num_points": len(points_3d),
        "method": method,
    }


def extract_wall_line_2d_simple(wall_segment, method="plane_normal"):
    """
    Extract 2D line segment from wall point cloud

    Takes a wall's 3D point cloud and plane equation, converts it to a
    clean 2D line segment with start and end points.

    Args:
        wall_segment: Dict with:
            - 'points': Nx3 array of wall points
            - 'plane_model': [a, b, c, d] plane equation (ax + by + cz + d = 0)
        method: How to determine line direction
            - 'plane_normal': Use plane normal (perpendicular to wall) [fastest]
            - 'pca': Use PCA on points (data-driven)
            - 'hough': Use Hough Transform (most robust to outliers)

    Returns:
        line_info: Dict with:
            - 'start': (x, y) start point
            - 'end': (x, y) end point
            - 'direction': (dx, dy) unit vector along wall
            - 'length': length in meters
            - 'centroid': (x, y) center of wall
            - 'plane_model': original plane equation
            - 'method': method used
    """
    points_3d = wall_segment["points"]
    plane_model = wall_segment["plane_model"]
    a, b, c, d = plane_model

    # Project to 2D (ignore Z)
    points_2d = points_3d[:, :2]

    # ========================================================================
    # METHOD SELECTION
    # ========================================================================

    if method == "plane_normal":
        # The plane normal [a, b, c] is perpendicular to the wall
        # In 2D, the wall direction is perpendicular to [a, b]
        # Rotate [a, b] by 90° to get wall direction
        wall_direction = np.array([-b, a])

        # Normalize to unit vector
        norm = np.linalg.norm(wall_direction)
        if norm < 1e-6:
            print("Warning: Wall direction has zero length")
            wall_direction = np.array([1, 0])
        else:
            wall_direction = wall_direction / norm

    elif method == "pca":
        # Use PCA to find principal direction of points
        centroid = points_2d.mean(axis=0)
        centered = points_2d - centroid

        # Covariance matrix
        cov = np.cov(centered.T)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Principal direction (eigenvector with largest eigenvalue)
        principal_idx = np.argmax(eigenvalues)
        wall_direction = eigenvectors[:, principal_idx].real

    elif method == "hough":
        # Use Hough Transform to detect line
        wall_direction = extract_line_direction_hough(points_2d)

    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'plane_normal', 'pca', or 'hough'"
        )

    # ========================================================================
    # COMPUTE WALL EXTENT
    # ========================================================================

    centroid = points_2d.mean(axis=0)

    # Project all points onto the wall direction
    centered = points_2d - centroid
    distances = np.dot(centered, wall_direction)

    # Find min/max distance (wall endpoints)
    min_dist = distances.min()
    max_dist = distances.max()

    # Compute start and end points
    start = centroid + min_dist * wall_direction
    end = centroid + max_dist * wall_direction

    length = np.linalg.norm(end - start)

    line_info = {
        "start": start,
        "end": end,
        "direction": wall_direction,
        "length": length,
        "centroid": centroid,
        "plane_model": plane_model,
        "num_points": len(points_3d),
        "method": method,
    }

    return line_info


def find_all_line_intersections(wall_lines):
    """
    Find all intersection points between wall lines

    For 4 walls forming a rectangle, should give 4 corners

    Parameters:
    -----------
    wall_lines : list of dict
        Each has 'start', 'end', 'direction', 'centroid'

    Returns:
    --------
    corners : list of dict
        Each corner has 'point', 'wall_indices', 'angle'
    """
    from itertools import combinations

    print(f"\n{'='*70}")
    print(f"FINDING LINE INTERSECTIONS")
    print(f"{'='*70}")

    corners = []

    # Try all pairs of lines
    for i, j in combinations(range(len(wall_lines)), 2):
        line_i = wall_lines[i]
        line_j = wall_lines[j]

        # Check angle between lines
        dir_i = line_i["direction"]
        dir_j = line_j["direction"]

        dot = np.dot(dir_i, dir_j)
        angle = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))

        print(f"\nWall {i} ↔ Wall {j}:")
        print(f"  Angle between lines: {angle:.1f}°")

        # Check if lines are roughly perpendicular (for rectangle)
        if angle < 15:  # Parallel (< 15°)
            print(f"  → Parallel (skip)")
            continue
        elif angle > 165:  # Also parallel (> 165°)
            print(f"  → Parallel (skip)")
            continue

        # Lines are not parallel, find intersection
        intersection = intersect_infinite_lines(line_i, line_j)

        if intersection is not None:
            corners.append(
                {"point": intersection, "wall_indices": (i, j), "angle": angle}
            )

            print(
                f"  → Intersection at: [{intersection[0]:.3f}, {intersection[1]:.3f}]"
            )
            print(f"  → Angle: {angle:.1f}°")

    print(f"\n{'='*70}")
    print(f"FOUND {len(corners)} CORNERS")
    print(f"{'='*70}")

    return corners


def intersect_infinite_lines(line1, line2):
    """
    Find intersection of two infinite lines (not segments)

    Each line defined by: point + t * direction

    Parameters:
    -----------
    line1, line2 : dict
        Each has 'centroid' and 'direction'

    Returns:
    --------
    intersection : np.ndarray (2,) or None
        Intersection point [x, y] or None if parallel
    """

    # Line 1: p1 + t1 * d1
    p1 = line1["centroid"]
    d1 = line1["direction"]

    # Line 2: p2 + t2 * d2
    p2 = line2["centroid"]
    d2 = line2["direction"]

    # Solve: p1 + t1*d1 = p2 + t2*d2
    # Rearrange: t1*d1 - t2*d2 = p2 - p1
    # Matrix form: [d1 | -d2] * [t1; t2] = p2 - p1

    A = np.column_stack([d1, -d2])
    b = p2 - p1

    # Check if lines are parallel (determinant ≈ 0)
    det = np.linalg.det(A)

    if abs(det) < 1e-6:
        print(f"    Lines are parallel (det={det:.2e})")
        return None

    try:
        t = np.linalg.solve(A, b)

        # Calculate intersection point (should be same from both lines)
        intersection_1 = p1 + t[0] * d1
        intersection_2 = p2 + t[1] * d2

        # Verify they're the same (sanity check)
        diff = np.linalg.norm(intersection_1 - intersection_2)
        if diff > 1e-3:
            print(f"    Warning: Intersection mismatch: {diff*1000:.1f}mm")

        return intersection_1

    except np.linalg.LinAlgError:
        print(f"    Failed to solve linear system")
        return None


def calculate_rectangle_dimensions(corners, wall_lines):
    """
    Calculate rectangle dimensions from corners

    Parameters:
    -----------
    corners : list of dict
        Corner information
    wall_lines : list of dict
        Wall line information

    Returns:
    --------
    dimensions : dict
        Rectangle dimensions and statistics
    """

    if len(corners) != 4:
        print(f"⚠️  Warning: Expected 4 corners, got {len(corners)}")
        if len(corners) < 4:
            print(f"   Cannot form rectangle with {len(corners)} corners")
            return None

    print(f"\n{'='*70}")
    print(f"CALCULATING RECTANGLE DIMENSIONS")
    print(f"{'='*70}")

    # Extract corner points
    corner_points = np.array([c["point"] for c in corners])

    # Order corners (clockwise or counter-clockwise)
    # Find centroid
    centroid = corner_points.mean(axis=0)

    # Calculate angles from centroid
    angles = []
    for point in corner_points:
        vec = point - centroid
        angle = np.arctan2(vec[1], vec[0])
        angles.append(angle)

    # Sort by angle
    sorted_indices = np.argsort(angles)
    ordered_corners = corner_points[sorted_indices]

    print(f"\nOrdered corners (counter-clockwise):")
    for i, corner in enumerate(ordered_corners):
        print(f"  Corner {i+1}: [{corner[0]:7.3f}, {corner[1]:7.3f}]")

    # Calculate edge lengths
    edges = []
    for i in range(4):
        corner_curr = ordered_corners[i]
        corner_next = ordered_corners[(i + 1) % 4]

        edge_length = np.linalg.norm(corner_next - corner_curr)
        edges.append(edge_length)

        print(f"\nEdge {i+1}: {edge_length:.3f}m")
        print(f"  From corner {i+1} to corner {((i+1) % 4) + 1}")

    # Calculate corner angles
    corner_angles = []
    for i in range(4):
        corner_prev = ordered_corners[(i - 1) % 4]
        corner_curr = ordered_corners[i]
        corner_next = ordered_corners[(i + 1) % 4]

        # Vectors
        v1 = corner_prev - corner_curr
        v2 = corner_next - corner_curr

        # Angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        corner_angles.append(angle)

        deviation = abs(angle - 90)
        status = "✓" if deviation < 5 else "⚠️"

        print(
            f"\nCorner {i+1} angle: {angle:.2f}° (deviation: {deviation:.2f}°) {status}"
        )

    # Calculate perimeter and area
    perimeter = sum(edges)

    # Area using shoelace formula
    x = ordered_corners[:, 0]
    y = ordered_corners[:, 1]
    area = 0.5 * abs(
        sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))
    )

    # Identify width and height
    # Opposite edges should be similar
    width = (edges[0] + edges[2]) / 2
    height = (edges[1] + edges[3]) / 2

    # Validate rectangle
    edge_variance = np.std([edges[0] - edges[2], edges[1] - edges[3]])
    angle_variance = np.std([abs(a - 90) for a in corner_angles])

    is_valid = (edge_variance < 0.1) and (angle_variance < 5)

    print(f"\n{'='*70}")
    print(f"RECTANGLE SUMMARY")
    print(f"{'='*70}")
    print(f"Width:  {width:.3f}m (edges 1 & 3: {edges[0]:.3f}m, {edges[2]:.3f}m)")
    print(f"Height: {height:.3f}m (edges 2 & 4: {edges[1]:.3f}m, {edges[3]:.3f}m)")
    print(f"Perimeter: {perimeter:.3f}m")
    print(f"Area: {area:.3f}m²")
    print(f"Valid rectangle: {is_valid}")

    dimensions = {
        "corners": ordered_corners,
        "edges": edges,
        "width": width,
        "height": height,
        "perimeter": perimeter,
        "area": area,
        "corner_angles": corner_angles,
        "is_valid": is_valid,
        "edge_variance": edge_variance,
        "angle_variance": angle_variance,
    }

    return dimensions


def visualize_all_wall_lines(walls, wall_lines):
    """
    Visualize all walls and their extracted lines
    """
    plt.figure(figsize=(12, 10))

    colors = ["red", "blue", "green", "orange"]

    for i, (wall, line_info) in enumerate(zip(walls, wall_lines)):
        color = colors[i % len(colors)]

        # Plot points
        points_2d = wall["points"][:, :2]
        plt.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            s=1,
            c=color,
            alpha=0.3,
            label=f"Wall {i} points",
        )

        # Plot extracted line
        start = line_info["start"]
        end = line_info["end"]
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=3,
            label=f"Wall {i} line",
        )
        plt.plot([start[0], end[0]], [start[1], end[1]], "o", color=color, markersize=8)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("All Walls with Extracted Lines")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_walls_with_corners(wall_lines, corners):
    """
    Visualize wall lines and their intersection corners
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = ["red", "blue", "green", "orange"]

    # Plot each wall line
    for i, line_info in enumerate(wall_lines):
        color = colors[i % len(colors)]

        start = line_info["start"]
        end = line_info["end"]

        # Plot line
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=3,
            label=f'Wall {i} ({line_info["length"]:.2f}m)',
            alpha=0.8,
        )

        # Plot endpoints
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            "o",
            color=color,
            markersize=8,
            alpha=0.5,
        )

        # Label
        mid = (start + end) / 2
        ax.text(
            mid[0],
            mid[1],
            f"W{i}",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Plot corners
    if len(corners) > 0:
        corner_points = np.array([c["point"] for c in corners])

        ax.scatter(
            corner_points[:, 0],
            corner_points[:, 1],
            c="black",
            s=200,
            marker="o",
            edgecolors="yellow",
            linewidths=3,
            label="Corners",
            zorder=10,
        )

        # Label corners
        for i, corner in enumerate(corners):
            point = corner["point"]
            walls = corner["wall_indices"]
            ax.text(
                point[0],
                point[1],
                f"C{i+1}",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
            )

    ax.set_xlabel("X (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Wall Lines with {len(corners)} Corners", fontsize=14, fontweight="bold"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_complete_floorplan(walls, wall_lines, corners, dimensions):
    """
    Complete visualization: wall points + lines + corners + dimensions
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 12))

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    # ========================================================================
    # Plot wall point clouds
    # ========================================================================

    for i, wall in enumerate(walls):
        points_2d = wall["points"][:, :2]
        color = colors[i % len(colors)]

        ax.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            s=1,
            c=color,
            alpha=0.2,
            label=f"Wall {i+1} points",
        )

    # ========================================================================
    # Plot extracted wall lines
    # ========================================================================

    for i, line_info in enumerate(wall_lines):
        color = colors[i % len(colors)]
        start = line_info["start"]
        end = line_info["end"]

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=4,
            alpha=0.9,
            solid_capstyle="round",
        )

    # ========================================================================
    # Plot corners and dimensions (if available)
    # ========================================================================

    if dimensions and len(dimensions["corners"]) == 4:
        ordered_corners = dimensions["corners"]
        edges = dimensions["edges"]

        # Corners
        ax.scatter(
            ordered_corners[:, 0],
            ordered_corners[:, 1],
            c="black",
            s=300,
            marker="o",
            edgecolors="yellow",
            linewidths=4,
            zorder=10,
        )

        for i, corner in enumerate(ordered_corners):
            ax.text(
                corner[0],
                corner[1],
                f"C{i+1}",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                zorder=11,
            )

        # Dimensions
        for i in range(4):
            corner_curr = ordered_corners[i]
            corner_next = ordered_corners[(i + 1) % 4]

            mid = (corner_curr + corner_next) / 2
            edge_vec = corner_next - corner_curr
            edge_length = np.linalg.norm(edge_vec)

            if edge_length > 0:
                edge_vec_norm = edge_vec / edge_length
                perp = np.array([-edge_vec_norm[1], edge_vec_norm[0]])
                label_pos = mid + perp * 0.3

                ax.text(
                    label_pos[0],
                    label_pos[1],
                    f"{edges[i]:.2f}m",
                    fontsize=13,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor="yellow",
                        alpha=0.95,
                        edgecolor="black",
                        linewidth=2,
                    ),
                )

        # Title with dimensions
        ax.set_title(
            f"Complete Floor Plan\n"
            f"{dimensions['width']:.2f}m × {dimensions['height']:.2f}m  |  "
            f"Perimeter: {dimensions['perimeter']:.2f}m  |  "
            f"Area: {dimensions['area']:.2f}m²",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
    else:
        ax.set_title("Complete Floor Plan", fontsize=15, fontweight="bold")

    ax.set_xlabel("X (m)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()
