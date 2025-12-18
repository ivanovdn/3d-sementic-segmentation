import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


def compute_polygon_dimensions(boundary_polygon):
    """
    Compute wall dimensions from polygon vertices

    Each vertex = corner
    Each edge = wall
    Distance between consecutive vertices = wall length

    Parameters:
    -----------
    polygon_points : np.ndarray (N, 2)
        Simplified polygon vertices (ordered)

    Returns:
    --------
    dimensions : list of dict
        Each wall dimension
    """
    polygon_points = np.array(boundary_polygon.exterior.coords)[:-1]
    print(f"\n{'='*70}")
    print(f"POLYGON DIMENSIONS")
    print(f"{'='*70}")
    print(f"Number of corners: {len(polygon_points)}")

    n = len(polygon_points)
    dimensions = []

    for i in range(n):
        # Corner i to corner i+1
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % n]  # Wrap around

        # Wall vector
        vec = p2 - p1
        length = np.linalg.norm(vec)

        # Wall angle (from +X axis)
        angle_rad = np.arctan2(vec[1], vec[0])
        angle_deg = np.degrees(angle_rad) % 360

        dimensions.append(
            {
                "wall_id": i + 1,
                "corner_start": i + 1,
                "corner_end": (i + 1) % n + 1,
                "start_point": p1,
                "end_point": p2,
                "length": length,
                "angle": angle_deg,
            }
        )

        print(
            f"Wall {i+1:2d}: Corner {i+1} → Corner {(i+1)%n + 1}, "
            f"Length: {length:6.3f}m, Angle: {angle_deg:6.1f}°"
        )

    # Total perimeter
    total_perimeter = sum(d["length"] for d in dimensions)

    # Area (using shoelace formula)

    area = Polygon(polygon_points).area

    print(f"\n{'─'*70}")
    print(f"Total perimeter: {total_perimeter:.3f}m")
    print(f"Total area:      {area:.3f}m²")
    print(f"{'='*70}")

    return dimensions


def visualize_polygon_dimensions(boundary_polygon, dimensions):
    """
    Visualize polygon with labeled dimensions
    """

    polygon_points = np.array(boundary_polygon.exterior.coords)[:-1]
    fig, ax = plt.subplots(figsize=(10, 8))

    # Close polygon for plotting
    poly_closed = np.vstack([polygon_points, polygon_points[0]])

    # Plot polygon edges
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], "b-", linewidth=2, alpha=0.6)

    # Plot corners
    ax.plot(
        polygon_points[:, 0],
        polygon_points[:, 1],
        "ro",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=2,
        label="Corners",
    )

    # Label corners
    for i, point in enumerate(polygon_points):
        ax.text(
            point[0],
            point[1],
            f"C{i+1}",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
        )

    # Label wall dimensions
    for dim in dimensions:
        # Midpoint
        mid_x = (dim["start_point"][0] + dim["end_point"][0]) / 2
        mid_y = (dim["start_point"][1] + dim["end_point"][1]) / 2

        # Offset label perpendicular to wall
        vec = dim["end_point"] - dim["start_point"]
        length = np.linalg.norm(vec)
        if length > 0:
            vec_norm = vec / length
            # Perpendicular vector (rotate 90°)
            perp = np.array([-vec_norm[1], vec_norm[0]])
            offset = 0.15  # 15cm offset
            label_x = mid_x + perp[0] * offset
            label_y = mid_y + perp[1] * offset
        else:
            label_x, label_y = mid_x, mid_y

        # Label
        ax.text(
            label_x,
            label_y,
            f"W{dim['wall_id']}\n{dim['length']:.2f}m",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="yellow",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            ),
        )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=12, fontweight="bold")

    # Title with summary
    total_perim = sum(d["length"] for d in dimensions)
    area = Polygon(polygon_points).area
    ax.set_title(
        f"Room Dimensions\n"
        f"{len(polygon_points)} corners, "
        f"Perimeter: {total_perim:.2f}m, "
        f"Area: {area:.2f}m²",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def compute_median_wall_thickness(walls):
    """
    Compute median wall thickness from detected walls

    Parameters:
    -----------
    walls : list of dict
        Detected walls from RANSAC (with 'thickness' field)

    Returns:
    --------
    median_thickness : float
        Median wall thickness (meters)
    """

    thicknesses = [w["thickness"] for w in walls if w["thickness"] < 0.5]

    if len(thicknesses) == 0:
        print("⚠️  No valid wall thickness measurements!")
        return 0.15  # Default: 15cm

    median = np.median(thicknesses)
    mean = np.mean(thicknesses)

    # print(f"\n{'='*70}")
    # print(f"WALL THICKNESS ANALYSIS")
    # print(f"{'='*70}")
    # print(f"Walls analyzed: {len(thicknesses)}")
    # print(f"Median thickness: {median:.3f}m ({median*100:.1f}cm)")
    # print(f"Mean thickness: {mean:.3f}m ({mean*100:.1f}cm)")
    # print(f"Range: {min(thicknesses):.3f}m - {max(thicknesses):.3f}m")

    return median


def offset_boundary_inward(boundary_polygon, offset_distance):
    """
    Shrink boundary polygon inward by offset distance

    This converts EXPLICIT boundary (scan surface)
    to IMPLICIT boundary (wall centerline or inner edge)

    Parameters:
    -----------
    boundary_polygon : shapely.Polygon
        Original boundary (from concave hull)
    offset_distance : float
        Distance to offset inward (meters)
        - Use wall_thickness/2 for centerline
        - Use wall_thickness for inner edge

    Returns:
    --------
    offset_polygon : shapely.Polygon
        Offset boundary
    """

    # print(f"\n{'='*70}")
    # print(f"BOUNDARY OFFSET (EXPLICIT → IMPLICIT)")
    # print(f"{'='*70}")
    # print(f"Offset distance: {offset_distance:.3f}m ({offset_distance*100:.1f}cm)")

    # Negative buffer = shrink inward
    offset_polygon = boundary_polygon.buffer(-offset_distance)

    # Check result
    if offset_polygon.is_empty:
        # print("⚠️  Offset too large! Polygon collapsed.")
        # print("   Using half offset...")
        offset_polygon = boundary_polygon.buffer(-offset_distance / 2)

    original_area = boundary_polygon.area
    offset_area = offset_polygon.area
    area_reduction = original_area - offset_area

    # print(f"\nOriginal area: {original_area:.2f}m²")
    # print(f"Offset area:   {offset_area:.2f}m²")
    # print(
    #     f"Reduction:     {area_reduction:.2f}m² ({area_reduction/original_area*100:.1f}%)"
    # )

    return offset_polygon
