import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union


def extract_dimensions_from_convex_hull(wall_points_2d):
    """
    Extract wall dimensions from convex hull

    Parameters:
    -----------
    wall_points_2d : np.ndarray (N, 2)
        2D wall points (X, Y)

    Returns:
    --------
    dimensions : list of dict
        Wall dimensions with length, angle, start/end points
    hull_vertices : np.ndarray
        Convex hull vertex coordinates
    """

    print(f"\n{'='*70}")
    print(f"CONVEX HULL DIMENSION EXTRACTION")
    print(f"{'='*70}")
    print(f"Input points: {len(wall_points_2d):,}")

    # Compute convex hull
    hull = ConvexHull(wall_points_2d)

    # Get vertices in order
    hull_vertices = wall_points_2d[hull.vertices]

    print(f"Hull vertices: {len(hull_vertices)}")
    print(f"Hull perimeter: {hull.area:.3f}m")  # In 2D, .area is perimeter!
    print(f"Hull area: {hull.volume:.3f}m²")  # In 2D, .volume is area!

    # Compute dimensions (corner-to-corner)
    dimensions = []

    for i in range(len(hull_vertices)):
        start = hull_vertices[i]
        end = hull_vertices[(i + 1) % len(hull_vertices)]

        # Wall vector
        vec = end - start
        length = np.linalg.norm(vec)

        # Angle (from +X axis)
        angle_rad = np.arctan2(vec[1], vec[0])
        angle_deg = np.degrees(angle_rad) % 360

        dimensions.append(
            {
                "wall_id": i + 1,
                "corner_start": i + 1,
                "corner_end": (i + 1) % len(hull_vertices) + 1,
                "start_point": start,
                "end_point": end,
                "length": length,
                "angle": angle_deg,
            }
        )

    # Print table
    print(f"\n{'='*70}")
    print(f"WALL DIMENSIONS")
    print(f"{'='*70}")
    print(f"{'Wall':<6} {'Length (m)':<12} {'Angle (°)':<10} {'Corner':<10}")
    print(f"{'-'*70}")

    for dim in dimensions:
        print(
            f"W{dim['wall_id']:<5} {dim['length']:<12.3f} {dim['angle']:<10.1f} "
            f"C{dim['corner_start']} → C{dim['corner_end']}"
        )

    total_perimeter = sum(d["length"] for d in dimensions)

    print(f"{'-'*70}")
    print(f"{'TOTAL':<6} {total_perimeter:<12.3f}")
    print(f"{'='*70}")

    return dimensions, hull_vertices


def visualize_convex_hull_dimensions(wall_points_2d, hull_vertices, dimensions):
    """
    Visualize convex hull with dimension labels
    """

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot wall points
    ax.scatter(
        wall_points_2d[:, 0],
        wall_points_2d[:, 1],
        c="blue",
        s=1,
        alpha=0.3,
        label="Wall points",
    )

    # Plot hull polygon
    hull_closed = np.vstack([hull_vertices, hull_vertices[0]])
    ax.plot(
        hull_closed[:, 0], hull_closed[:, 1], "r-", linewidth=2, label="Convex hull"
    )

    # Plot corners
    ax.plot(
        hull_vertices[:, 0],
        hull_vertices[:, 1],
        "go",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=2,
        label="Corners",
    )

    # Label corners
    for i, corner in enumerate(hull_vertices):
        ax.text(
            corner[0],
            corner[1],
            f"C{i+1}",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
        )

    # Label walls with dimensions
    for dim in dimensions:
        # Midpoint
        mid_x = (dim["start_point"][0] + dim["end_point"][0]) / 2
        mid_y = (dim["start_point"][1] + dim["end_point"][1]) / 2

        # Offset label perpendicular to wall
        vec = dim["end_point"] - dim["start_point"]
        length = np.linalg.norm(vec)
        if length > 0:
            vec_norm = vec / length
            perp = np.array([-vec_norm[1], vec_norm[0]])
            offset = 0.20  # 20cm offset
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
                alpha=0.9,
                edgecolor="black",
                linewidth=2,
            ),
        )

    area = Polygon(hull_vertices).area
    total_perim = sum(d["length"] for d in dimensions)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Convex Hull Dimensions\n"
        f"{len(hull_vertices)} corners, "
        f"Perimeter: {total_perim:.2f}m, "
        f"Area: {area:.2f}m²",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def extract_dimensions_from_concave_hull(wall_points_2d, alpha=0.5):
    """
    Extract dimensions using CONCAVE hull (more accurate)

    Parameters:
    -----------
    wall_points_2d : np.ndarray (N, 2)
        2D wall points
    alpha : float
        Concavity parameter (lower = more detail)
        - 0.05 = very detailed
        - 0.5 = moderate
        - 2.0 = almost convex

    Returns:
    --------
    dimensions : list of dict
    hull_vertices : np.ndarray
    """

    print(f"\n{'='*70}")
    print(f"CONCAVE HULL DIMENSION EXTRACTION")
    print(f"{'='*70}")
    print(f"Input points: {len(wall_points_2d):,}")
    print(f"Alpha (concavity): {alpha}")

    # Create concave hull
    multi_point = MultiPoint(wall_points_2d)
    concave_hull = shapely.concave_hull(multi_point, ratio=alpha)

    # Get vertices
    hull_vertices = np.array(concave_hull.exterior.coords)[:-1]

    print(f"Hull vertices: {len(hull_vertices)}")
    print(f"Hull perimeter: {concave_hull.length:.3f}m")
    print(f"Hull area: {concave_hull.area:.3f}m²")

    # Compute dimensions
    dimensions = []

    for i in range(len(hull_vertices)):
        start = hull_vertices[i]
        end = hull_vertices[(i + 1) % len(hull_vertices)]

        vec = end - start
        length = np.linalg.norm(vec)
        angle_deg = np.degrees(np.arctan2(vec[1], vec[0])) % 360

        dimensions.append(
            {
                "wall_id": i + 1,
                "start_point": start,
                "end_point": end,
                "length": length,
                "angle": angle_deg,
            }
        )

    # Print summary
    print(f"\nTotal walls: {len(dimensions)}")
    print(f"Total perimeter: {sum(d['length'] for d in dimensions):.3f}m")

    return dimensions, hull_vertices


def extract_room_boundary_from_walls(
    wall_points_2d, method="concave_hull", concave_ratio=0.3, alpha=1.0
):
    """
    Extract room boundary polygon from all wall points

    Methods:
    - 'concave_hull': Shapely concave hull (best for your case)
    - 'alpha_shape': Alpha shapes

    Parameters:
    -----------
    wall_points_2d : np.ndarray (N, 2)
        All wall points (X, Y only)
    method : str
        Boundary extraction method
    concave_ratio : float
        For concave_hull method (0-1, lower = tighter fit)
    alpha : float
        For alpha_shape method

    Returns:
    --------
    boundary_polygon : shapely.Polygon
        Room boundary
    """

    print(f"\n{'='*70}")
    print(f"BOUNDARY EXTRACTION FROM WALL POINTS")
    print(f"{'='*70}")
    print(f"Wall points: {len(wall_points_2d):,}")
    print(f"Method: {method}")

    # Create multipoint
    multi_point = MultiPoint(wall_points_2d)

    if method == "concave_hull":
        # Use shapely's concave hull (perfect for this!)
        boundary = shapely.concave_hull(multi_point, ratio=concave_ratio)

        print(f"Concave hull ratio: {concave_ratio}")

    elif method == "alpha_shape":
        # Alpha shapes (alternative)
        # Compute Delaunay triangulation
        tri = Delaunay(wall_points_2d)

        # Filter triangles by edge length (alpha parameter)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                p1 = wall_points_2d[simplex[i]]
                p2 = wall_points_2d[simplex[(i + 1) % 3]]
                edge_length = np.linalg.norm(p2 - p1)

                if edge_length < alpha:
                    edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                    edges.add(edge)

        # Build polygon from edges (simplified)
        # This is complex, using concave_hull is easier
        boundary = shapely.concave_hull(multi_point, ratio=concave_ratio)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Validate
    if not isinstance(boundary, Polygon):
        print(f"⚠️  Result is not a polygon, converting...")
        boundary = boundary.convex_hull

    # Get stats
    area = boundary.area
    perimeter = boundary.length
    num_vertices = len(boundary.exterior.coords) - 1

    print(f"\n✓ Boundary extracted:")
    print(f"  Vertices: {num_vertices}")
    print(f"  Perimeter: {perimeter:.2f}m")
    print(f"  Area: {area:.2f}m²")

    return boundary
