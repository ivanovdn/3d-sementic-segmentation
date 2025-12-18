import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from simplification.cutil import simplify_coords_vw


def compare_vw(boundary_polygon, thresholds):
    # Get your concave hull points
    polygon_points = np.array(boundary_polygon.exterior.coords)

    # V-W uses "effective area" threshold (m²), not distance
    # Smaller threshold = more detail, larger threshold = more simplified

    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5))

    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]

        # Apply Visvalingam-Whyatt simplification
        simplified_points = simplify_coords_vw(polygon_points, epsilon=threshold)

        # Create polygon for area calculation
        simplified_poly = Polygon(simplified_points)

        # Plot
        x, y = simplified_points[:, 0], simplified_points[:, 1]
        ax.fill(x, y, alpha=0.3, fc="blue")
        ax.plot(x, y, "r-", linewidth=2)
        ax.plot(x, y, "ro", markersize=6)

        corners = len(simplified_points) - 1
        area = simplified_poly.area

        # # Test 10-point deviation if enough points
        # if len(simplified_points) >= 10:
        #     # from wall_segment_detection import fit_line_to_points
        #     _, _, err = fit_line_to_points(simplified_points[0:10])
        #     error_text = f'\n10-pt err: {err*100:.1f}cm'
        # else:
        #     error_text = ''

        ax.set_title(
            f"Threshold={threshold}m²\n{corners} corners, Area={area:.2f}m²",
            fontsize=10,
        )
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.show()

    # Also print summary
    print("\n" + "=" * 70)
    print("VISVALINGAM-WHYATT SIMPLIFICATION COMPARISON")
    print("=" * 70)
    print(f"{'Threshold (m²)':<15} {'Vertices':<10} {'Area (m²)':<10}")
    print("-" * 70)

    for threshold in thresholds:
        simplified_points = simplify_coords_vw(polygon_points, epsilon=threshold)
        poly = Polygon(simplified_points)
        print(f"{threshold:<15.4f} {len(simplified_points)-1:<10} {poly.area:<10.2f}")


def compare_dp(boundary_polygon, tolerances):
    fig, axes = plt.subplots(1, len(tolerances), figsize=(5 * len(tolerances), 5))

    for idx, tolerance in enumerate(tolerances):
        ax = axes[idx]

        # mimplify
        simplified = boundary_polygon.simplify(tolerance, preserve_topology=True)

        x, y = simplified.exterior.xy
        ax.fill(x, y, alpha=0.3, fc="blue")
        ax.plot(x, y, "r-", linewidth=2)
        ax.plot(x, y, "ro", markersize=6)

        corners = len(simplified.exterior.coords) - 1
        area = simplified.area

        ax.set_title(f"Tolerance={tolerance}m\n{corners} corners, Area={area:.1f}m²")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    for tolerance in tolerances:
        simplified = boundary_polygon.simplify(tolerance, preserve_topology=True)
        corners = len(simplified.exterior.coords) - 1
        area = simplified.area
        print(f"{tolerance:<15.4f} {corners:<10} {area:<10.2f}")
