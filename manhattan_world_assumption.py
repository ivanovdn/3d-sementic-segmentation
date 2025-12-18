import numpy as np
from sklearn.cluster import KMeans


class ManhattanWorldRectangleFitter:

    def __init__(self, wall_lines, wall_points):
        self.wall_lines = wall_lines
        self.wall_points = wall_points
        self.n_walls = len(wall_lines)

    def fit(self, verbose=True):
        """Fit rectangle with correct direction snapping"""

        if verbose:
            print(f"\n{'='*70}")
            print(f"MANHATTAN WORLD FITTING")
            print(f"{'='*70}")

        # ====================================================================
        # STEP 1: Get wall directions (as unit vectors)
        # ====================================================================

        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 1: WALL DIRECTIONS")
            print(f"{'='*70}")

        directions = []
        angles = []

        for i, wall in enumerate(self.wall_lines):
            direction = np.array(wall["direction"])
            angle = np.arctan2(direction[1], direction[0])

            directions.append(direction)
            angles.append(angle)

            if verbose:
                print(
                    f"Wall {i}: [{direction[0]:6.3f}, {direction[1]:6.3f}], "
                    f"angle={np.degrees(angle):7.2f}°"
                )

        directions = np.array(directions)
        angles = np.array(angles)

        # ====================================================================
        # STEP 2: Find 2 dominant PERPENDICULAR directions
        # ====================================================================

        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 2: FIND DOMINANT PERPENDICULAR DIRECTIONS")
            print(f"{'='*70}")

        # Key insight: In a rectangle, we expect TWO perpendicular directions
        # Method: Find one dominant direction, then make second perpendicular

        # Project all directions to [-90°, 90°] range
        # This maps opposite directions to the same direction
        angles_normalized = []
        for angle in angles:
            # Normalize to [-π/2, π/2]
            while angle > np.pi / 2:
                angle -= np.pi
            while angle < -np.pi / 2:
                angle += np.pi
            angles_normalized.append(angle)

        angles_normalized = np.array(angles_normalized)

        if verbose:
            print(f"\nNormalized angles:")
            for i, (orig, norm) in enumerate(zip(angles, angles_normalized)):
                print(
                    f"  Wall {i}: {np.degrees(orig):7.2f}° → {np.degrees(norm):7.2f}°"
                )

        # Cluster into 2 groups
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(angles_normalized.reshape(-1, 1))

        cluster_centers = km.cluster_centers_.flatten()

        # The two dominant directions (perpendicular by construction)
        dominant_angle_1 = cluster_centers[0]
        dominant_angle_2 = cluster_centers[1]

        # Ensure they're perpendicular
        angle_diff = abs(dominant_angle_2 - dominant_angle_1)

        if verbose:
            print(f"\nClustered directions:")
            print(f"  Direction 1: {np.degrees(dominant_angle_1):7.2f}°")
            print(f"  Direction 2: {np.degrees(dominant_angle_2):7.2f}°")
            print(f"  Angle difference: {np.degrees(angle_diff):.2f}°")

        if abs(angle_diff - np.pi / 2) > np.radians(10):
            # Not perpendicular enough, force it
            if verbose:
                print(f"  ⚠️  Not perpendicular, forcing 90°")

            avg_angle = (dominant_angle_1 + dominant_angle_2) / 2
            dominant_angle_1 = avg_angle
            dominant_angle_2 = avg_angle + np.pi / 2

        # Convert to direction vectors
        dominant_dir_1 = np.array([np.cos(dominant_angle_1), np.sin(dominant_angle_1)])
        dominant_dir_2 = np.array([np.cos(dominant_angle_2), np.sin(dominant_angle_2)])

        if verbose:
            print(f"\nDominant direction vectors:")
            print(
                f"  Direction 1: [{dominant_dir_1[0]:6.3f}, {dominant_dir_1[1]:6.3f}]"
            )
            print(
                f"  Direction 2: [{dominant_dir_2[0]:6.3f}, {dominant_dir_2[1]:6.3f}]"
            )
            print(
                f"  Dot product: {np.dot(dominant_dir_1, dominant_dir_2):.6f} (should be ~0)"
            )

        # ====================================================================
        # STEP 3: Snap each wall to nearest dominant direction
        # ====================================================================

        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 3: SNAP WALLS TO DOMINANT DIRECTIONS")
            print(f"{'='*70}")

        snapped_directions = []
        snapped_labels = []

        for i, direction in enumerate(directions):
            # Calculate alignment with each dominant direction
            # Use absolute dot product (direction can be flipped)
            align_1 = abs(np.dot(direction, dominant_dir_1))
            align_2 = abs(np.dot(direction, dominant_dir_2))

            if align_1 > align_2:
                # Closer to direction 1
                snapped_dir = dominant_dir_1
                label = 0

                # Check if we need to flip
                if np.dot(direction, dominant_dir_1) < 0:
                    snapped_dir = -dominant_dir_1
            else:
                # Closer to direction 2
                snapped_dir = dominant_dir_2
                label = 1

                # Check if we need to flip
                if np.dot(direction, dominant_dir_2) < 0:
                    snapped_dir = -dominant_dir_2

            snapped_directions.append(snapped_dir)
            snapped_labels.append(label)

            # Calculate error
            original_angle = np.arctan2(direction[1], direction[0])
            snapped_angle = np.arctan2(snapped_dir[1], snapped_dir[0])

            # Angular error (handles wrapping)
            error_angle = np.arctan2(
                np.sin(original_angle - snapped_angle),
                np.cos(original_angle - snapped_angle),
            )
            error_deg = abs(np.degrees(error_angle))

            if verbose:
                print(f"\nWall {i}:")
                print(
                    f"  Original: [{direction[0]:6.3f}, {direction[1]:6.3f}] "
                    f"({np.degrees(original_angle):7.2f}°)"
                )
                print(
                    f"  Snapped:  [{snapped_dir[0]:6.3f}, {snapped_dir[1]:6.3f}] "
                    f"({np.degrees(snapped_angle):7.2f}°)"
                )
                print(f"  Direction group: {label+1}")
                print(f"  Error: {error_deg:.2f}°")

                if error_deg > 5:
                    print(f"  ⚠️  Large snap error!")

        # ====================================================================
        # STEP 4: Refit line offsets
        # ====================================================================

        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 4: REFIT LINE OFFSETS")
            print(f"{'='*70}")

        def fit_line_offset(points, direction):
            """Fit line with fixed direction through points"""
            normal = np.array([-direction[1], direction[0]])
            centroid = points.mean(axis=0)
            d = -normal.dot(centroid)
            return normal, d

        snapped_lines = []

        for i, snapped_dir in enumerate(snapped_directions):
            wall_pts = self.wall_points[i][:, :2]

            normal, d = fit_line_offset(wall_pts, snapped_dir)

            snapped_line = {
                "direction": snapped_dir,
                "normal": normal,
                "offset": d,
                "label": snapped_labels[i],
            }

            snapped_lines.append(snapped_line)

            # Calculate RMSE
            distances = np.abs(wall_pts.dot(normal) + d)
            rms = np.sqrt(np.mean(distances**2))

            if verbose:
                print(f"\nWall {i} (Direction group {snapped_labels[i]+1}):")
                print(f"  Direction: [{snapped_dir[0]:6.3f}, {snapped_dir[1]:6.3f}]")
                print(f"  Normal: [{normal[0]:6.3f}, {normal[1]:6.3f}]")
                print(f"  Offset: {d:7.3f}")
                print(f"  RMS error: {rms*100:.2f}cm")

                if rms > 0.10:
                    print(f"  ⚠️  High RMS error")

        # ====================================================================
        # STEP 5: Group walls by label
        # ====================================================================

        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 5: GROUP WALLS")
            print(f"{'='*70}")

        groups = {0: [], 1: []}
        for i, (line, label) in enumerate(zip(snapped_lines, snapped_labels)):
            groups[label].append((i, line))

        if verbose:
            print(f"\nWall groups:")
            for label, walls in groups.items():
                wall_ids = [w[0] for w in walls]
                print(f"  Group {label+1}: Walls {wall_ids}")

        if len(groups[0]) != 2 or len(groups[1]) != 2:
            print(f"\n❌ ERROR: Expected 2 walls per group")
            print(f"   Group 1: {len(groups[0])} walls")
            print(f"   Group 2: {len(groups[1])} walls")
            return None

        # ====================================================================
        # STEP 6: Intersect lines to find corners
        # ====================================================================

        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 6: FIND CORNERS")
            print(f"{'='*70}")

        def intersect_lines(line1, line2):
            """Intersect two lines: n1·x + d1 = 0, n2·x + d2 = 0"""
            n1, d1 = line1["normal"], line1["offset"]
            n2, d2 = line2["normal"], line2["offset"]

            A = np.vstack([n1, n2])
            b = -np.array([d1, d2])

            det = np.linalg.det(A)
            if abs(det) < 1e-10:
                return None

            return np.linalg.solve(A, b)

        # Get the 4 walls
        idx_a0, line_a0 = groups[0][0]
        idx_a1, line_a1 = groups[0][1]
        idx_b0, line_b0 = groups[1][0]
        idx_b1, line_b1 = groups[1][1]

        # Find 4 corners
        corner_pairs = [
            ((idx_a0, line_a0), (idx_b0, line_b0)),
            ((idx_b0, line_b0), (idx_a1, line_a1)),
            ((idx_a1, line_a1), (idx_b1, line_b1)),
            ((idx_b1, line_b1), (idx_a0, line_a0)),
        ]

        corners = []

        for i, ((idx1, line1), (idx2, line2)) in enumerate(corner_pairs):
            corner = intersect_lines(line1, line2)

            if corner is None:
                print(f"❌ Failed to intersect walls {idx1} and {idx2}")
                return None

            corners.append(corner)

            if verbose:
                print(
                    f"  Corner {i+1} (Wall {idx1} ∩ Wall {idx2}): "
                    f"[{corner[0]:7.3f}, {corner[1]:7.3f}]"
                )

        corners = np.array(corners)

        # ====================================================================
        # STEP 7: Calculate rectangle properties
        # ====================================================================

        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 7: RECTANGLE PROPERTIES")
            print(f"{'='*70}")

        edges = []
        for i in range(4):
            edge = np.linalg.norm(corners[(i + 1) % 4] - corners[i])
            edges.append(edge)

        if verbose:
            print(f"\nEdges:")
            for i, edge in enumerate(edges):
                print(f"  Edge {i}: {edge:.4f}m")

            print(f"\nOpposite edges:")
            print(
                f"  0 vs 2: {edges[0]:.4f}m vs {edges[2]:.4f}m "
                f"(diff: {abs(edges[0]-edges[2])*100:.2f}cm)"
            )
            print(
                f"  1 vs 3: {edges[1]:.4f}m vs {edges[3]:.4f}m "
                f"(diff: {abs(edges[1]-edges[3])*100:.2f}cm)"
            )

            print(f"\nCorner angles:")
            for i in range(4):
                v1 = corners[(i - 1) % 4] - corners[i]
                v2 = corners[(i + 1) % 4] - corners[i]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                error = abs(angle - 90)
                print(f"  Corner {i+1}: {angle:.4f}° (error: {error:.4f}°)")

        width = (edges[0] + edges[2]) / 2
        height = (edges[1] + edges[3]) / 2
        center = corners.mean(axis=0)
        perimeter = sum(edges)
        area = width * height

        rectangle = {
            "corners": corners,
            "edges": edges,
            "width": width,
            "height": height,
            "center": center,
            "perimeter": perimeter,
            "area": area,
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ FINAL RECTANGLE")
            print(f"{'='*70}")
            print(f"Width:  {width:.4f}m")
            print(f"Height: {height:.4f}m")
            print(f"Perimeter: {perimeter:.4f}m")
            print(f"Area: {area:.2f}m²")
            print(f"Center: [{center[0]:.3f}, {center[1]:.3f}]")

        return rectangle
