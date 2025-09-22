import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


class StructuralRANSAC:
    def __init__(self, point_cloud, voxel_size=0.02):
        """
        Initialize with point cloud
        """
        # Downsample for efficiency
        self.original_cloud = point_cloud
        self.cloud = point_cloud.voxel_down_sample(voxel_size)

        # Get points and normals
        self.cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        self.points = np.asarray(self.cloud.points)
        self.normals = np.asarray(self.cloud.normals)

        # Build KDTree for neighbor searches
        self.kdtree = KDTree(self.points)

        # Parameters
        self.params = {
            "ransac_threshold": 0.05,  # 5cm distance threshold
            "ransac_iterations": 1000,
            "min_plane_points": 100,
            "angle_threshold_deg": 10,  # For normal similarity
            "wall_angle_tolerance": 10,  # Degrees from vertical
            "floor_ceiling_angle_tolerance": 10,  # Degrees from horizontal
        }

        # Beam parametrs
        self.params.update(
            {
                "beam_min_length": 1.0,  # Minimum 1 meter long
                "beam_max_width": 0.5,  # Maximum 50cm wide
                "beam_max_height": 0.4,  # Maximum 40cm tall
                "beam_elongation_ratio": 3.0,  # Length must be 3x width
                "beam_ceiling_distance": 0.5,  # Max distance from ceiling
                "beam_min_points": 50,  # Minimum points for valid beam
                "beam_merge_distance": 0.3,  # Distance to merge parallel beams
            }
        )

        # Results storage
        self.segments = {}
        self.remaining_indices = np.arange(len(self.points))

    def ransac_plane(
        self, indices=None, normal_constraint=None, position_constraint=None
    ):
        """
        Core RANSAC plane fitting with constraints

        normal_constraint: Expected normal direction [x, y, z]
        position_constraint: 'lowest' for floor, 'highest' for ceiling
        """
        if indices is None:
            indices = self.remaining_indices

        if len(indices) < self.params["min_plane_points"]:
            return None, []

        points = self.points[indices]
        normals = self.normals[indices] if self.normals is not None else None

        best_plane = None
        best_inliers = []
        best_score = 0

        for _ in range(self.params["ransac_iterations"]):
            # Sample 3 random points
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_idx]

            # Calculate plane parameters
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample_points[0])

            # Check normal constraint
            if normal_constraint is not None:
                angle = np.arccos(
                    np.clip(np.abs(np.dot(normal, normal_constraint)), -1, 1)
                )
                if np.degrees(angle) > self.params["angle_threshold_deg"]:
                    continue

            # Calculate distances to plane
            distances = np.abs(np.dot(points, normal) + d)
            inlier_mask = distances < self.params["ransac_threshold"]
            inliers = indices[inlier_mask]

            # Score based on number of inliers and constraint satisfaction
            score = len(inliers)

            # Bonus for position constraint
            if position_constraint == "lowest" and normal[2] > 0.9:
                score *= 1.5
            elif position_constraint == "highest" and normal[2] < -0.9:
                score *= 1.5

            if score > best_score:
                best_score = score
                best_plane = {"normal": normal, "d": d, "point": sample_points[0]}
                best_inliers = inliers

        return best_plane, best_inliers

    def segment_floor_ceiling(self):
        """
        Detect floor and ceiling as the lowest and highest horizontal planes
        """
        print("Detecting floor and ceiling...")

        # Z-axis pointing up
        horizontal_normal = np.array([0, 0, 1])

        # Group points by Z-height
        z_values = self.points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        height_range = z_max - z_min

        # FLOOR: Search in bottom 30% of points
        floor_candidates = self.remaining_indices[
            z_values[self.remaining_indices] < z_min + 0.3 * height_range
        ]
        floor_plane, floor_inliers = self.ransac_plane(
            indices=floor_candidates,
            normal_constraint=horizontal_normal,
            position_constraint="lowest",
        )

        if floor_plane and len(floor_inliers) > self.params["min_plane_points"]:
            self.segments["floor"] = {
                "plane": floor_plane,
                "indices": floor_inliers,
                "type": "floor",
            }
            self.remaining_indices = np.setdiff1d(self.remaining_indices, floor_inliers)
            print(f"  Floor found: {len(floor_inliers)} points")

        # CEILING: Search in top 30% of points
        ceiling_candidates = self.remaining_indices[
            z_values[self.remaining_indices] > z_min + 0.7 * height_range
        ]
        ceiling_plane, ceiling_inliers = self.ransac_plane(
            indices=ceiling_candidates,
            normal_constraint=horizontal_normal,
            position_constraint="highest",
        )

        if ceiling_plane and len(ceiling_inliers) > self.params["min_plane_points"]:
            self.segments["ceiling"] = {
                "plane": ceiling_plane,
                "indices": ceiling_inliers,
                "type": "ceiling",
            }
            self.remaining_indices = np.setdiff1d(
                self.remaining_indices, ceiling_inliers
            )
            print(f"  Ceiling found: {len(ceiling_inliers)} points")

    def segment_walls(self, num_walls=10):
        """
        Detect multiple wall planes
        """
        print("Detecting walls...")
        wall_count = 0

        for i in range(num_walls):
            if len(self.remaining_indices) < self.params["min_plane_points"]:
                break

            # Walls are roughly vertical
            wall_plane, wall_inliers = self.ransac_plane(indices=self.remaining_indices)

            if (
                wall_plane is None
                or len(wall_inliers) < self.params["min_plane_points"]
            ):
                break

            # Check if plane is vertical (normal perpendicular to Z-axis)
            normal = wall_plane["normal"]
            verticality = np.abs(normal[2])  # Should be close to 0 for walls

            if verticality > np.sin(np.radians(self.params["wall_angle_tolerance"])):
                continue  # Not vertical enough, skip

            # Refine wall with region growing
            # wall_inliers = self.region_grow_plane(wall_plane, wall_inliers)

            if len(wall_inliers) > self.params["min_plane_points"]:
                self.segments[f"wall_{wall_count}"] = {
                    "plane": wall_plane,
                    "indices": wall_inliers,
                    "type": "wall",
                }
                self.remaining_indices = np.setdiff1d(
                    self.remaining_indices, wall_inliers
                )
                wall_count += 1
                print(f"  Wall {wall_count} found: {len(wall_inliers)} points")

    def region_grow_plane(self, plane, initial_inliers, growth_threshold=0.1):
        """
        Grow plane region from initial RANSAC inliers
        """
        normal = plane["normal"]
        d = plane["d"]

        # Create queue with boundary points
        inlier_set = set(initial_inliers)
        boundary = set()

        # Find neighbors of initial inliers
        for idx in initial_inliers:
            neighbors = self.kdtree.query_radius([self.points[idx]], r=0.1)[0]
            for n in neighbors:
                if n not in inlier_set and n in self.remaining_indices:
                    boundary.add(n)

        # Grow region
        while boundary:
            current = boundary.pop()
            point = self.points[current]

            # Check distance to plane
            distance = np.abs(np.dot(point, normal) + d)

            if distance < growth_threshold:
                # Check normal consistency if available
                if self.normals is not None:
                    point_normal = self.normals[current]
                    angle = np.arccos(
                        np.clip(np.abs(np.dot(point_normal, normal)), -1, 1)
                    )
                    if np.degrees(angle) > self.params["angle_threshold_deg"]:
                        continue

                # Add to inliers
                inlier_set.add(current)

                # Add new neighbors to boundary
                neighbors = self.kdtree.query_radius([point], r=0.1)[0]
                for n in neighbors:
                    if n not in inlier_set and n in self.remaining_indices:
                        boundary.add(n)

        return np.array(list(inlier_set))

    def segment_beams(self):
        """
        Main beam detection method
        """
        print("Detecting beams...")

        # Get ceiling info if available
        ceiling_plane = self.segments.get("ceiling", {}).get("plane", None)
        if ceiling_plane is None:
            print("  Warning: No ceiling found, beam detection may be less accurate")
            ceiling_z = np.max(self.points[:, 2])  # Use max height as approximation
        else:
            # Get average ceiling height
            ceiling_indices = self.segments["ceiling"]["indices"]
            ceiling_z = np.mean(self.points[ceiling_indices, 2])

        # Filter points near ceiling
        beam_candidates = self.get_beam_candidate_points(ceiling_z)

        if len(beam_candidates) < self.params["beam_min_points"]:
            print("  No beam candidates found")
            return

        # Detect beams using multiple strategies
        beams = []

        # Strategy 1: RANSAC-based line detection for linear beams
        linear_beams = self.detect_linear_beams(beam_candidates)
        beams.extend(linear_beams)

        # Strategy 2: Oriented bounding box for rectangular beams
        # box_beams = self.detect_box_beams(beam_candidates)
        # beams.extend(box_beams)

        # Strategy 3: Cylinder detection for round beams
        cylinder_beams = self.detect_cylinder_beams(beam_candidates)
        beams.extend(cylinder_beams)

        # Merge overlapping detections
        merged_beams = self.merge_overlapping_beams(beams)

        # Add to segments
        for i, beam in enumerate(merged_beams):
            self.segments[f"beam_{i}"] = beam
            self.remaining_indices = np.setdiff1d(
                self.remaining_indices, beam["indices"]
            )
            print(
                f"  Beam {i} found: {len(beam['indices'])} points, type: {beam['beam_type']}"
            )

    def get_beam_candidate_points(self, ceiling_z):
        """
        Filter points that could potentially be beams
        """
        # Get points below ceiling but not too far
        z_values = self.points[self.remaining_indices, 2]
        distance_from_ceiling = ceiling_z - z_values

        # Beam candidates are within specified distance from ceiling
        near_ceiling_mask = (distance_from_ceiling >= 0) & (
            distance_from_ceiling <= self.params["beam_ceiling_distance"]
        )

        beam_candidates = self.remaining_indices[near_ceiling_mask]

        # Additional filtering based on local density
        # Beams have higher point density than empty space
        if len(beam_candidates) > 0:
            densities = self.compute_local_density(beam_candidates)
            density_threshold = np.percentile(densities, 30)  # Keep denser regions
            beam_candidates = beam_candidates[densities > density_threshold]

        return beam_candidates

    def compute_local_density(self, indices, radius=0.1):
        """
        Compute local point density for each point
        """
        densities = np.zeros(len(indices))
        points = self.points[indices]

        for i, point in enumerate(points):
            neighbors = self.kdtree.query_radius([point], r=radius)[0]
            densities[i] = len(neighbors) / (
                4 / 3 * np.pi * radius**3
            )  # Points per volume

        return densities

    # Linear Beam Detection (Strategy 1)
    def detect_linear_beams(self, candidate_indices):
        """
        Detect beams as linear structures using RANSAC line fitting
        """
        beams = []
        remaining = candidate_indices.copy()

        while len(remaining) > self.params["beam_min_points"]:
            # RANSAC line fitting
            line, inliers = self.ransac_line_3d(remaining)

            if line is None or len(inliers) < self.params["beam_min_points"]:
                break

            # Validate beam properties
            beam_props = self.compute_beam_properties(inliers)

            if self.validate_beam(beam_props):
                beams.append(
                    {
                        "indices": inliers,
                        "type": "beam",
                        "beam_type": "linear",
                        "line": line,
                        "properties": beam_props,
                    }
                )
                remaining = np.setdiff1d(remaining, inliers)
            else:
                # Remove some points and try again
                remaining = remaining[len(inliers) // 4 :]

        return beams

    def ransac_line_3d(self, indices, iterations=500):
        """
        RANSAC 3D line fitting
        A 3D line is defined by a point and direction vector
        """
        if len(indices) < 2:
            return None, []

        points = self.points[indices]
        best_line = None
        best_inliers = []
        best_score = 0

        for _ in range(iterations):
            # Sample 2 points to define a line
            sample_idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[sample_idx]

            # Line direction vector
            direction = p2 - p1
            if np.linalg.norm(direction) < 0.01:  # Points too close
                continue
            direction = direction / np.linalg.norm(direction)

            # Check if line is roughly horizontal
            if abs(direction[2]) > 0.3:  # More than ~17 degrees from horizontal
                continue

            # Calculate distances from points to line
            # Distance = ||(p - p1) - ((p - p1)·d)d||
            vectors_to_p1 = points - p1
            projections = np.dot(vectors_to_p1, direction)
            perpendicular_vectors = (
                vectors_to_p1 - projections[:, np.newaxis] * direction
            )
            distances = np.linalg.norm(perpendicular_vectors, axis=1)

            # Find inliers (points close to line)
            inlier_mask = distances < self.params["ransac_threshold"]
            inliers = indices[inlier_mask]

            # Score based on number of inliers and line length
            if len(inliers) > self.params["beam_min_points"]:
                line_length = np.max(projections[inlier_mask]) - np.min(
                    projections[inlier_mask]
                )
                score = len(inliers) * (1 + line_length / 10)  # Favor longer lines

                if score > best_score:
                    best_score = score
                    best_line = {"point": p1, "direction": direction}
                    best_inliers = inliers

        return best_line, best_inliers

    # Box Beam Detection (Strategy 2)
    def detect_box_beams(self, candidate_indices):
        """
        Detect beams as oriented bounding boxes
        Better for beams with rectangular cross-sections
        """
        beams = []
        remaining = candidate_indices.copy()

        # Cluster points first
        clusters = self.cluster_points(remaining, eps=0.2)

        for cluster in clusters:
            if len(cluster) < self.params["beam_min_points"]:
                continue

            # Fit oriented bounding box
            obb = self.fit_oriented_bounding_box(cluster)

            if obb is None:
                continue

            # Check if OBB is beam-like
            if self.is_beam_shaped_obb(obb):
                beams.append(
                    {
                        "indices": cluster,
                        "type": "beam",
                        "beam_type": "box",
                        "obb": obb,
                        "properties": self.compute_beam_properties(cluster),
                    }
                )

        return beams

    def fit_oriented_bounding_box(self, indices):
        """
        Fit minimum volume oriented bounding box using PCA
        """
        if len(indices) < self.params["beam_min_points"]:
            return None

        points = self.points[indices]

        # Center points
        center = np.mean(points, axis=0)
        centered_points = points - center

        # PCA to find principal axes
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        pca.fit(centered_points)

        # Transform to principal component space
        transformed = pca.transform(centered_points)

        # Find bounding box in transformed space
        min_bounds = np.min(transformed, axis=0)
        max_bounds = np.max(transformed, axis=0)
        dimensions = max_bounds - min_bounds

        # Create OBB
        obb = {
            "center": center,
            "axes": pca.components_,  # Rotation matrix
            "dimensions": dimensions,
            "volume": np.prod(dimensions),
        }

        return obb

    def is_beam_shaped_obb(self, obb):
        """
        Check if OBB dimensions match beam characteristics
        """
        dims = sorted(obb["dimensions"])  # [smallest, middle, largest]

        # Check elongation: longest dimension should be much larger
        if dims[2] < self.params["beam_min_length"]:
            return False

        if dims[2] / dims[1] < self.params["beam_elongation_ratio"]:
            return False

        # Check cross-section size
        if dims[0] > self.params["beam_max_height"]:
            return False
        if dims[1] > self.params["beam_max_width"]:
            return False

        # Check orientation (longest axis should be horizontal)
        longest_axis = obb["axes"][np.argmax(obb["dimensions"])]
        if abs(longest_axis[2]) > 0.3:  # Too vertical
            return False

        return True

    # Cylinder Beam Detection (Strategy 3)
    def detect_cylinder_beams(self, candidate_indices):
        """
        Detect cylindrical beams (pipes, round beams)
        """
        beams = []
        remaining = candidate_indices.copy()

        while len(remaining) > self.params["beam_min_points"]:
            cylinder, inliers = self.ransac_cylinder(
                remaining, axis_constraint="horizontal"
            )

            if cylinder is None or len(inliers) < self.params["beam_min_points"]:
                break

            # Validate cylinder as beam
            if (
                cylinder["radius"] < self.params["beam_max_width"] / 2
                and cylinder["length"] > self.params["beam_min_length"]
            ):
                beams.append(
                    {
                        "indices": inliers,
                        "type": "beam",
                        "beam_type": "cylinder",
                        "cylinder": cylinder,
                        "properties": self.compute_beam_properties(inliers),
                    }
                )
                remaining = np.setdiff1d(remaining, inliers)
            else:
                break

        return beams

    def ransac_cylinder(self, indices, axis_constraint="horizontal", iterations=500):
        """
        RANSAC cylinder fitting with axis constraint
        """
        if len(indices) < 5:  # Need at least 5 points for cylinder
            return None, []

        points = self.points[indices]
        best_cylinder = None
        best_inliers = []
        best_score = 0

        for _ in range(iterations):
            # For horizontal constraint, we can simplify:
            # Sample 2 points to define axis, then find radius
            sample_idx = np.random.choice(len(points), 5, replace=False)
            sample_points = points[sample_idx]

            # Estimate axis from first two points
            axis_direction = sample_points[1] - sample_points[0]
            if np.linalg.norm(axis_direction) < 0.01:
                continue
            axis_direction = axis_direction / np.linalg.norm(axis_direction)

            # Check horizontality
            if axis_constraint == "horizontal" and abs(axis_direction[2]) > 0.2:
                continue

            # Estimate radius using remaining points
            axis_point = sample_points[0]
            distances_to_axis = []

            for p in sample_points[2:]:
                vec_to_point = p - axis_point
                projection = np.dot(vec_to_point, axis_direction)
                perpendicular = vec_to_point - projection * axis_direction
                distances_to_axis.append(np.linalg.norm(perpendicular))

            radius = np.median(distances_to_axis)

            if radius > self.params["beam_max_width"] / 2:
                continue

            # Check all points
            all_distances = []
            projections = []

            for p in points:
                vec_to_point = p - axis_point
                projection = np.dot(vec_to_point, axis_direction)
                perpendicular = vec_to_point - projection * axis_direction
                distance = np.linalg.norm(perpendicular)
                all_distances.append(distance)
                projections.append(projection)

            all_distances = np.array(all_distances)
            projections = np.array(projections)

            # Find inliers (points within threshold of cylinder surface)
            inlier_mask = (
                np.abs(all_distances - radius) < self.params["ransac_threshold"]
            )
            inliers = indices[inlier_mask]

            if len(inliers) > self.params["beam_min_points"]:
                # Calculate cylinder length
                length = np.max(projections[inlier_mask]) - np.min(
                    projections[inlier_mask]
                )

                if length > self.params["beam_min_length"]:
                    score = len(inliers) * (1 + length / 10)

                    if score > best_score:
                        best_score = score
                        best_cylinder = {
                            "axis_point": axis_point,
                            "axis_direction": axis_direction,
                            "radius": radius,
                            "length": length,
                        }
                        best_inliers = inliers

        return best_cylinder, best_inliers

    # Merge and Validate Beams
    def merge_overlapping_beams(self, beams):
        """
        Merge beam detections that represent the same physical beam
        """
        if len(beams) <= 1:
            return beams

        merged = []
        used = set()

        for i, beam1 in enumerate(beams):
            if i in used:
                continue

            merged_beam = beam1.copy()
            merged_indices = set(beam1["indices"])

            for j, beam2 in enumerate(beams[i + 1 :], i + 1):
                if j in used:
                    continue

                # Check if beams overlap or are very close
                if self.beams_should_merge(beam1, beam2):
                    merged_indices.update(beam2["indices"])
                    used.add(j)

            merged_beam["indices"] = np.array(list(merged_indices))
            merged.append(merged_beam)
            used.add(i)

        return merged

    def beams_should_merge(self, beam1, beam2):
        """
        Determine if two beam detections should be merged
        """
        # Check spatial overlap
        indices1 = set(beam1["indices"])
        indices2 = set(beam2["indices"])

        overlap = len(indices1 & indices2)
        if overlap > 0:
            return True

        # Check if parallel and close
        points1 = self.points[beam1["indices"]]
        points2 = self.points[beam2["indices"]]

        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)

        distance = np.linalg.norm(centroid1 - centroid2)

        if distance < self.params["beam_merge_distance"]:
            # Check if aligned (similar direction)
            if "line" in beam1 and "line" in beam2:
                dir1 = beam1["line"]["direction"]
                dir2 = beam2["line"]["direction"]
                alignment = abs(np.dot(dir1, dir2))
                if alignment > 0.9:  # Nearly parallel
                    return True

        return False

    def compute_beam_properties(self, indices):
        """
        Compute geometric properties of a beam
        """
        points = self.points[indices]

        # Compute principal components
        center = np.mean(points, axis=0)
        centered = points - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Approximate dimensions (2 * std dev)
        dimensions = 2 * np.sqrt(eigenvalues)

        return {
            "center": center,
            "dimensions": dimensions,
            "principal_axes": eigenvectors,
            "length": dimensions[0],
            "width": dimensions[1],
            "height": dimensions[2],
            "elongation": dimensions[0] / max(dimensions[1], 0.01),
            "point_count": len(indices),
        }

    def validate_beam(self, properties):
        """
        Validate if detected structure is actually a beam
        """
        # Check dimensions
        if properties["length"] < self.params["beam_min_length"]:
            return False

        if properties["width"] > self.params["beam_max_width"]:
            return False

        if properties["height"] > self.params["beam_max_height"]:
            return False

        # Check elongation
        if properties["elongation"] < self.params["beam_elongation_ratio"]:
            return False

        # Check orientation (main axis should be horizontal)
        main_axis = properties["principal_axes"][:, 0]
        if abs(main_axis[2]) > 0.3:  # Too vertical
            return False

        return True

    def segment(self):
        """
        Main segmentation pipeline
        """
        print("Starting structural segmentation...")
        start_time = time.time()

        # 1. Detect floor and ceiling first
        self.segment_floor_ceiling()

        # 2. Detect walls
        self.segment_walls()

        # 3. Beams detection
        self.segment_beams()

        # 3. Post-processing
        # self.post_process_segments()

        print(f"Segmentation complete in {time.time() - start_time:.2f} seconds")
        print(f"Found: {len(self.segments)} structural elements")

        # Print summary
        element_counts = {}
        for name, data in self.segments.items():
            element_type = data["type"]
            element_counts[element_type] = element_counts.get(element_type, 0) + 1

        for element_type, count in element_counts.items():
            print(f"  {element_type}: {count}")

        print(f"Remaining unsegmented points: {len(self.remaining_indices)}")

        return self.segments

    def visualize_segments(self):
        """
        Visualize segmentation results
        """
        # Color map for different segment types
        colors = {
            "floor": [0.4, 0.2, 0.1],  # Blue
            "ceiling": [0.8, 0.8, 0.8],  # Green
            "wall": [0.6, 0.6, 0.5],  # Red
            "beam": [1, 1, 0],  # Yellow
            "unknown": [0.8, 0.2, 0.2],  # Gray
        }

        # Create colored point cloud
        point_colors = np.ones((len(self.points), 3)) * 0.5  # Default gray
        point_colors[:] = [0.8, 0.2, 0.2]

        for segment_name, segment_data in self.segments.items():
            segment_type = segment_data["type"]
            color = colors.get(segment_type, colors["unknown"])

            # # Add variation for multiple walls
            # if "wall" in segment_type:
            #     wall_num = (
            #         int(segment_name.split("_")[-1]) if "_" in segment_name else 0
            #     )
            #     color = plt.cm.hsv(wall_num * 0.1 % 1)[:3]

            point_colors[segment_data["indices"]] = color

        # Create Open3D point cloud for visualization
        vis_cloud = o3d.geometry.PointCloud()
        vis_cloud.points = o3d.utility.Vector3dVector(self.points)
        vis_cloud.colors = o3d.utility.Vector3dVector(point_colors)

        # Visualize
        o3d.visualization.draw_geometries(
            [vis_cloud], window_name="Structural Segmentation", width=1024, height=768
        )

        return vis_cloud
