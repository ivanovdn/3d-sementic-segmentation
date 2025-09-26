import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


class StructuralRANSAC:
    def __init__(self, point_cloud, voxel_size=0.02, downsample=True):
        """
        Initialize with point cloud
        """
        # Downsample for efficiency
        # self.original_cloud = point_cloud

        if downsample:
            self.cloud = point_cloud.voxel_down_sample(voxel_size)
        else:
            self.cloud = point_cloud

        # Get points and normals
        self.cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        self.points = np.asarray(self.cloud.points)
        self.normals = np.asarray(self.cloud.normals)

        # Build KDTree for neighbor searches
        self.kdtree = KDTree(self.points)

        # Parameters - Basic
        self.params = {
            "ransac_threshold": 0.05,
            "ransac_iterations": 1000,
            "min_plane_points": 100,
            "angle_threshold_deg": 10,
            "wall_angle_tolerance": 10,
            "floor_ceiling_angle_tolerance": 10,
            # Beam-specific parameters (strict)
            "beam_min_length": 1.5,
            "beam_max_width": 0.4,
            "beam_max_height": 0.3,
            "beam_elongation_ratio": 4.0,
            "beam_ceiling_distance": 0.3,
            "beam_min_points": 100,
            "beam_horizontality_threshold": 0.95,
            "beam_min_density": 50,
            "beam_straightness_threshold": 0.05,
            "column_beam_angle_threshold": 80,
            "beam_merge_distance": 0.3,
        }
        self.params.update(
            {
                # Wall detection
                "wall_min_height": 1.5,  # Walls must be at least 1.5m tall
                "wall_min_width": 1.0,  # Walls must be at least 1m wide
                "wall_verticality_threshold": 0.95,  # How vertical (0-1)
                "wall_floor_connection_threshold": 0.3,  # Max distance from floor
                "wall_ceiling_connection_threshold": 0.3,  # Max distance from ceiling
                "wall_min_height_coverage": 0.6,  # Must cover 60% of room height
                "furniture_max_depth": 0.8,  # Furniture typically < 80cm deep
                "wall_merge_angle": 5,  # Degrees - merge nearly parallel walls
                "wall_merge_distance": 0.2,  # Distance to merge parallel walls
            }
        )

        # Results storage
        self.segments = {}
        self.remaining_indices = np.arange(len(self.points))

    # ============ CLUSTERING METHODS ============

    def cluster_points(self, indices, eps=0.2, min_samples=10):
        """
        DBSCAN clustering for point indices
        """
        if len(indices) == 0:
            return []

        points = self.points[indices]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

        clusters = []
        for label in set(clustering.labels_):
            if label == -1:  # Skip noise
                continue

            cluster_mask = clustering.labels_ == label
            cluster_indices = indices[cluster_mask]
            clusters.append(cluster_indices)

        return clusters

    def cluster_points_dbscan(self, indices, eps=0.1, min_samples=10):
        """
        DBSCAN clustering with custom parameters
        """
        return self.cluster_points(indices, eps=eps, min_samples=min_samples)

    # ============ PLANE DETECTION METHODS ============

    def ransac_plane(
        self, indices=None, normal_constraint=None, position_constraint=None
    ):
        """
        Core RANSAC plane fitting
        """
        if indices is None:
            indices = self.remaining_indices

        if len(indices) < self.params["min_plane_points"]:
            return None, []

        points = self.points[indices]

        best_plane = None
        best_inliers = []
        best_score = 0

        for iteration in range(self.params["ransac_iterations"]):
            # Sample 3 points
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_idx]

            # Calculate plane
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample_points[0])

            # Check constraints
            if normal_constraint is not None:
                angle = np.arccos(
                    np.clip(np.abs(np.dot(normal, normal_constraint)), -1, 1)
                )
                if np.degrees(angle) > self.params["angle_threshold_deg"]:
                    continue

            # Find inliers
            distances = np.abs(np.dot(points, normal) + d)
            inlier_mask = distances < self.params["ransac_threshold"]
            inliers = indices[inlier_mask]

            # Score
            score = len(inliers)

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
        Detect floor and ceiling
        """
        print("Detecting floor and ceiling...")

        horizontal_normal = np.array([0, 0, 1])
        z_values = self.points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        height_range = z_max - z_min

        # Floor
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

        # Ceiling
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
        Detect walls
        """
        print("Detecting walls...")
        wall_count = 0

        for i in range(num_walls):
            if len(self.remaining_indices) < self.params["min_plane_points"]:
                break

            wall_plane, wall_inliers = self.ransac_plane(self.remaining_indices)

            if (
                wall_plane is None
                or len(wall_inliers) < self.params["min_plane_points"]
            ):
                break

            # Check verticality
            normal = wall_plane["normal"]
            verticality = np.abs(normal[2])

            if verticality > np.sin(np.radians(self.params["wall_angle_tolerance"])):
                continue

            # Region growing
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
        Region growing for plane refinement
        """
        normal = plane["normal"]
        d = plane["d"]

        inlier_set = set(initial_inliers)
        boundary = set()

        for idx in initial_inliers:
            neighbors = self.kdtree.query_radius([self.points[idx]], r=0.1)[0]
            for n in neighbors:
                if n not in inlier_set and n in self.remaining_indices:
                    boundary.add(n)

        while boundary:
            current = boundary.pop()
            point = self.points[current]

            distance = np.abs(np.dot(point, normal) + d)

            if distance < growth_threshold:
                if self.normals is not None:
                    point_normal = self.normals[current]
                    angle = np.arccos(
                        np.clip(np.abs(np.dot(point_normal, normal)), -1, 1)
                    )
                    if np.degrees(angle) > self.params["angle_threshold_deg"]:
                        continue

                inlier_set.add(current)

                neighbors = self.kdtree.query_radius([point], r=0.1)[0]
                for n in neighbors:
                    if n not in inlier_set and n in self.remaining_indices:
                        boundary.add(n)

        return np.array(list(inlier_set))

    # ============ BEAM DETECTION METHODS ============

    def identify_vertical_structures(self):
        """
        Identify vertical structures to exclude from beam detection
        """
        vertical_indices = []

        clusters = self.cluster_points(self.remaining_indices, eps=0.3)

        for cluster in clusters:
            if len(cluster) < 50:
                continue

            points = self.points[cluster]
            centered = points - np.mean(points, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            main_axis = eigenvectors[:, np.argmax(eigenvalues)]
            verticality = abs(main_axis[2])

            if verticality > 0.8:
                vertical_indices.extend(cluster)

        return np.array(vertical_indices)

    def compute_local_density(self, indices, radius=0.1):
        """
        Compute local point density
        """
        densities = np.zeros(len(indices))
        points = self.points[indices]

        for i, point in enumerate(points):
            neighbors = self.kdtree.query_radius([point], r=radius)[0]
            densities[i] = len(neighbors) / (4 / 3 * np.pi * radius**3)

        return densities

    def get_beam_candidate_points(self, ceiling_z, exclude_indices=None):
        """
        Get candidate points for beam detection
        """
        candidates = self.remaining_indices.copy()

        if exclude_indices is not None and len(exclude_indices) > 0:
            candidates = np.setdiff1d(candidates, exclude_indices)

        if len(candidates) == 0:
            return np.array([])

        points = self.points[candidates]
        z_values = points[:, 2]
        distance_from_ceiling = ceiling_z - z_values

        height_mask = (distance_from_ceiling >= 0) & (
            distance_from_ceiling <= self.params["beam_ceiling_distance"]
        )

        candidates = candidates[height_mask]

        if len(candidates) < self.params["beam_min_points"]:
            return candidates

        if self.normals is not None:
            normals = self.normals[candidates]
            vertical_normal_mask = np.abs(normals[:, 2]) < 0.3
            candidates = candidates[vertical_normal_mask]

        if len(candidates) > 0:
            densities = self.compute_local_density(candidates, radius=0.15)
            density_threshold = np.percentile(densities, 40)
            candidates = candidates[densities > density_threshold]

        return candidates

    def segment_beams(self):
        """
        Main beam detection with robust validation
        """
        print("Detecting beams...")

        vertical_structures = self.identify_vertical_structures()

        ceiling_plane = self.segments.get("ceiling", {}).get("plane", None)
        if ceiling_plane is None:
            print("  Warning: No ceiling found, beam detection skipped")
            return

        ceiling_indices = self.segments["ceiling"]["indices"]
        ceiling_z = np.mean(self.points[ceiling_indices, 2])

        beam_candidates = self.get_beam_candidate_points(
            ceiling_z, exclude_indices=vertical_structures
        )

        if len(beam_candidates) < self.params["beam_min_points"]:
            print("  No valid beam candidates found")
            return

        beams = self.detect_validated_beams(beam_candidates, ceiling_plane)

        for i, beam in enumerate(beams):
            self.segments[f"beam_{i}"] = beam
            self.remaining_indices = np.setdiff1d(
                self.remaining_indices, beam["indices"]
            )
            print(
                f"  Beam {i} found: {len(beam['indices'])} points, type: {beam['beam_type']}"
            )

    def detect_validated_beams(self, candidate_indices, ceiling_plane):
        """
        Detect beams with validation
        """
        beams = []
        remaining = candidate_indices.copy()

        strategies = [
            ("linear", self.detect_linear_beams_validated),
            ("box", self.detect_box_beams_validated),
        ]

        for strategy_name, strategy_func in strategies:
            if len(remaining) < self.params["beam_min_points"]:
                break

            detected = strategy_func(remaining, ceiling_plane)

            for beam in detected:
                if self.validate_beam_strict(beam, ceiling_plane):
                    beams.append(beam)
                    remaining = np.setdiff1d(remaining, beam["indices"])

        return beams

    def detect_linear_beams_validated(self, candidate_indices, ceiling_plane):
        """
        Linear beam detection
        """
        beams = []
        remaining = candidate_indices.copy()

        max_attempts = 10
        attempts = 0

        while (
            len(remaining) > self.params["beam_min_points"] and attempts < max_attempts
        ):
            attempts += 1

            line, inliers = self.ransac_horizontal_line(remaining)

            if line is None or len(inliers) < self.params["beam_min_points"]:
                break

            beam_props = self.compute_beam_properties_detailed(inliers)

            beam_candidate = {
                "indices": inliers,
                "type": "beam",
                "beam_type": "linear",
                "line": line,
                "properties": beam_props,
            }

            if self.validate_beam_geometry(beam_candidate):
                beams.append(beam_candidate)
                remaining = np.setdiff1d(remaining, inliers)
            else:
                remove_count = max(len(inliers) // 4, 10)
                remaining = remaining[remove_count:]

        return beams

    def detect_box_beams_validated(self, candidate_indices, ceiling_plane):
        """
        Box beam detection using clustering
        """
        beams = []

        clusters = self.cluster_points_dbscan(
            candidate_indices, eps=0.15, min_samples=self.params["beam_min_points"] // 2
        )

        for cluster in clusters:
            if len(cluster) < self.params["beam_min_points"]:
                continue

            obb = self.fit_oriented_bounding_box(cluster)

            if obb is None:
                continue

            if not self.is_beam_shaped_obb_strict(obb):
                continue

            beam_candidate = {
                "indices": cluster,
                "type": "beam",
                "beam_type": "box",
                "obb": obb,
                "properties": self.compute_beam_properties_detailed(cluster),
            }

            if self.validate_beam_strict(beam_candidate, ceiling_plane):
                beams.append(beam_candidate)

        return beams

    def ransac_horizontal_line(self, indices, iterations=500):
        """
        RANSAC for horizontal lines
        """
        if len(indices) < 2:
            return None, []

        points = self.points[indices]
        best_line = None
        best_inliers = []
        best_score = 0

        for _ in range(iterations):
            sample_idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[sample_idx]

            if abs(p1[2] - p2[2]) > 0.1:
                continue

            direction = p2 - p1
            length = np.linalg.norm(direction)

            if length < 0.5:
                continue

            direction = direction / length

            horizontality = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
            if horizontality < self.params["beam_horizontality_threshold"]:
                continue

            distances = self.point_to_line_distances(points, p1, direction)
            inlier_mask = distances < self.params["ransac_threshold"]
            inliers = indices[inlier_mask]

            if len(inliers) < self.params["beam_min_points"]:
                continue

            inlier_points = points[inlier_mask]
            straightness = self.compute_straightness(inlier_points, p1, direction)

            if straightness > self.params["beam_straightness_threshold"]:
                continue

            line_length = self.compute_line_extent(inlier_points, p1, direction)
            density = len(inliers) / max(line_length, 0.1)

            if density < self.params["beam_min_density"]:
                continue

            score = len(inliers) * (1 + line_length / 10) * horizontality

            if score > best_score:
                best_score = score
                best_line = {"point": p1, "direction": direction}
                best_inliers = inliers

        return best_line, best_inliers

    def point_to_line_distances(self, points, line_point, line_direction):
        """
        Compute point-to-line distances
        """
        vectors_to_point = points - line_point
        projections = np.dot(vectors_to_point, line_direction)
        perpendicular_vectors = (
            vectors_to_point - projections[:, np.newaxis] * line_direction
        )
        distances = np.linalg.norm(perpendicular_vectors, axis=1)
        return distances

    def compute_straightness(self, points, line_point, line_direction):
        """
        Measure straightness of points
        """
        vectors = points - line_point
        projections = np.dot(vectors, line_direction)

        sorted_idx = np.argsort(projections)
        sorted_points = points[sorted_idx]

        deviations = []
        for i in range(1, len(sorted_points) - 1):
            expected_direction = sorted_points[i + 1] - sorted_points[i - 1]
            if np.linalg.norm(expected_direction) > 0:
                expected_direction = expected_direction / np.linalg.norm(
                    expected_direction
                )

                actual_direction = sorted_points[i] - sorted_points[i - 1]
                if np.linalg.norm(actual_direction) > 0:
                    actual_direction = actual_direction / np.linalg.norm(
                        actual_direction
                    )
                    deviation = 1 - abs(np.dot(expected_direction, actual_direction))
                    deviations.append(deviation)

        return np.mean(deviations) if deviations else 0

    def compute_line_extent(self, points, line_point, line_direction):
        """
        Compute extent along line
        """
        vectors = points - line_point
        projections = np.dot(vectors, line_direction)
        return np.max(projections) - np.min(projections)

    def compute_beam_properties_detailed(self, indices):
        """
        Compute detailed beam properties
        """
        points = self.points[indices]

        center = np.mean(points, axis=0)
        centered = points - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        dimensions = 2 * np.sqrt(eigenvalues)

        # Compactness
        try:
            hull = ConvexHull(points)
            hull_volume = hull.volume
            bbox_volume = np.prod(dimensions)
            compactness = hull_volume / max(bbox_volume, 0.001)
        except:
            compactness = 0.5

        # Straightness
        straightness = 0
        if len(points) > 2:
            main_axis = eigenvectors[:, 0]
            projections = np.dot(centered, main_axis)
            sorted_idx = np.argsort(projections)
            sorted_points = centered[sorted_idx]

            if len(sorted_points) > 2:
                line_vec = sorted_points[-1] - sorted_points[0]
                if np.linalg.norm(line_vec) > 0:
                    line_vec = line_vec / np.linalg.norm(line_vec)

                    deviations = []
                    for point in sorted_points[1:-1]:
                        vec_to_point = point - sorted_points[0]
                        projection = np.dot(vec_to_point, line_vec)
                        perpendicular = vec_to_point - projection * line_vec
                        deviation = np.linalg.norm(perpendicular)
                        deviations.append(deviation)

                    straightness = np.mean(deviations) if deviations else 0

        return {
            "center": center,
            "dimensions": dimensions,
            "principal_axes": eigenvectors,
            "length": dimensions[0],
            "width": dimensions[1],
            "height": dimensions[2],
            "elongation": dimensions[0] / max(dimensions[1], 0.01),
            "point_count": len(indices),
            "compactness": compactness,
            "straightness": straightness,
            "eigenvalues": eigenvalues,
        }

    def fit_oriented_bounding_box(self, indices):
        """
        Fit OBB using PCA
        """
        if len(indices) < self.params["beam_min_points"]:
            return None

        points = self.points[indices]
        center = np.mean(points, axis=0)
        centered_points = points - center

        pca = PCA(n_components=3)
        pca.fit(centered_points)

        transformed = pca.transform(centered_points)

        min_bounds = np.min(transformed, axis=0)
        max_bounds = np.max(transformed, axis=0)
        dimensions = max_bounds - min_bounds

        obb = {
            "center": center,
            "axes": pca.components_,
            "dimensions": dimensions,
            "volume": np.prod(dimensions),
        }

        return obb

    def is_beam_shaped_obb_strict(self, obb):
        """
        Check if OBB is beam-shaped
        """
        dims = sorted(obb["dimensions"])

        if dims[2] < self.params["beam_min_length"]:
            return False

        elongation = dims[2] / dims[1]
        if elongation < self.params["beam_elongation_ratio"]:
            return False

        if (
            dims[0] > self.params["beam_max_height"]
            or dims[1] > self.params["beam_max_width"]
        ):
            return False

        longest_axis_idx = np.argmax(obb["dimensions"])
        longest_axis = obb["axes"][longest_axis_idx]

        horizontality = np.sqrt(longest_axis[0] ** 2 + longest_axis[1] ** 2)
        if horizontality < self.params["beam_horizontality_threshold"]:
            return False

        return True

    def validate_beam_geometry(self, beam_candidate):
        """
        Validate beam geometry
        """
        props = beam_candidate["properties"]

        if props["length"] < self.params["beam_min_length"]:
            return False

        if props["width"] > self.params["beam_max_width"]:
            return False

        if props["height"] > self.params["beam_max_height"]:
            return False

        if props["elongation"] < self.params["beam_elongation_ratio"]:
            return False

        main_axis = props["principal_axes"][:, 0]
        horizontality = np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2)

        if horizontality < self.params["beam_horizontality_threshold"]:
            return False

        if props["compactness"] < 0.5:
            return False

        if props.get("straightness", 1.0) > self.params["beam_straightness_threshold"]:
            return False

        return True

    def validate_beam_strict(self, beam, ceiling_plane):
        """
        Strict beam validation
        """
        if not self.validate_beam_geometry(beam):
            return False

        points = self.points[beam["indices"]]

        ceiling_normal = ceiling_plane["normal"]
        ceiling_d = ceiling_plane["d"]

        distances_to_ceiling = np.abs(np.dot(points, ceiling_normal) + ceiling_d)
        mean_distance = np.mean(distances_to_ceiling)

        if mean_distance > self.params["beam_ceiling_distance"]:
            return False

        for wall_name, wall_data in self.segments.items():
            if "wall" not in wall_name:
                continue

            wall_plane = wall_data["plane"]
            wall_normal = wall_plane["normal"]

            if "line" in beam:
                beam_direction = beam["line"]["direction"]
                alignment = abs(np.dot(beam_direction, wall_normal))
                if alignment > 0.8:
                    return False

        props = beam["properties"]
        cross_section_ratio = props["width"] / max(props["height"], 0.01)

        if cross_section_ratio > 5 or cross_section_ratio < 0.2:
            return False

        return True

    # ============ MAIN PIPELINE ============

    def segment(self):
        """
        Main segmentation pipeline
        """
        print("Starting structural segmentation...")
        start_time = time.time()

        self.segment_floor_ceiling()
        self.segment_walls_improved()
        self.segment_beams()

        print(f"Segmentation complete in {time.time() - start_time:.2f} seconds")
        print(f"Found: {len(self.segments)} structural elements")

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

    def segment_walls_improved(self, num_walls=15):
        """
        Enhanced wall detection that distinguishes walls from furniture
        """
        print("Detecting walls with validation...")

        # Get room dimensions for context
        room_height = self.get_room_height()
        floor_z = self.get_floor_height()
        ceiling_z = self.get_ceiling_height()

        wall_candidates = []
        furniture_candidates = []
        wall_count = 0

        # Try to find more potential wall planes
        for i in range(num_walls * 2):  # Try more iterations
            if len(self.remaining_indices) < self.params["min_plane_points"]:
                break

            # Find a plane
            plane, inliers = self.ransac_plane(self.remaining_indices)

            if plane is None or len(inliers) < self.params["min_plane_points"]:
                break

            # Check if plane is vertical
            normal = plane["normal"]
            verticality = 1.0 - abs(normal[2])  # Should be close to 1 for walls

            if verticality < 0.8:  # Not vertical enough
                continue

            # Temporarily remove these points to find more planes
            temp_remaining = np.setdiff1d(self.remaining_indices, inliers)

            # Classify as wall or furniture
            classification = self.classify_vertical_plane(
                plane, inliers, room_height, floor_z, ceiling_z
            )

            if classification == "wall":
                wall_candidates.append(
                    {"plane": plane, "indices": inliers, "verticality": verticality}
                )
            elif classification == "furniture":
                furniture_candidates.append({"plane": plane, "indices": inliers})

            # Update remaining indices
            self.remaining_indices = temp_remaining

        # Merge similar walls
        merged_walls = self.merge_similar_walls(wall_candidates)

        # Add walls to segments
        for i, wall_data in enumerate(merged_walls):
            # Refine with region growing
            refined_indices = self.region_grow_plane(
                wall_data["plane"],
                wall_data["indices"],
                growth_threshold=0.15,  # Slightly larger for walls with gaps
            )

            self.segments[f"wall_{i}"] = {
                "plane": wall_data["plane"],
                "indices": refined_indices,
                "type": "wall",
            }
            wall_count += 1
            print(f"  Wall {i} found: {len(refined_indices)} points")

        # Optionally add furniture as separate category
        for i, furniture_data in enumerate(furniture_candidates):
            self.segments[f"furniture_{i}"] = {
                "plane": furniture_data["plane"],
                "indices": furniture_data["indices"],
                "type": "furniture",
            }

        print(
            f"  Total: {wall_count} walls, {len(furniture_candidates)} furniture pieces"
        )

        # Restore remaining indices (excluding confirmed walls)
        used_indices = []
        for segment_name, segment_data in self.segments.items():
            if "wall" in segment_name:
                used_indices.extend(segment_data["indices"])
        self.remaining_indices = np.setdiff1d(
            np.arange(len(self.points)), np.array(used_indices)
        )

    def classify_vertical_plane(self, plane, indices, room_height, floor_z, ceiling_z):
        """
        Classify vertical plane as wall or furniture based on geometric properties
        """
        points = self.points[indices]

        # 1. Check height span
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        height_span = z_max - z_min

        # 2. Check connections to floor and ceiling
        distance_to_floor = z_min - floor_z
        distance_to_ceiling = ceiling_z - z_max

        # 3. Check depth (thickness in direction of normal)
        normal = plane["normal"]
        # Project points onto normal direction
        center = np.mean(points, axis=0)
        centered = points - center
        projections = np.dot(centered, normal)
        depth = np.max(projections) - np.min(projections)

        # 4. Check width (extent perpendicular to normal and vertical)
        horizontal_dir = np.array([normal[1], -normal[0], 0])
        if np.linalg.norm(horizontal_dir) > 0:
            horizontal_dir = horizontal_dir / np.linalg.norm(horizontal_dir)
            width_projections = np.dot(centered, horizontal_dir)
            width = np.max(width_projections) - np.min(width_projections)
        else:
            width = 0

        # 5. Height coverage ratio
        height_coverage = height_span / room_height

        # Classification rules
        is_wall = True

        # Walls should span most of room height
        if height_span < self.params["wall_min_height"]:
            is_wall = False

        # Walls should be near floor and ceiling
        if (
            distance_to_floor > self.params["wall_floor_connection_threshold"]
            and distance_to_ceiling > self.params["wall_ceiling_connection_threshold"]
        ):
            is_wall = False

        # Walls should cover significant portion of room height
        if height_coverage < self.params["wall_min_height_coverage"]:
            is_wall = False

        # Furniture is typically thicker (depth) and shorter
        if depth > self.params["furniture_max_depth"] and height_span < 2.0:
            is_wall = False

        # Walls should have minimum width
        if width < self.params["wall_min_width"] and height_span < 2.0:
            is_wall = False

        return "wall" if is_wall else "furniture"

    def merge_similar_walls(self, wall_candidates):
        """
        Merge wall segments that are likely the same wall
        (handles walls fragmented by windows, doors, etc.)
        """
        if len(wall_candidates) <= 1:
            return wall_candidates

        merged = []
        used = set()

        for i, wall1 in enumerate(wall_candidates):
            if i in used:
                continue

            merged_wall = wall1.copy()
            merged_indices = set(wall1["indices"])

            for j, wall2 in enumerate(wall_candidates[i + 1 :], i + 1):
                if j in used:
                    continue

                # Check if walls should merge
                if self.should_merge_walls(wall1, wall2):
                    merged_indices.update(wall2["indices"])
                    used.add(j)

            merged_wall["indices"] = np.array(list(merged_indices))
            merged.append(merged_wall)
            used.add(i)

        return merged

    def should_merge_walls(self, wall1, wall2):
        """
        Check if two wall candidates should be merged
        """
        normal1 = wall1["plane"]["normal"]
        normal2 = wall2["plane"]["normal"]

        # Check if normals are parallel
        angle = np.arccos(np.clip(abs(np.dot(normal1, normal2)), -1, 1))
        if np.degrees(angle) > self.params["wall_merge_angle"]:
            return False

        # Check if walls are coplanar (close in space)
        points1 = self.points[wall1["indices"]]
        points2 = self.points[wall2["indices"]]

        center1 = np.mean(points1, axis=0)
        center2 = np.mean(points2, axis=0)

        # Distance from center2 to plane1
        d1 = wall1["plane"]["d"]
        distance_to_plane = abs(np.dot(center2, normal1) + d1)

        if distance_to_plane > self.params["wall_merge_distance"]:
            return False

        # Check if walls are adjacent (not too far apart)
        min_distance = np.min(
            [
                np.min(
                    np.linalg.norm(
                        points1[:, np.newaxis, :] - points2[np.newaxis, :, :], axis=2
                    )
                )
                for _ in [0]  # Trick to compute min distance efficiently
            ]
        )

        # Simplified: check distance between bounding boxes
        bbox1_min = np.min(points1, axis=0)
        bbox1_max = np.max(points1, axis=0)
        bbox2_min = np.min(points2, axis=0)
        bbox2_max = np.max(points2, axis=0)

        # Check if bounding boxes are close
        gap = np.maximum(0, np.maximum(bbox1_min - bbox2_max, bbox2_min - bbox1_max))
        max_gap = np.max(gap)

        if max_gap > 0.5:  # More than 50cm gap
            return False

        return True

    def get_room_height(self):
        """Get room height from floor to ceiling"""
        floor_z = self.get_floor_height()
        ceiling_z = self.get_ceiling_height()
        return ceiling_z - floor_z

    def get_floor_height(self):
        """Get floor height"""
        if "floor" in self.segments:
            floor_points = self.points[self.segments["floor"]["indices"]]
            return np.mean(floor_points[:, 2])
        else:
            # Estimate from lowest points
            return np.percentile(self.points[:, 2], 5)

    def get_ceiling_height(self):
        """Get ceiling height"""
        if "ceiling" in self.segments:
            ceiling_points = self.points[self.segments["ceiling"]["indices"]]
            return np.mean(ceiling_points[:, 2])
        else:
            # Estimate from highest points
            return np.percentile(self.points[:, 2], 95)

    def segment_walls_projection_based(self):
        """
        Alternative method: Project points to 2D and find wall lines
        """
        print("Detecting walls using 2D projection...")

        # Get points between floor and ceiling
        z_values = self.points[:, 2]
        floor_z = self.get_floor_height()
        ceiling_z = self.get_ceiling_height()

        # Get points in middle height (avoid floor/ceiling)
        mid_height = (floor_z + ceiling_z) / 2
        height_band = 0.5  # Look at 1m band around middle

        mask = (z_values > mid_height - height_band) & (
            z_values < mid_height + height_band
        )
        mid_points = self.points[mask]
        mid_indices = np.where(mask)[0]

        # Project to 2D (top view)
        points_2d = mid_points[:, :2]

        # Find line segments using RANSAC in 2D
        wall_lines = self.detect_2d_lines(points_2d, mid_indices)

        # Extend lines to full height
        walls = []
        for line in wall_lines:
            wall_indices = self.extend_wall_to_full_height(line)
            walls.append(wall_indices)

        return walls

    def detect_2d_lines(self, points_2d, indices):
        """
        Detect lines in 2D projection
        """
        from sklearn.linear_model import RANSACRegressor

        lines = []
        remaining = indices.copy()

        while len(remaining) > 100:
            # Fit line using RANSAC
            X = points_2d[remaining][:, 0].reshape(-1, 1)
            y = points_2d[remaining][:, 1]

            ransac = RANSACRegressor(residual_threshold=0.1)
            ransac.fit(X, y)

            inlier_mask = ransac.inlier_mask_
            if np.sum(inlier_mask) < 100:
                break

            lines.append({"indices": remaining[inlier_mask], "model": ransac})

            remaining = remaining[~inlier_mask]

        return lines

    def refine_wall_segment(self, wall_name, wall_indices, predictions):
        """
        Refine a single wall segment based on predictions
        """
        refined = {}

        # Separate by prediction
        wall_mask = predictions == 2
        window_mask = predictions == 5
        door_mask = predictions == 6

        # Keep wall points
        if np.any(wall_mask):
            self.segments[wall_name]["indices"] = wall_indices[wall_mask]

        # Add windows
        if np.any(window_mask):
            window_indices = wall_indices[window_mask]
            # Cluster to separate multiple windows
            window_clusters = self.cluster_points(window_indices, eps=0.3)
            for i, cluster in enumerate(window_clusters):
                self.segments[f"window_{i}"] = {
                    "indices": cluster,
                    "type": "window",
                    "parent": wall_name,
                }

        # Add doors
        if np.any(door_mask):
            door_indices = wall_indices[door_mask]
            door_clusters = self.cluster_points(door_indices, eps=0.3)
            for i, cluster in enumerate(door_clusters):
                self.segments[f"door_{i}"] = {
                    "indices": cluster,
                    "type": "door",
                    "parent": wall_name,
                }

        return self.segments
