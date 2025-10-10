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

        # self.original_cloud = point_cloud

        if downsample:
            self.cloud = point_cloud.voxel_down_sample(voxel_size)
        else:
            self.cloud = point_cloud

        self.cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        self.points = np.asarray(self.cloud.points)
        self.normals = np.asarray(self.cloud.normals)

        self.kdtree = KDTree(self.points)

        # Parameters - Basic
        self.params = {
            "ransac_threshold": 0.05,
            "ransac_iterations": 1000,
            "min_plane_points": 100,
            "angle_threshold_deg": 10,
            "wall_angle_tolerance": 40,
            "floor_ceiling_angle_tolerance": 10,
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
            if label == -1:  # skip noise
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

        for _ in range(self.params["ransac_iterations"]):
            # sample 3 points
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_idx]

            # salculate plane
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample_points[0])

            if normal_constraint is not None:
                angle = np.arccos(
                    np.clip(np.abs(np.dot(normal, normal_constraint)), -1, 1)
                )
                if np.degrees(angle) > self.params["angle_threshold_deg"]:
                    continue

            #  inliers
            distances = np.abs(np.dot(points, normal) + d)
            inlier_mask = distances < self.params["ransac_threshold"]
            inliers = indices[inlier_mask]

            # score
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

    # ============ MAIN PIPELINE ============

    def segment(self):
        """
        Main segmentation pipeline
        """
        print("Starting structural segmentation...")
        start_time = time.time()

        self.segment_floor_ceiling()
        self.segment_walls_improved()

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
            "floor": [0.4, 0.2, 0.1],  # dark brown
            "ceiling": [0.8, 0.8, 0.8],  # light gray
            "wall": [0.6, 0.6, 0.5],  # beige
            "window": [0.0, 0.6, 1.0],  # bright blue
            "door": [1.0, 0.6, 0.0],  # bright orange
            "table": [0.0, 0.8, 0.0],  #  bright green
            "chair": [0.0, 0.8, 0.0],  #  bright green
            "board": [0.2, 0.2, 0.2],  # dark
            "beam": [1, 1, 0],  # Yellow
            "column": [0.7, 0.0, 0.7],  # bright purple
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

        print("Detecting walls with validation...")

        room_height = self.get_room_height()
        floor_z = self.get_floor_height()
        ceiling_z = self.get_ceiling_height()

        wall_candidates = []
        furniture_candidates = []
        wall_count = 0

        # to find more potential wall planes
        for i in range(num_walls * 2):
            if len(self.remaining_indices) < self.params["min_plane_points"]:
                break

            # detect plane
            plane, inliers = self.ransac_plane(self.remaining_indices)

            if plane is None or len(inliers) < self.params["min_plane_points"]:
                break

            # check if plane is vertical
            normal = plane["normal"]
            verticality = 1.0 - abs(normal[2])

            if verticality < 0.8:
                continue

            temp_remaining = np.setdiff1d(self.remaining_indices, inliers)

            classification = self.classify_vertical_plane(
                plane, inliers, room_height, floor_z, ceiling_z
            )

            if classification == "wall":
                wall_candidates.append(
                    {"plane": plane, "indices": inliers, "verticality": verticality}
                )
            elif classification == "furniture":
                furniture_candidates.append({"plane": plane, "indices": inliers})

            self.remaining_indices = temp_remaining

        merged_walls = self.merge_similar_walls(wall_candidates)

        for i, wall_data in enumerate(merged_walls):
            # Refine with region growing
            refined_indices = self.region_grow_plane(
                wall_data["plane"],
                wall_data["indices"],
                growth_threshold=0.15,
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
            self.remaining_indices, np.array(used_indices)
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

        gap = np.maximum(0, np.maximum(bbox1_min - bbox2_max, bbox2_min - bbox1_max))
        max_gap = np.max(gap)

        if max_gap > 0.5:
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

            return np.percentile(self.points[:, 2], 95)

    def segment_walls_projection_based(self):
        """
        Alternative method: Project points to 2D and find wall lines
        """
        print("Detecting walls using 2D projection...")

        z_values = self.points[:, 2]
        floor_z = self.get_floor_height()
        ceiling_z = self.get_ceiling_height()

        mid_height = (floor_z + ceiling_z) / 2
        height_band = 0.5

        mask = (z_values > mid_height - height_band) & (
            z_values < mid_height + height_band
        )
        mid_points = self.points[mask]
        mid_indices = np.where(mask)[0]

        points_2d = mid_points[:, :2]

        wall_lines = self.detect_2d_lines(points_2d, mid_indices)

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

        wall_mask = predictions == 2
        beam_mask = predictions == 3
        window_mask = predictions == 5
        door_mask = predictions == 6
        board_mask = predictions == 11

        # refine walls
        if np.any(wall_mask):
            self.segments[wall_name]["indices"] = wall_indices[wall_mask]

        # add beam
        if sum(beam_mask) > 300:
            self.segments[f"beam_{wall_name}"] = {
                "indices": wall_indices[beam_mask],
                "type": "beam",
                "parent": wall_name,
            }
            print("added beam")

        # Add windows
        if sum(window_mask) > 300:
            window_indices = wall_indices[window_mask]
            # Cluster to separate multiple windows
            window_clusters = self.cluster_points(window_indices, eps=0.3)
            for i, cluster in enumerate(window_clusters):
                self.segments[f"window_{wall_name}_{i}"] = {
                    "indices": cluster,
                    "type": "window",
                    "parent": wall_name,
                }
                print("added window")

        # Add doors
        if sum(door_mask) > 300:
            door_indices = wall_indices[door_mask]
            door_clusters = self.cluster_points(door_indices, eps=0.3)
            for i, cluster in enumerate(door_clusters):
                self.segments[f"door_{wall_name}_{i}"] = {
                    "indices": cluster,
                    "type": "door",
                    "parent": wall_name,
                }
                print("added door")

        # add board
        if sum(board_mask):
            self.segments[f"board_{wall_name}"] = {
                "indices": wall_indices[board_mask],
                "type": "board",
                "parent": wall_name,
            }
            print("added board")

        return self.segments
