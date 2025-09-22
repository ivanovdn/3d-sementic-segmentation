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

        for iteration in range(self.params["ransac_iterations"]):
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
            wall_inliers = self.region_grow_plane(wall_plane, wall_inliers)

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

        # 3. Post-processing
        # self.post_process_segments()

        print(f"Segmentation complete in {time.time() - start_time:.2f} seconds")
        print(f"Found: {len(self.segments)} structural elements")
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
