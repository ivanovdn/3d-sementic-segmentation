# from ransac_numpy import StructuralRANSAC
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml


class SemanticSegmentor:
    def __init__(self, s3dis_validator, ransac_segmentor, pointnet_segmentor, config):
        # self.s3dis_validator = s3dis_validator
        # self.config = self._read_config()
        self.config = config
        self.classes = {
            "ceiling": 0,
            "floor": 1,
            "wall": 2,
            "beam": 3,
            "column": 4,
            "window": 5,
            "door": 6,
            "table": 7,
            "chair": 8,
            "sofa": 9,
            "bookcase": 10,
            "board": 11,
            "clutter": 12,
        }

        if config["dataset"] == "s3dis":
            self.points, self.labels = s3dis_validator.load_and_process_room(
                self.config["area"],
                self.config["room"],
                self.config["subsample_method"],
                self.config["subsample_ratio"],
            )

            self.pcd = self._create_pcd()
        else:
            self.points, self.pcd = s3dis_validator.read_pcd_and_extract_points(
                self.config["subsample_ratio"]
            )

        # Segmentors
        self.pointnet_segmentor = pointnet_segmentor()
        self.ransac_segmentor = ransac_segmentor(self.pcd)

    def _read_config(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        return config

    def _create_pcd(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.points[:, 3:])
        return pcd

    def ransac_segment(self):
        self.ransac_segmentor.segment(
            self.config["ransac_with_walls"],
            self.config["segment_walls_improved"],
            self.config["region_growing"],
        )

    def map_ransac_to_s3dis(
        self,
    ):
        """
        Map RANSAC segments to S3DIS label format
        """
        point_labels = np.full(len(self.labels), -1)  # -1 for unlabeled

        for segment_type, data in self.ransac_segmentor.segments.items():
            # Parse segment type (e.g., 'wall_1' -> 'wall')
            base_type = segment_type.split("_")[0]

            if base_type in self.classes:
                point_labels[data["indices"]] = self.classes[base_type]
        return point_labels

    def pointnet_segmentation(self, points, type):
        data_blocks, block_indices = self.pointnet_segmentor.create_pointnet_blocks(
            points, self.config["block_size"], self.config["stride"]
        )
        all_predictions, all_confidences = self.pointnet_segmentor.inference(
            data_blocks, type
        )
        predictions = self.pointnet_segmentor.reconstruct_furniture_labels(
            points, all_predictions, all_confidences, block_indices, type
        )
        return predictions

    def refind_walls(self):
        refined_walls = {}
        for segment in self.ransac_segmentor.segments:
            if segment.startswith("wall"):
                idx = self.ransac_segmentor.segments[segment]["indices"]
                wall_points = self.points[idx]
                predictions = self.pointnet_segmentation(wall_points, "wall")
                refined = self.ransac_segmentor.refine_wall_segment(
                    segment, idx, predictions
                )
                refined_walls.update(refined)
        self.ransac_segmentor.segments.update(refined_walls)

    def refined_with_pointnet(self):
        rest_points = self.points[self.ransac_segmentor.remaining_indices]
        predictions = self.pointnet_segmentation(rest_points, "all")
        pointnet_refined = {}
        refined_indices = []
        for class_name, class_id in self.classes.items():
            class_mask = predictions == class_id
            if sum(class_mask) > 300 and class_id not in (0, 1):
                idx = self.ransac_segmentor.remaining_indices[class_mask]
                pointnet_refined[f"{class_name}_pointnet"] = {
                    "indices": idx,
                    "type": class_name,
                    "parent": "rest",
                }
                refined_indices.extend(idx)
        self.ransac_segmentor.segments.update(pointnet_refined)
        self.ransac_segmentor.remaining_indices = np.setdiff1d(
            self.ransac_segmentor.remaining_indices, refined_indices
        )

    def visualize_pcd(self):
        o3d.visualization.draw_geometries([self.pcd])

    def visualize_segments(self, predictions):
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

        for class_name, class_id in self.classes.items():
            if class_name in self.config["elements"]:
                mask = predictions == class_id
                color = colors.get(class_name, colors["unknown"])
                point_colors[mask] = color

        # Create Open3D point cloud for visualization
        vis_cloud = o3d.geometry.PointCloud()
        vis_cloud.points = o3d.utility.Vector3dVector(self.points[:, :3])
        vis_cloud.colors = o3d.utility.Vector3dVector(point_colors)

        # Visualize
        o3d.visualization.draw_geometries(
            [vis_cloud], window_name="Structural Segmentation", width=1024, height=768
        )

        return vis_cloud
