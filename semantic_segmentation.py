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

        self.points, self.labels = s3dis_validator.load_and_process_room(
            self.config["area"], self.config["room"], self.config["subsample_method"]
        )

        self.pcd = self._create_pcd()
        self.pointnet_segmentor = pointnet_segmentor()
        self.ransac_segmentor = ransac_segmentor(self.pcd, downsample=False)

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
