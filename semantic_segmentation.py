# from ransac_numpy import StructuralRANSAC
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml


class SemanticSegmentor:
    def __init__(self, s3dis_validator, ransac_segmentor, pointnet_segmentor):
        # self.s3dis_validator = s3dis_validator
        self.config = self._read_config()
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
        self.ransac_segmentor.segment(self.config["region_growing"])

    def _read_config(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        return config

    def _create_pcd(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.points[:, 3:])
        return pcd

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
