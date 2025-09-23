import os

import numpy as np
import open3d as o3d
from sklearn.metrics import classification_report, confusion_matrix


class S3DISValidator:
    def __init__(self, s3dis_path):
        self.class_map = {
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
        self.structural_classes = [0, 1, 2, 3, 4]  # ceiling, floor, wall, beam, column
        self.s3dis_path = s3dis_path

    def load_s3dis_room(self, area, room):
        """Load S3DIS room with ground truth labels"""
        room_path = f"{self.s3dis_path}/Area_{area}/{room}"
        print(room_path)
        points = []
        labels = []

        # Read each annotation file
        anno_path = os.path.join(room_path, "Annotations")
        for file in os.listdir(anno_path):
            if file.endswith(".txt"):
                class_name = file.split("_")[0]
                if class_name in self.class_map:
                    class_id = self.class_map[class_name]
                    data = np.loadtxt(os.path.join(anno_path, file))
                    xyz = data[:, :3]
                    rgb = data[:, 3:6] / 255.0
                    points.append(np.hstack([xyz, rgb]))
                    labels.append(np.full(len(data), class_id))

        return np.vstack(points), np.hstack(labels)
