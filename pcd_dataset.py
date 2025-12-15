import os

import numpy as np
import open3d as o3d


class PointCloudDataset:
    def __init__(self, file_path, voxel_size):
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

        self.file_path = file_path
        self.voxel_size = voxel_size
        self.pcd = self.read_pcd()

    def load_s3dis_room_full(self, area, room):
        """Load S3DIS room with ground truth labels"""
        room_path = f"{self.file_path}/Area_{area}/{room}"
        print(room_path)
        points = []
        labels = []

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

    def load_s3dis_room_uniform(self, area, room, subsample_rate=3):
        """
        Load S3DIS room with ground truth labels, subsampled for efficiency

        Args:
            area: Area number (1-6)
            room: Room name
            subsample_rate: Keep every Nth point (e.g., 3 = keep every 3rd point)
        """
        room_path = f"{self.file_path}/Area_{area}/{room}"
        print(f"Loading {room_path} with subsampling rate 1/{subsample_rate}")

        points = []
        labels = []
        total_points_original = 0
        total_points_subsampled = 0

        anno_path = os.path.join(room_path, "Annotations")

        for file in os.listdir(anno_path):
            if file.endswith(".txt"):
                class_name = file.split("_")[0]
                if class_name in self.class_map:
                    class_id = self.class_map[class_name]
                    data = np.loadtxt(os.path.join(anno_path, file))

                    total_points_original += len(data)

                    subsampled_indices = np.arange(0, len(data), subsample_rate)
                    data_subsampled = data[subsampled_indices]

                    total_points_subsampled += len(data_subsampled)

                    if len(data_subsampled) > 0:
                        xyz = data_subsampled[:, :3]
                        rgb = data_subsampled[:, 3:6] / 255.0
                        points.append(np.hstack([xyz, rgb]))
                        labels.append(np.full(len(data_subsampled), class_id))

        print(f"  Original points: {total_points_original:,}")
        print(
            f"  Subsampled points: {total_points_subsampled:,} "
            f"({100*total_points_subsampled/total_points_original:.1f}%)"
        )

        return np.vstack(points), np.hstack(labels)

    def load_s3dis_room_random(self, area, room, subsample_ratio=0.3, seed=42):

        room_path = f"{self.file_path}/Area_{area}/{room}"
        print(f"Loading {room_path} with {subsample_ratio*100:.0f}% subsampling")

        np.random.seed(seed)

        points = []
        labels = []
        total_points_original = 0
        total_points_subsampled = 0

        anno_path = os.path.join(room_path, "Annotations")

        for file in os.listdir(anno_path):
            if file.endswith(".txt"):
                class_name = file.split("_")[0]
                if class_name in self.class_map:
                    class_id = self.class_map[class_name]
                    data = np.loadtxt(os.path.join(anno_path, file))

                    total_points_original += len(data)

                    # Random subsampling
                    n_points = len(data)
                    n_sample = int(n_points * subsample_ratio)

                    if n_sample > 0:
                        # Random selection without replacement
                        sample_indices = np.random.choice(
                            n_points, n_sample, replace=False
                        )
                        data_subsampled = data[sample_indices]

                        total_points_subsampled += len(data_subsampled)

                        xyz = data_subsampled[:, :3]
                        rgb = data_subsampled[:, 3:6] / 255.0
                        points.append(np.hstack([xyz, rgb]))
                        labels.append(np.full(len(data_subsampled), class_id))

        print(f"  Original points: {total_points_original:,}")
        print(
            f"  Subsampled points: {total_points_subsampled:,} "
            f"({100*total_points_subsampled/total_points_original:.1f}%)"
        )

        if len(points) > 0:
            return np.vstack(points), np.hstack(labels)
        else:
            return np.array([]), np.array([])

    def load_s3dis_room_voxel(self, area, room, voxel_size):

        room_path = f"{self.file_path}/Area_{area}/{room}"
        print(f"Loading {room_path} with voxel size {voxel_size}m")

        # First load all data
        all_points = []
        all_labels = []
        all_colors = []

        anno_path = os.path.join(room_path, "Annotations")

        for file in os.listdir(anno_path):
            if file.endswith(".txt"):
                class_name = file.split("_")[0]
                if class_name in self.class_map:
                    class_id = self.class_map[class_name]
                    data = np.loadtxt(os.path.join(anno_path, file))

                    if len(data) > 0:
                        all_points.append(data[:, :3])
                        all_colors.append(data[:, 3:6])
                        all_labels.extend([class_id] * len(data))

        if len(all_points) == 0:
            return np.array([]), np.array([])

        # Combine all data
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        all_labels = np.array(all_labels)

        print(f"  Original points: {len(all_points):,}")

        # Voxel-based subsampling
        subsampled_indices = self.voxel_subsample_indices(all_points, voxel_size)

        # Apply subsampling
        points_subsampled = all_points[subsampled_indices]
        colors_subsampled = all_colors[subsampled_indices] / 255.0
        labels_subsampled = all_labels[subsampled_indices]

        print(
            f"  Subsampled points: {len(points_subsampled):,} "
            f"({100*len(points_subsampled)/len(all_points):.1f}%)"
        )

        # Combine XYZ and RGB
        points_with_color = np.hstack([points_subsampled, colors_subsampled])

        return points_with_color, labels_subsampled

    def voxel_subsample_indices(self, points, voxel_size):

        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # Create unique voxel keys
        voxel_keys = (
            voxel_indices[:, 0] * 1000000
            + voxel_indices[:, 1] * 1000
            + voxel_indices[:, 2]
        )

        # Find unique voxels and representative points
        selected_indices = []
        unique_voxels = {}

        for i, key in enumerate(voxel_keys):
            if key not in unique_voxels:
                unique_voxels[key] = []
            unique_voxels[key].append(i)

        # For each voxel, select the point closest to voxel center
        for key, indices in unique_voxels.items():
            if len(indices) == 1:
                selected_indices.append(indices[0])
            else:
                # Get voxel center
                voxel_points = points[indices]
                voxel_center = np.mean(voxel_points, axis=0)

                # Find closest point to center
                distances = np.linalg.norm(voxel_points - voxel_center, axis=1)
                closest_idx = indices[np.argmin(distances)]
                selected_indices.append(closest_idx)

        return np.array(selected_indices)

    def load_s3dis_room_adaptive(self, area, room, subsample_rates=None):

        if subsample_rates is None:
            # Default: subsample large structures more, keep small objects
            subsample_rates = {
                "ceiling": 0.2,  # Heavily subsample (large, flat)
                "floor": 0.2,  # Heavily subsample (large, flat)
                "wall": 0.3,  # Moderate subsample
                "beam": 0.5,  # Keep more (smaller, important)
                "column": 0.5,  # Keep more (smaller, important)
                "window": 0.6,  # Keep more (features)
                "door": 0.6,  # Keep more (features)
                "table": 0.4,  # Moderate
                "chair": 0.4,  # Moderate
                "sofa": 0.4,  # Moderate
                "bookcase": 0.4,  # Moderate
                "board": 0.5,  # Keep more
                "clutter": 0.3,  # Moderate subsample
            }

        room_path = f"{self.file_path}/Area_{area}/{room}"
        print(f"Loading {room_path} with adaptive subsampling")

        points = []
        labels = []
        total_points_original = 0
        total_points_subsampled = 0

        anno_path = os.path.join(room_path, "Annotations")

        for file in os.listdir(anno_path):
            if file.endswith(".txt"):
                class_name = file.split("_")[0]
                if class_name in self.class_map:
                    class_id = self.class_map[class_name]
                    data = np.loadtxt(os.path.join(anno_path, file))

                    total_points_original += len(data)

                    # Get subsample rate for this class
                    subsample_ratio = subsample_rates.get(class_name, 0.3)
                    n_points = len(data)
                    n_sample = max(1, int(n_points * subsample_ratio))

                    # Random sampling
                    if n_sample < n_points:
                        sample_indices = np.random.choice(
                            n_points, n_sample, replace=False
                        )
                        data_subsampled = data[sample_indices]
                    else:
                        data_subsampled = data

                    total_points_subsampled += len(data_subsampled)

                    if len(data_subsampled) > 0:
                        xyz = data_subsampled[:, :3]
                        rgb = data_subsampled[:, 3:6] / 255.0
                        points.append(np.hstack([xyz, rgb]))
                        labels.append(np.full(len(data_subsampled), class_id))

                    print(
                        f"    {class_name}: {len(data):,} → {len(data_subsampled):,} "
                        f"({subsample_ratio*100:.0f}%)"
                    )

        print(
            f"  Total: {total_points_original:,} → {total_points_subsampled:,} "
            f"({100*total_points_subsampled/total_points_original:.1f}%)"
        )

        return np.vstack(points), np.hstack(labels)

    def load_and_process_room(self, area, room, subsample_method, subsample_ratio):
        """
        Load S3DIS and subsample
        """
        if subsample_method == "uniform":
            # Take every Nth point
            points, labels = self.load_s3dis_room_uniform(area, room, subsample_rate=3)
        elif subsample_method == "random":
            # Keep points randomly
            points, labels = self.load_s3dis_room_random(
                area, room, subsample_ratio=0.3
            )
        elif subsample_method == "voxel":
            # voxel grid
            points, labels = self.load_s3dis_room_voxel(area, room, subsample_ratio)
        elif subsample_method == "adaptive":
            # Different rates per class
            points, labels = self.load_s3dis_room_adaptive(area, room)

        # # Create point cloud for segmentation
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # if points.shape[1] >= 6:
        #     pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])

        return points, labels

    def read_pcd(self):
        """
        Read pcd file and extract point cloud with colors

        Returns:
            pcd: o3d.geometry.PointCloud()
        """

        pcd = o3d.io.read_point_cloud(self.file_path)

        pcd = pcd.voxel_down_sample(self.voxel_size)

        points_xyz = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Swap axis
        R_yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        points_xyz = points_xyz @ R_yup_to_zup.T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Combine xyz and rgb
        # points = np.concatenate([points_xyz, colors], axis=1)
        return pcd

    def clean_pcd(self, method, nb, r):
        if method == "radius":
            self.cleaned_pcd, ind = self.pcd.remove_radius_outlier(
                nb_points=nb, radius=r
            )
        else:
            self.cleaned_pcd, ind = self.pcd.remove_statistical_outlier(
                nb_neighbors=nb, std_ratio=r
            )
        out_pcd = self.pcd.select_by_index(ind, invert=True)
        out_pcd.paint_uniform_color([1, 0, 0])
        self.out_pcd = out_pcd

    def visualize_pcd(self, type="raw"):
        if type == "raw":
            o3d.visualization.draw_geometries([self.pcd])
        else:
            o3d.visualization.draw_geometries([self.cleaned_pcd, self.out_pcd])
