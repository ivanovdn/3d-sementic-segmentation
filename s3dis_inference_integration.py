import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Import the existing S3DIS model and utilities
# Assuming the repo structure from the provided files
from pointnet2_sem_seg import get_model as get_pointnet2_model
# from pointnet_sem_seg import get_model as get_pointnet_model


class S3DISInference:
    """
    Class for running inference on custom meshes using pretrained S3DIS models
    """

    def __init__(self, model_path, model_type="pointnet2", num_classes=13):
        """
        Initialize the inference class

        Args:
            model_path: Path to pretrained model (.pth file)
            model_type: 'pointnet' or 'pointnet2'
            num_classes: Number of classes (13 for S3DIS)
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")

        # S3DIS class names
        self.class_names = [
            "ceiling",
            "floor",
            "wall",
            "beam",
            "column",
            "window",
            "door",
            "table",
            "chair",
            "sofa",
            "bookcase",
            "board",
            "clutter",
        ]

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        """Load the pretrained model"""

        if self.model_type == "pointnet2":
            model = get_pointnet2_model(self.num_classes)
        elif self.model_type == "pointnet":
            model = get_pointnet_model(self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        return model

    def preprocess_mesh_for_inference(
        self, pcd_path, block_size=1.0, stride=0.5, num_point=4096
    ):
        """
        Preprocess mesh using the same format as S3DIS dataset
        This integrates with the existing data preprocessing pipeline
        """

        # Import the mesh preprocessing function we created earlier
        from mesh_preprocessing import read_pcd_and_extract_points

        print("Reading mesh and extracting points...")
        points = read_pcd_and_extract_points(pcd_path)

        # Use the same preprocessing as S3DIS dataset
        # Based on the S3DISDataLoader and indoor3d_util.py
        data_blocks = self._create_blocks_like_s3dis(
            points, block_size, stride, num_point
        )

        return data_blocks

    def _create_blocks_like_s3dis(self, points, block_size, stride, num_point):
        """
        Create blocks similar to S3DIS dataset processing
        Based on the ScannetDatasetWholeScene class logic
        """

        # Normalize coordinates
        xyz_min = np.min(points[:, :3], axis=0)
        points[:, :3] -= xyz_min

        # Normalize colors to [0,1]
        points[:, 3:6] = points[:, 3:6].astype(np.float32)

        coord_max = np.max(points[:, :3], axis=0)

        # Create grid
        grid_x = int(np.ceil(float(coord_max[0] - block_size) / stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - block_size) / stride) + 1)

        data_blocks = []
        block_indices = []

        for index_y in range(grid_y):
            for index_x in range(grid_x):
                s_x = index_x * stride
                e_x = min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size

                s_y = index_y * stride
                e_y = min(s_y + block_size, coord_max[1])

                # Find points in block
                point_idxs = np.where(
                    (points[:, 0] >= s_x - 0.001)
                    & (points[:, 0] <= e_x + 0.001)
                    & (points[:, 1] >= s_y - 0.001)
                    & (points[:, 1] <= e_y + 0.001)
                )[0]

                if point_idxs.size < 200:
                    continue

                # Sample points
                if point_idxs.size >= num_point:
                    selected_point_idxs = np.random.choice(
                        point_idxs, num_point, replace=False
                    )
                else:
                    selected_point_idxs = np.random.choice(
                        point_idxs, num_point, replace=True
                    )

                # Process block similar to S3DIS format
                selected_points = points[selected_point_idxs, :]

                # Create 9-channel format like S3DIS
                current_points = np.zeros((num_point, 9))

                # Normalized coordinates (channels 6,7,8)
                current_points[:, 6] = selected_points[:, 0] / coord_max[0]
                current_points[:, 7] = selected_points[:, 1] / coord_max[1]
                current_points[:, 8] = selected_points[:, 2] / coord_max[2]

                # Center the block
                selected_points[:, 0] = selected_points[:, 0] - (s_x + block_size / 2.0)
                selected_points[:, 1] = selected_points[:, 1] - (s_y + block_size / 2.0)

                # Copy xyz and rgb (channels 0-5)
                current_points[:, 0:6] = selected_points

                data_blocks.append(current_points)
                block_indices.append(selected_point_idxs)

        return data_blocks, block_indices

    def run_inference(self, mesh_path, output_path=None):
        """
        Run semantic segmentation inference on a mesh

        Args:
            mesh_path: Path to input mesh
            output_path: Path to save results (optional)

        Returns:
            predictions: List of predictions for each block
            class_counts: Count of each predicted class
        """

        print(f"Running inference on {mesh_path}")

        # Preprocess mesh
        data_blocks, block_indices = self.preprocess_mesh_for_inference(mesh_path)

        print(f"Created {len(data_blocks)} blocks for inference")

        if len(data_blocks) == 0:
            raise ValueError("No valid blocks created from mesh")

        # Run inference on each block
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for i, block in enumerate(data_blocks):
                if i % 10 == 0:
                    print(f"Processing block {i+1}/{len(data_blocks)}")

                # Convert to tensor and add batch dimension
                block_tensor = (
                    torch.FloatTensor(block).transpose(0, 1).unsqueeze(0)
                )  # (1, 9, 4096)
                block_tensor = block_tensor.to(self.device)

                # Forward pass
                pred, _ = self.model(block_tensor)

                # Get predictions
                pred = pred.contiguous().view(-1, self.num_classes)  # (4096, 13)
                # pred_choice = pred.data.max(1)[1]  # (4096,)
                # pred_probs = F.softmax(pred, dim=1)  # (4096, 13)

                pred_probs = F.softmax(pred, dim=1)
                confidences = torch.max(pred_probs, dim=1)[0]
                pred_choice = pred_probs.data.max(1)[1]
                low_confidence_mask = confidences < 0.2
                pred_choice[low_confidence_mask] = 12  # clutter

                all_predictions.append(pred_choice.cpu().numpy())
                all_probabilities.append(pred_probs.cpu().numpy())

        # Analyze results
        class_counts = self._analyze_predictions(all_predictions)

        # Save results if requested
        if output_path:
            self._save_results(
                all_predictions, all_probabilities, output_path, class_counts
            )
        np.save(output_path + "block_indices.npy", block_indices)
        return all_predictions, class_counts, block_indices

    def _analyze_predictions(self, predictions):
        """Analyze prediction results"""

        # Flatten all predictions
        all_preds = np.concatenate(predictions)

        # Count each class
        class_counts = {}
        total_points = len(all_preds)

        for class_idx in range(self.num_classes):
            count = np.sum(all_preds == class_idx)
            percentage = (count / total_points) * 100
            class_counts[self.class_names[class_idx]] = {
                "count": int(count),
                "percentage": float(percentage),
            }

        return class_counts

    def _save_results(self, predictions, probabilities, output_path, class_counts):
        """Save inference results"""

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        np.save(output_dir / "predictions.npy", predictions)
        np.save(output_dir / "probabilities.npy", probabilities)

        # Save analysis
        import json

        with open(output_dir / "class_analysis.json", "w") as f:
            json.dump(class_counts, f, indent=2)

        print(f"Results saved to {output_path}")

    def print_results(self, class_counts):
        """Print inference results"""

        print("\n" + "=" * 50)
        print("SEMANTIC SEGMENTATION RESULTS")
        print("=" * 50)

        for class_name, stats in class_counts.items():
            print(
                f"{class_name:12s}: {stats['count']:8d} points ({stats['percentage']:5.1f}%)"
            )

        print("=" * 50)


def main():
    """Example usage"""

    parser = argparse.ArgumentParser(
        description="Run semantic segmentation on custom mesh"
    )
    parser.add_argument(
        "--mesh_path", type=str, required=True, help="Path to input mesh file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--model_type", type=str, default="pointnet2", choices=["pointnet", "pointnet2"]
    )
    parser.add_argument("--output_path", type=str, help="Path to save results")

    args = parser.parse_args()

    # Initialize inference class
    inference = S3DISInference(model_path=args.model_path, model_type=args.model_type)

    # Run inference
    predictions, class_counts = inference.run_inference(
        mesh_path=args.mesh_path, output_path=args.output_path
    )

    # Print results
    inference.print_results(class_counts)


if __name__ == "__main__":
    main()

# Example usage in code:
"""
# Initialize inference
inference = S3DISInference(
    model_path='path/to/pretrained_model.pth',
    model_type='pointnet2'
)

# Run inference on your mesh
predictions, class_counts = inference.run_inference(
    mesh_path='path/to/your/room.ply',
    output_path='results/'
)

# Print results
inference.print_results(class_counts)
"""
