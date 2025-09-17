import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from mesh_preprocessing import read_pcd_and_extract_points
from s3dis_inference_integration import S3DISInference


class QuickVisualizer:
    """Quick visualization for your semantic segmentation results"""

    def __init__(self):
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

        # Distinctive colors for each class (RGB in [0,1])
        self.class_colors = np.array(
            [
                [0.9, 0.9, 0.9],  # ceiling - light gray
                [0.6, 0.3, 0.0],  # floor - brown
                [0.8, 0.8, 0.6],  # wall - beige
                [0.4, 0.2, 0.0],  # beam - dark brown
                [0.7, 0.7, 0.7],  # column - gray
                [0.0, 0.5, 1.0],  # window - blue
                [0.8, 0.4, 0.0],  # door - orange
                [0.6, 0.3, 0.3],  # table - dark red
                [1.0, 0.0, 0.0],  # chair - red
                [0.0, 0.8, 0.0],  # sofa - green
                [0.5, 0.0, 0.5],  # bookcase - purple
                [0.0, 0.0, 0.0],  # board - black
                [1.0, 1.0, 0.0],  # clutter - yellow
            ]
        )

    def reconstruct_with_labels(self, pcd_path, predictions, block_indices):
        """Reconstruct point cloud with semantic labels using majority voting"""

        print("Reconstructing point cloud with semantic labels...")

        # Load original point cloud
        original_points = read_pcd_and_extract_points(pcd_path)
        print(f"Original point cloud has {len(original_points)} points")

        # Initialize voting system
        num_points = len(original_points)
        vote_counts = np.zeros((num_points, len(self.class_names)), dtype=np.int32)

        # Accumulate votes from all blocks
        total_votes = 0
        for block_pred, block_idx in zip(predictions, block_indices):
            for i, point_idx in enumerate(block_idx):
                if point_idx < num_points:
                    predicted_class = block_pred[i]
                    if 0 <= predicted_class < len(self.class_names):
                        vote_counts[point_idx, predicted_class] += 1
                        total_votes += 1

        print(f"Collected {total_votes} votes from {len(predictions)} blocks")

        # Assign final labels based on majority voting
        point_labels = np.argmax(vote_counts, axis=1)

        # Handle points with no votes (assign to clutter)
        no_votes_mask = np.sum(vote_counts, axis=1) == 0
        point_labels[no_votes_mask] = 12  # clutter

        print(f"Points with no votes: {np.sum(no_votes_mask)}")

        # Combine original points with labels
        labeled_points = np.column_stack([original_points, point_labels])

        return labeled_points

    def create_colored_pointcloud(self, labeled_points, save_path=None):
        """Create and save point cloud with semantic colors"""

        print("Creating colored point cloud...")

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(labeled_points[:, :3])

        # Assign semantic colors
        semantic_colors = np.zeros((len(labeled_points), 3))
        for i, label in enumerate(labeled_points[:, 6]):
            label = int(label)
            if 0 <= label < len(self.class_colors):
                semantic_colors[i] = self.class_colors[label]
            else:
                semantic_colors[i] = [0.5, 0.5, 0.5]  # Gray for unknown

        pcd.colors = o3d.utility.Vector3dVector(semantic_colors)

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(save_path), pcd)
            print(f"Colored point cloud saved to: {save_path}")

        return pcd

    def visualize(self, labeled_points, class_counts, save_colored_pcd=None):
        """Visualize the semantic segmentation results"""

        # Create colored point cloud
        colored_pcd = self.create_colored_pointcloud(labeled_points, save_colored_pcd)

        # Show 3D visualization
        print("Opening 3D visualization... Close the window to continue.")
        o3d.visualization.draw_geometries(
            [colored_pcd],
            window_name="Semantic Segmentation Results",
            width=1200,
            height=800,
            point_show_normal=False,
        )

        # Show color legend
        self.show_legend(class_counts)

    def show_legend(self, class_counts, save_path=None):
        """Show color legend with class statistics"""

        # Filter classes with points
        active_classes = [
            (name, stats) for name, stats in class_counts.items() if stats["count"] > 0
        ]

        # Sort by count (descending)
        active_classes.sort(key=lambda x: x[1]["count"], reverse=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        y_positions = range(len(active_classes))

        # Create bars and labels
        for i, (class_name, stats) in enumerate(active_classes):
            class_idx = self.class_names.index(class_name)
            color = self.class_colors[class_idx]

            # Create colored bar
            ax.barh(i, 1, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

            # Add text label
            label = f"{class_name.capitalize()}: {stats['count']:,} points ({stats['percentage']:.1f}%)"
            ax.text(0.02, i, label, va="center", fontsize=11, fontweight="bold")

        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(active_classes) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(
            "Semantic Segmentation Results - Class Distribution",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Remove axes borders
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Legend saved to: {save_path}")

        plt.show()


def run_complete_inference_with_visualization(
    pcd_path, model_path, output_dir="results"
):
    """Run complete inference pipeline with visualization"""

    print("=" * 60)
    print("SEMANTIC SEGMENTATION WITH VISUALIZATION")
    print("=" * 60)

    # Initialize inference
    print("Loading model...")
    inference = S3DISInference(model_path=model_path, model_type="pointnet2")

    # Run inference
    predictions, class_counts, block_indices = inference.run_inference(
        pcd_path, output_dir
    )

    # Initialize visualizer
    visualizer = QuickVisualizer()

    # Reconstruct point cloud with labels
    labeled_points = visualizer.reconstruct_with_labels(
        pcd_path, predictions, block_indices
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save colored point cloud
    colored_pcd_path = output_path / "semantic_colored_pointcloud.ply"
    legend_path = output_path / "class_legend.png"

    # Visualize results
    visualizer.visualize(
        labeled_points, class_counts, save_colored_pcd=colored_pcd_path
    )
    visualizer.show_legend(class_counts, save_path=legend_path)

    # Save labeled points as numpy array
    np.save(output_path / "labeled_pointcloud.npy", labeled_points)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("Files created:")
    print(f"  - semantic_colored_pointcloud.ply (3D point cloud with colors)")
    print(f"  - class_legend.png (color legend)")
    print(f"  - labeled_pointcloud.npy (point cloud + labels)")

    return labeled_points, class_counts


def visualize_existing_results(
    pcd_path, predictions_file, output_dir="visualization_results"
):
    """Visualize existing inference results (if you already have predictions saved)"""

    print("Visualizing existing results...")

    # Load saved predictions
    predictions = np.load(predictions_file, allow_pickle=True)

    # You'll need to provide block_indices - this is a limitation of saved results
    # For now, create dummy block indices (you should save these in your inference)
    print("Warning: Using dummy block indices. For accurate reconstruction,")
    print("block indices should be saved during inference.")

    # Create dummy block indices (this won't be perfectly accurate)
    original_points = read_pcd_and_extract_points(pcd_path)
    total_points = len(original_points)
    points_per_block = 4096

    block_indices = []
    for i, pred in enumerate(predictions):
        start_idx = (i * points_per_block) % total_points
        end_idx = min(start_idx + points_per_block, total_points)
        indices = np.arange(start_idx, end_idx)
        if len(indices) < points_per_block:
            # Wrap around or duplicate
            additional = np.random.choice(total_points, points_per_block - len(indices))
            indices = np.concatenate([indices, additional])
        block_indices.append(indices)

    # Calculate class counts from predictions
    all_preds = np.concatenate(predictions)
    class_names = [
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

    class_counts = {}
    total_points_pred = len(all_preds)
    for i, class_name in enumerate(class_names):
        count = np.sum(all_preds == i)
        percentage = (count / total_points_pred) * 100
        class_counts[class_name] = {
            "count": int(count),
            "percentage": float(percentage),
        }

    # Visualize
    visualizer = QuickVisualizer()
    labeled_points = visualizer.reconstruct_with_labels(
        pcd_path, predictions, block_indices
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    colored_pcd_path = output_path / "semantic_colored_pointcloud.ply"
    visualizer.visualize(
        labeled_points, class_counts, save_colored_pcd=colored_pcd_path
    )

    return labeled_points, class_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize semantic segmentation results"
    )
    parser.add_argument(
        "--pcd_path", type=str, required=True, help="Path to input point cloud"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to trained model (for full inference)"
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        help="Path to saved predictions (for visualization only)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )

    args = parser.parse_args()

    if args.model_path:
        # Run complete inference with visualization
        labeled_points, class_counts = run_complete_inference_with_visualization(
            args.pcd_path, args.model_path, args.output_dir
        )
    elif args.predictions_file:
        # Visualize existing results
        labeled_points, class_counts = visualize_existing_results(
            args.pcd_path, args.predictions_file, args.output_dir
        )
    else:
        print(
            "Error: Provide either --model_path for full inference or --predictions_file for visualization only"
        )
        sys.exit(1)


# Example usage with your current results:
"""
# If you want to run this with your existing setup, modify your s3dis_inference_integration.py
# to save block_indices along with predictions, then use:

visualizer = QuickVisualizer()

# Your class counts from the output you provided
class_counts = {
    'ceiling': {'count': 75312, 'percentage': 23.0},
    'floor': {'count': 59458, 'percentage': 18.1},
    'wall': {'count': 58522, 'percentage': 17.9},
    'beam': {'count': 1, 'percentage': 0.0},
    'column': {'count': 87, 'percentage': 0.0},
    'window': {'count': 6042, 'percentage': 1.8},
    'door': {'count': 8773, 'percentage': 2.7},
    'table': {'count': 10427, 'percentage': 3.2},
    'chair': {'count': 8993, 'percentage': 2.7},
    'sofa': {'count': 22429, 'percentage': 6.8},
    'bookcase': {'count': 11757, 'percentage': 3.6},
    'board': {'count': 4199, 'percentage': 1.3},
    'clutter': {'count': 61680, 'percentage': 18.8}
}

# Load your predictions and block_indices
predictions = np.load("path/to/predictions.npy", allow_pickle=True)
block_indices = np.load("path/to/block_indices.npy", allow_pickle=True)  # You need to save this

# Reconstruct and visualize
labeled_points = visualizer.reconstruct_with_labels("path/to/pointcloud.ply", predictions, block_indices)
visualizer.visualize(labeled_points, class_counts, save_colored_pcd="semantic_result.ply")
"""
