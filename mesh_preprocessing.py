import os

import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors


def read_mesh_and_extract_points(mesh_path, num_points=100_000):
    """
    Read mesh file and extract point cloud with colors

    Args:
        mesh_path: Path to mesh file (.ply, .obj, .off, etc.)
        num_points: Number of points to sample from mesh

    Returns:
        points: numpy array of shape (N, 6) containing [x, y, z, r, g, b]
    """

    # Method 1: Using trimesh (recommended)
    try:
        mesh = trimesh.load(mesh_path)

        # Sample points from mesh surface
        points_xyz, face_indices = mesh.sample(num_points, return_index=True)

        # Get colors if available
        if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
            # Get colors from faces
            colors = mesh.visual.face_colors[face_indices][:, :3]  # RGB only
        elif (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
        ):
            # Interpolate vertex colors to sampled points
            colors = interpolate_vertex_colors_to_points(mesh, points_xyz, face_indices)
        else:
            # Default gray color if no colors available
            colors = np.full((num_points, 3), 128, dtype=np.uint8)

    except Exception as e:
        print(f"Trimesh failed: {e}")

        # Method 2: Using Open3D as fallback
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)

            # Sample points
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            points_xyz = np.asarray(pcd.points)

            # Get colors
            if len(pcd.colors) > 0:
                colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            else:
                colors = np.full((num_points, 3), 128, dtype=np.uint8)

        except Exception as e2:
            raise Exception(f"Both trimesh and open3d failed: {e}, {e2}")

    # Combine xyz and rgb
    points = np.concatenate([points_xyz, colors], axis=1)
    return points


def read_pcd_and_extract_points(pcd_path):
    """
    Read mesh file and extract point cloud with colors

    Args:
        mesh_path: Path to mesh file (.ply, .obj, .off, etc.)
        num_points: Number of points to sample from mesh

    Returns:
        points: numpy array of shape (N, 6) containing [x, y, z, r, g, b]
    """

    pcd = o3d.io.read_point_cloud(pcd_path)
    points_xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Combine xyz and rgb
    points = np.concatenate([points_xyz, colors], axis=1)
    return points


def interpolate_vertex_colors_to_points(mesh, points, face_indices):
    """
    Interpolate vertex colors to sampled points using barycentric coordinates
    """
    faces = mesh.faces[face_indices]
    vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGB only

    # For simplicity, use the color of the first vertex of each face
    # You could implement proper barycentric interpolation here
    colors = vertex_colors[faces[:, 0]]
    return colors


def preprocess_for_s3dis_inference(points, block_size=1.0, stride=0.5, num_point=4096):
    """
    Preprocess point cloud similar to S3DIS dataset format

    Args:
        points: numpy array of shape (N, 6) containing [x, y, z, r, g, b]
        block_size: Size of each block for processing
        stride: Stride for sliding window
        num_point: Number of points per block

    Returns:
        processed_blocks: List of numpy arrays, each of shape (num_point, 9)
                         Format: [x, y, z, r, g, b, norm_x, norm_y, norm_z]
    """

    # Step 1: Normalize the point cloud (shift to origin)
    xyz_min = np.min(points[:, :3], axis=0)
    points[:, :3] -= xyz_min

    # Step 2: Normalize RGB to [0, 1]
    points[:, 3:6] = points[:, 3:6].astype(np.float32) / 255.0

    # Step 3: Get room bounds
    xyz_max = np.max(points[:, :3], axis=0)

    # Step 4: Create sliding window blocks
    blocks = []

    # Calculate grid dimensions
    grid_x = int(np.ceil((xyz_max[0] - block_size) / stride)) + 1
    grid_y = int(np.ceil((xyz_max[1] - block_size) / stride)) + 1

    for i in range(grid_x):
        for j in range(grid_y):
            # Define block boundaries
            x_start = i * stride
            x_end = x_start + block_size
            y_start = j * stride
            y_end = y_start + block_size

            # Find points in current block
            mask = (
                (points[:, 0] >= x_start)
                & (points[:, 0] <= x_end)
                & (points[:, 1] >= y_start)
                & (points[:, 1] <= y_end)
            )

            block_points = points[mask]

            # Skip if too few points
            if len(block_points) < 100:
                continue

            # Sample or duplicate points to get exactly num_point points
            if len(block_points) >= num_point:
                # Random sampling
                indices = np.random.choice(len(block_points), num_point, replace=False)
                sampled_points = block_points[indices]
            else:
                # Duplicate points
                indices = np.random.choice(len(block_points), num_point, replace=True)
                sampled_points = block_points[indices]

            # Create 9-channel format: [x, y, z, r, g, b, norm_x, norm_y, norm_z]
            processed_block = np.zeros((num_point, 9))

            # Center the block
            block_center = np.array([(x_start + x_end) / 2, (y_start + y_end) / 2, 0])
            centered_points = sampled_points.copy()
            centered_points[:, 0] -= block_center[0]
            centered_points[:, 1] -= block_center[1]

            # Fill the processed block
            processed_block[:, 0:6] = centered_points  # xyz + rgb

            # Add normalized coordinates (6th, 7th, 8th channels)
            processed_block[:, 6] = sampled_points[:, 0] / xyz_max[0]  # normalized x
            processed_block[:, 7] = sampled_points[:, 1] / xyz_max[1]  # normalized y
            processed_block[:, 8] = sampled_points[:, 2] / xyz_max[2]  # normalized z

            blocks.append(processed_block)

    return blocks


def prepare_for_inference(mesh_path, model_input_format="torch"):
    """
    Complete preprocessing pipeline for inference

    Args:
        mesh_path: Path to mesh file
        model_input_format: 'torch' or 'numpy'

    Returns:
        data_blocks: Preprocessed blocks ready for model inference
        block_info: Information about each block for reconstruction
    """

    print("Step 1: Reading mesh and extracting points...")
    points = read_pcd_and_extract_points(pcd_path)
    print(f"Extracted {len(points)} points from pcd")

    print("Step 2: Preprocessing for S3DIS format...")
    blocks = preprocess_for_s3dis_inference(points)
    print(f"Created {len(blocks)} blocks for inference")

    if len(blocks) == 0:
        raise ValueError(
            "No valid blocks created. Check your mesh size and parameters."
        )

    # Convert to appropriate format
    if model_input_format == "torch":
        # Convert to torch tensors and transpose for PointNet++ format
        data_blocks = []
        for block in blocks:
            # PointNet++ expects (batch, channels, points) format
            block_tensor = (
                torch.FloatTensor(block).transpose(0, 1).unsqueeze(0)
            )  # (1, 9, 4096)
            data_blocks.append(block_tensor)
    else:
        data_blocks = blocks

    # Create block info for reconstruction (if needed later)
    block_info = {
        "num_blocks": len(blocks),
        "points_per_block": len(blocks[0]) if blocks else 0,
        "original_bounds": (
            np.min(points[:, :3], axis=0).tolist() if len(points) > 0 else None
        ),
    }

    return data_blocks, block_info


def run_inference_example(mesh_path, model_path=None):
    """
    Example of how to run inference on preprocessed mesh
    """

    # Preprocess mesh
    data_blocks, block_info = prepare_for_inference(mesh_path)

    print(f"Prepared {len(data_blocks)} blocks for inference")
    print(f"Each block shape: {data_blocks[0].shape}")  # Should be (1, 9, 4096)

    # Here you would load your pretrained model and run inference
    # model = load_pretrained_model(model_path)
    # predictions = []
    # for block in data_blocks:
    #     with torch.no_grad():
    #         pred, _ = model(block)
    #         predictions.append(pred)

    return data_blocks, block_info


# Example usage
if __name__ == "__main__":
    mesh_path = "path/to/your/room.ply"  # Replace with your mesh path

    try:
        data_blocks, block_info = run_inference_example(mesh_path)
        print("Preprocessing completed successfully!")
        print(f"Ready for inference with {block_info['num_blocks']} blocks")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
