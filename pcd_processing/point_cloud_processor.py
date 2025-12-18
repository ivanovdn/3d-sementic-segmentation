#!/usr/bin/env python3
"""
point_cloud_processor.py

Process binary point cloud files from iOS hybrid capture:
1. Load binary file (28 bytes per point: xyz + rgb + confidence)
2. Deduplicate using voxel grid with centroid averaging
3. Export to PLY format (ASCII or binary)
4. Print comprehensive statistics

Binary Format:
    - 28 bytes per point (tightly packed, no padding)
    - 7 float32 values: x, y, z, r, g, b, confidence
    - Colors normalized [0.0, 1.0] not [0, 255]
    - Little-endian byte order

Usage Examples:
    # Basic processing (5mm voxels, ASCII PLY)
    python point_cloud_processor.py scan_data.bin output.ply

    # High detail (2mm voxels)
    python point_cloud_processor.py scan.bin detailed.ply --voxel-size 0.002

    # Coarse + binary output (10× faster and smaller)
    python point_cloud_processor.py scan.bin coarse.ply --voxel-size 0.01 --binary-ply

    # Statistics only (no export)
    python point_cloud_processor.py scan.bin dummy.ply --stats-only

    # No deduplication (export raw points)
    python point_cloud_processor.py scan.bin raw.ply --no-dedup
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d


def load_binary_point_cloud(filepath: str) -> np.ndarray:
    """
    Load point cloud from binary file.

    Format: position(12 bytes) + color(12 bytes) + confidence(4 bytes) = 28 bytes per point
    Each field is float32 (4 bytes)

    Args:
        filepath: Path to binary point cloud file

    Returns:
        numpy array of shape (N, 7): [x, y, z, r, g, b, confidence]

    Raises:
        ValueError: If file size is not a multiple of 28 bytes
        FileNotFoundError: If file doesn't exist
    """
    print(f"📂 Loading {filepath}...")

    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = file_path.stat().st_size
    num_points = file_size // 28

    if file_size % 28 != 0:
        print(f"⚠️  WARNING: File size ({file_size}) is not a multiple of 28 bytes!")
        print(f"   Expected: {num_points * 28} bytes")
        print(f"   Truncating to {num_points} complete points")

    # Load all floats at once
    data = np.fromfile(filepath, dtype=np.float32)

    # Reshape to points (7 floats per point: xyz + rgb + confidence)
    points = data[: num_points * 7].reshape(-1, 7)

    print(f"✅ Loaded {num_points:,} points ({file_size / 1024 / 1024:.1f} MB)")

    return points


def voxel_grid_deduplication(
    points: np.ndarray, voxel_size: float = 0.005
) -> Tuple[np.ndarray, dict]:
    """
    Fast voxel grid deduplication using Open3D C++ backend.

    Uses Open3D's highly optimized voxel_down_sample with multithreaded C++ implementation.
    Confidence values are preserved using voxel-based lookup after downsampling.

    Args:
        points: Array of shape (N, 7) - [x, y, z, r, g, b, confidence]
        voxel_size: Voxel size in meters (0.005 = 5mm, 0.01 = 1cm)

    Returns:
        Tuple of:
        - Deduplicated points array of shape (M, 7) where M <= N
        - Statistics dictionary with deduplication metrics
    """
    print(f"🔧 Deduplicating with voxel size {voxel_size*1000:.1f}mm...")
    start_time = time.time()

    # Extract data
    positions = points[:, :3]
    colors = points[:, 3:6]
    confidences = points[:, 6]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Voxel downsampling (C++ multithreaded, very fast)
    print(f"   Processing {len(points):,} points...")
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Extract deduplicated data
    deduplicated_positions = np.asarray(downsampled.points, dtype=np.float32)
    deduplicated_colors = np.asarray(downsampled.colors, dtype=np.float32)
    num_unique_voxels = len(deduplicated_positions)

    # Handle confidence: Find max confidence per voxel using vectorized operations
    print(f"   Computing confidence values for {num_unique_voxels:,} voxels...")

    # Quantize original points to voxel grid
    orig_voxel_indices = np.floor(positions / voxel_size).astype(np.int32)
    dedup_voxel_indices = np.floor(deduplicated_positions / voxel_size).astype(np.int32)

    # Create voxel keys for both original and deduplicated points
    def voxel_key(indices):
        return (
            (indices[:, 0].astype(np.int64) + 1_000_000) * 1_000_000_000_000
            + (indices[:, 1].astype(np.int64) + 1_000_000) * 1_000_000
            + (indices[:, 2].astype(np.int64) + 1_000_000)
        )

    orig_keys = voxel_key(orig_voxel_indices)
    dedup_keys = voxel_key(dedup_voxel_indices)

    # Sort original points by voxel key for efficient grouping
    sort_idx = np.argsort(orig_keys)
    sorted_keys = orig_keys[sort_idx]
    sorted_confidences = confidences[sort_idx]

    # Find unique voxels and their boundaries in sorted array
    unique_sorted_keys, split_indices = np.unique(sorted_keys, return_index=True)

    # Compute max confidence per voxel using reduceat
    split_indices = np.append(split_indices, len(sorted_confidences))
    max_confidences_per_voxel = np.maximum.reduceat(
        sorted_confidences, split_indices[:-1]
    )

    # Create mapping from voxel key to max confidence
    voxel_to_confidence = dict(zip(unique_sorted_keys, max_confidences_per_voxel))

    # Look up confidence for each deduplicated point (vectorized)
    deduplicated_confidences = np.array(
        [voxel_to_confidence.get(key, 0.0) for key in dedup_keys], dtype=np.float32
    )

    # Combine into final array
    deduplicated_points = np.hstack(
        [
            deduplicated_positions,
            deduplicated_colors,
            deduplicated_confidences.reshape(-1, 1),
        ]
    )

    dedup_time = time.time() - start_time

    # Calculate statistics
    dedup_rate = (1 - num_unique_voxels / len(points)) * 100

    stats = {
        "original_count": len(points),
        "deduplicated_count": num_unique_voxels,
        "deduplication_rate": dedup_rate,
        "processing_time": dedup_time,
        "method": "open3d",
    }

    print(
        f"✅ Reduced from {len(points):,} to {num_unique_voxels:,} points in {dedup_time:.3f}s"
    )
    print(f"   Deduplication rate: {dedup_rate:.1f}%")
    print(f"   Processing speed: {len(points)/dedup_time/1000:.0f}K points/sec")

    return deduplicated_points, stats


def export_to_ply_ascii(points: np.ndarray, output_path: str):
    """
    Export points to PLY format (ASCII).

    Args:
        points: Array of shape (N, 7) - [x, y, z, r, g, b, confidence]
        output_path: Output PLY file path
    """
    print(f"💾 Exporting to ASCII PLY: {output_path}...")

    start_time = time.time()
    num_points = len(points)

    # Convert colors from [0, 1] to [0, 255]
    colors_uint8 = (np.clip(points[:, 3:6], 0, 1) * 255).astype(np.uint8)

    # Write PLY file
    with open(output_path, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Generated from iOS LiDAR hybrid capture\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float confidence\n")
        f.write("end_header\n")

        # Point data
        for i in range(num_points):
            x, y, z = points[i, :3]
            r, g, b = colors_uint8[i]
            confidence = points[i, 6]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {confidence:.3f}\n")

    duration = time.time() - start_time
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✅ Exported {num_points:,} points ({file_size:.1f} MB) in {duration:.2f}s")


def export_to_ply_binary(points: np.ndarray, output_path: str):
    """
    Export points to PLY format (binary - much faster and smaller).

    Binary PLY is typically 10× faster to write and 10× smaller than ASCII.

    Args:
        points: Array of shape (N, 7) - [x, y, z, r, g, b, confidence]
        output_path: Output PLY file path
    """
    print(f"💾 Exporting to binary PLY: {output_path}...")

    start_time = time.time()
    num_points = len(points)

    # Convert colors from [0, 1] to [0, 255]
    colors_uint8 = (np.clip(points[:, 3:6], 0, 1) * 255).astype(np.uint8)

    # Write header
    header = f"""ply
format binary_little_endian 1.0
comment Generated from iOS LiDAR hybrid capture
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float confidence
end_header
"""

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))

        # Write binary data (interleaved: xyz rgb confidence xyz rgb confidence ...)
        for i in range(num_points):
            # Position (3 floats = 12 bytes)
            f.write(points[i, :3].astype(np.float32).tobytes())
            # Color (3 bytes)
            f.write(colors_uint8[i].tobytes())
            # Confidence (1 float = 4 bytes)
            f.write(points[i, 6:7].astype(np.float32).tobytes())

    duration = time.time() - start_time
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✅ Exported {num_points:,} points ({file_size:.1f} MB) in {duration:.2f}s")


def print_statistics(points: np.ndarray):
    """
    Print detailed point cloud statistics.

    Args:
        points: Array of shape (N, 7) - [x, y, z, r, g, b, confidence]
    """
    print("\n📊 Point Cloud Statistics:")
    print(f"   Total points: {len(points):,}")

    positions = points[:, :3]
    colors = points[:, 3:6]
    confidences = points[:, 6]

    # Bounding box
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    print(f"   Bounding box:")
    print(f"     X: [{min_pos[0]:.3f}, {max_pos[0]:.3f}] m")
    print(f"     Y: [{min_pos[1]:.3f}, {max_pos[1]:.3f}] m")
    print(f"     Z: [{min_pos[2]:.3f}, {max_pos[2]:.3f}] m")

    # Dimensions
    dimensions = max_pos - min_pos
    print(
        f"   Dimensions: {dimensions[0]:.2f} × {dimensions[1]:.2f} × {dimensions[2]:.2f} m"
    )
    print(f"   Volume: {np.prod(dimensions):.2f} m³")

    # Point density
    volume = np.prod(dimensions)
    if volume > 0:
        density = len(points) / volume
        print(f"   Point density: {density:.0f} points/m³")

    # Color statistics
    print(f"   Color range:")
    print(f"     R: [{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
    print(f"     G: [{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}]")
    print(f"     B: [{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")

    # Confidence statistics
    print(f"   Confidence:")
    print(f"     Min: {confidences.min():.3f}")
    print(f"     Max: {confidences.max():.3f}")
    print(f"     Mean: {confidences.mean():.3f}")
    high_conf_count = (confidences > 0.8).sum()
    high_conf_pct = high_conf_count / len(confidences) * 100
    print(f"     High (>0.8): {high_conf_count:,} points ({high_conf_pct:.1f}%)")


def main():
    """
    Main entry point for point cloud processing.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Process binary point cloud from iOS hybrid capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing with 5mm voxels
  python point_cloud_processor.py scan_data.bin output.ply

  # High detail with 2mm voxels
  python point_cloud_processor.py scan.bin detailed.ply --voxel-size 0.002

  # Coarse with 1cm voxels, binary PLY output
  python point_cloud_processor.py scan.bin coarse.ply --voxel-size 0.01 --binary-ply

  # No deduplication, export raw points
  python point_cloud_processor.py scan.bin raw.ply --no-dedup

  # Statistics only, no export
  python point_cloud_processor.py scan.bin dummy.ply --stats-only
        """,
    )
    parser.add_argument("input", help="Input binary file (.bin)")
    parser.add_argument("output", help="Output PLY file (.ply)")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel size for deduplication in meters (default: 0.005 = 5mm)",
    )
    parser.add_argument(
        "--binary-ply",
        action="store_true",
        help="Export binary PLY (10× smaller and faster than ASCII)",
    )
    parser.add_argument(
        "--no-dedup", action="store_true", help="Skip deduplication (export raw points)"
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="Only print statistics, do not export"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    try:
        overall_start = time.time()

        # Step 1: Load binary point cloud
        load_start = time.time()
        points = load_binary_point_cloud(args.input)
        load_time = time.time() - load_start

        if len(points) == 0:
            print("❌ Error: No points loaded")
            return 1

        # Step 2: Deduplicate if requested
        dedup_time = 0
        if not args.no_dedup:
            dedup_start = time.time()
            points, dedup_stats = voxel_grid_deduplication(points, args.voxel_size)
            dedup_time = time.time() - dedup_start
        else:
            print("⚠️  Skipping deduplication (--no-dedup flag)")

        # Step 3: Print statistics
        print_statistics(points)

        # Step 4: Export to PLY (unless stats-only)
        export_time = 0
        if not args.stats_only:
            export_start = time.time()
            if args.binary_ply:
                export_to_ply_binary(points, args.output)
            else:
                export_to_ply_ascii(points, args.output)
            export_time = time.time() - export_start

            print("\n✅ Processing complete!")
        else:
            print("\n✅ Statistics complete (no export - stats-only mode)")

        # Print performance summary
        total_time = time.time() - overall_start
        print(f"\n⏱️  Performance Summary:")
        print(f"   Load time:       {load_time:.2f}s")
        if not args.no_dedup:
            print(f"   Dedup time:      {dedup_time:.2f}s")
        if not args.stats_only:
            print(f"   Export time:     {export_time:.2f}s")
        print(f"   Total time:      {total_time:.2f}s")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
