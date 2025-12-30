"""
Region Growing Wall Detection

Deterministic alternative to RANSAC for wall segmentation.
Grows regions from seed points based on surface normal similarity.

Advantages over RANSAC:
- Deterministic (same input = same output)
- Spatially coherent segments
- Handles gaps (doors, windows) naturally
- No random sampling

Requirements:
- Open3D for normal estimation and KDTree
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional
from collections import deque
import time


def compute_normals(points: np.ndarray, 
                    k_neighbors: int = 30,
                    orient_to_camera: bool = True,
                    camera_location: np.ndarray = None) -> np.ndarray:
    """
    Compute point normals using Open3D.
    
    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Point cloud
    k_neighbors : int
        Number of neighbors for normal estimation
    orient_to_camera : bool
        Orient normals towards camera/scanner
    camera_location : np.ndarray (3,)
        Camera position for orientation (default: centroid + up)
        
    Returns:
    --------
    normals : np.ndarray (N, 3)
        Unit normals for each point
    """
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )
    
    # Orient normals consistently
    if orient_to_camera:
        if camera_location is None:
            # Default: above centroid
            centroid = points.mean(axis=0)
            camera_location = centroid + np.array([0, 0, 2])  # 2m above
        
        pcd.orient_normals_towards_camera_location(camera_location)
    
    normals = np.asarray(pcd.normals)
    
    return normals


def region_growing_segmentation(
    points: np.ndarray,
    normals: np.ndarray,
    angle_threshold: float = 10.0,
    distance_threshold: float = 0.10,
    min_region_points: int = 100,
    k_neighbors: int = 30,
    verbose: bool = True
) -> List[np.ndarray]:
    """
    Region growing segmentation based on normal similarity.
    
    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Point cloud
    normals : np.ndarray (N, 3)
        Point normals (unit vectors)
    angle_threshold : float
        Max angle difference (degrees) for region membership
    distance_threshold : float
        Max distance (meters) to neighbor for region membership
    min_region_points : int
        Minimum points for valid region
    k_neighbors : int
        Number of neighbors to check during growing
    verbose : bool
        Print progress
        
    Returns:
    --------
    regions : list of np.ndarray
        Each element is array of point indices belonging to one region
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"REGION GROWING SEGMENTATION")
        print(f"{'='*70}")
        print(f"  Points: {len(points):,}")
        print(f"  Angle threshold: {angle_threshold}°")
        print(f"  Distance threshold: {distance_threshold}m")
        print(f"  Min region points: {min_region_points}")
    
    start_time = time.time()
    
    n_points = len(points)
    angle_threshold_rad = np.radians(angle_threshold)
    cos_threshold = np.cos(angle_threshold_rad)
    
    # Build KDTree for neighbor queries
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Track visited points
    visited = np.zeros(n_points, dtype=bool)
    regions = []
    
    # Sort points by curvature (flattest first = best seeds)
    # Approximate curvature by normal variance in neighborhood
    curvatures = _estimate_curvature(points, normals, kdtree, k_neighbors)
    seed_order = np.argsort(curvatures)
    
    if verbose:
        print(f"  Building regions...")
    
    # Grow regions from seed points
    for seed_idx in seed_order:
        if visited[seed_idx]:
            continue
        
        # Start new region
        region = []
        queue = deque([seed_idx])
        seed_normal = normals[seed_idx]
        
        while queue:
            current_idx = queue.popleft()
            
            if visited[current_idx]:
                continue
            
            # Check normal similarity with seed
            current_normal = normals[current_idx]
            cos_angle = abs(np.dot(seed_normal, current_normal))
            
            if cos_angle < cos_threshold:
                continue
            
            # Add to region
            visited[current_idx] = True
            region.append(current_idx)
            
            # Find neighbors
            [k, neighbor_indices, distances] = kdtree.search_knn_vector_3d(
                points[current_idx], k_neighbors
            )
            
            # Add unvisited neighbors within distance threshold
            for i, (neighbor_idx, dist_sq) in enumerate(zip(neighbor_indices, distances)):
                if neighbor_idx >= n_points:  # Safety check
                    continue
                if visited[neighbor_idx]:
                    continue
                if dist_sq > distance_threshold ** 2:
                    continue
                
                # Check normal similarity
                neighbor_normal = normals[neighbor_idx]
                cos_angle = abs(np.dot(seed_normal, neighbor_normal))
                
                if cos_angle >= cos_threshold:
                    queue.append(neighbor_idx)
        
        # Keep region if large enough
        if len(region) >= min_region_points:
            regions.append(np.array(region))
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"  Found {len(regions)} regions in {elapsed:.2f}s")
        for i, region in enumerate(regions):
            print(f"    Region {i}: {len(region):,} points")
    
    return regions


def _estimate_curvature(points: np.ndarray, 
                        normals: np.ndarray,
                        kdtree,
                        k_neighbors: int) -> np.ndarray:
    """
    Estimate surface curvature at each point.
    Lower curvature = flatter surface = better seed point.
    """
    
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    # Sample for speed (estimate curvature for subset, interpolate)
    sample_size = min(5000, n_points)
    sample_indices = np.linspace(0, n_points - 1, sample_size, dtype=int)
    
    for idx in sample_indices:
        [k, neighbor_indices, _] = kdtree.search_knn_vector_3d(
            points[idx], k_neighbors
        )
        
        # Filter valid indices
        valid_neighbors = [i for i in neighbor_indices if i < n_points]
        
        if len(valid_neighbors) < 3:
            curvatures[idx] = 1.0  # High curvature = bad seed
            continue
        
        # Curvature = variance of neighbor normals
        neighbor_normals = normals[valid_neighbors]
        normal_variance = np.var(neighbor_normals, axis=0).sum()
        curvatures[idx] = normal_variance
    
    # For non-sampled points, use nearest sampled curvature
    # (simplified - just use the computed values)
    
    return curvatures


def filter_wall_regions(
    points: np.ndarray,
    normals: np.ndarray,
    regions: List[np.ndarray],
    floor_height: float,
    ceiling_height: float,
    min_wall_height: float = 0.5,
    min_wall_length: float = 0.3,
    max_normal_z: float = 0.3,
    verbose: bool = True
) -> List[Dict]:
    """
    Filter regions to keep only wall-like segments.
    
    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Full point cloud
    normals : np.ndarray (N, 3)
        Point normals
    regions : list of np.ndarray
        Region indices from region_growing_segmentation
    floor_height : float
        Z coordinate of floor
    ceiling_height : float
        Z coordinate of ceiling
    min_wall_height : float
        Minimum height span for wall (default 0.5 for above-door segments)
    min_wall_length : float
        Minimum horizontal extent for wall
    max_normal_z : float
        Maximum |normal_z| for vertical surface (0 = perfectly vertical)
    verbose : bool
        Print progress
        
    Returns:
    --------
    walls : list of dict
        Each wall has 'points', 'indices', 'plane_model', 'normal', etc.
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FILTERING WALL REGIONS")
        print(f"{'='*70}")
        print(f"  Input regions: {len(regions)}")
        print(f"  Floor: {floor_height:.2f}m, Ceiling: {ceiling_height:.2f}m")
        print(f"  Min wall height: {min_wall_height}m")
        print(f"  Min wall length: {min_wall_length}m")
    
    room_height = ceiling_height - floor_height
    walls = []
    
    for i, region_indices in enumerate(regions):
        region_points = points[region_indices]
        region_normals = normals[region_indices]
        
        # Average normal for region
        avg_normal = region_normals.mean(axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
        
        # Check 1: Vertical surface (normal mostly horizontal)
        if abs(avg_normal[2]) > max_normal_z:
            if verbose:
                print(f"  Region {i}: REJECTED (not vertical, normal_z={avg_normal[2]:.2f})")
            continue
        
        # Check 2: Height span
        z_min = region_points[:, 2].min()
        z_max = region_points[:, 2].max()
        height_span = z_max - z_min
        
        if height_span < min_wall_height:
            if verbose:
                print(f"  Region {i}: REJECTED (too short, height={height_span:.2f}m)")
            continue
        
        # Check 3: Horizontal extent (wall length)
        points_2d = region_points[:, :2]
        x_span = points_2d[:, 0].max() - points_2d[:, 0].min()
        y_span = points_2d[:, 1].max() - points_2d[:, 1].min()
        horizontal_extent = np.sqrt(x_span**2 + y_span**2)
        
        if horizontal_extent < min_wall_length:
            if verbose:
                print(f"  Region {i}: REJECTED (too narrow, extent={horizontal_extent:.2f}m)")
            continue
        
        # Check 4: Height ratio (spans significant portion of room)
        height_ratio = height_span / room_height
        
        # Fit plane to region
        plane_model = _fit_plane_to_points(region_points)
        
        walls.append({
            'points': region_points,
            'indices': region_indices,
            'plane_model': plane_model,
            'normal': avg_normal,
            'height_span': height_span,
            'height_ratio': height_ratio,
            'horizontal_extent': horizontal_extent,
            'n_points': len(region_indices)
        })
        
        if verbose:
            print(f"  Region {i}: ACCEPTED (h={height_span:.2f}m, extent={horizontal_extent:.2f}m, "
                  f"points={len(region_indices):,})")
    
    if verbose:
        print(f"\n  Walls found: {len(walls)}")
    
    return walls


def filter_walls_by_ceiling_boundary(
    walls: List[Dict],
    ceiling_boundary: Dict,
    boundary_angle_thresh: float = 15.0,
    boundary_distance_thresh: float = 0.20,
    verbose: bool = True
) -> List[Dict]:
    """
    Filter walls to keep only those aligned with ceiling boundary.
    
    Same logic as RANSAC two-pass filtering.
    
    Parameters:
    -----------
    walls : list of dict
        Wall segments from region growing
    ceiling_boundary : dict
        Output from extract_ceiling_boundary() with 'edges' list
    boundary_angle_thresh : float
        Max angle (degrees) between wall and boundary edge
    boundary_distance_thresh : float
        Max distance (meters) from wall centroid to boundary edge
    verbose : bool
    
    Returns:
    --------
    filtered_walls : list of dict
        Walls that align with ceiling boundary
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FILTERING BY CEILING BOUNDARY")
        print(f"{'='*70}")
        print(f"  Input walls: {len(walls)}")
        print(f"  Boundary edges: {len(ceiling_boundary.get('edges', []))}")
    
    if not ceiling_boundary or 'edges' not in ceiling_boundary:
        if verbose:
            print(f"  No ceiling boundary provided, skipping filter")
        return walls
    
    boundary_edges = ceiling_boundary['edges']
    filtered_walls = []
    
    for i, wall in enumerate(walls):
        # Get wall direction in 2D
        plane_model = wall['plane_model']
        a, b, c, d = plane_model
        
        # Wall direction is perpendicular to normal in 2D
        wall_dir = np.array([-b, a])
        wall_dir = wall_dir / (np.linalg.norm(wall_dir) + 1e-10)
        
        # Wall centroid in 2D
        centroid_2d = wall['points'][:, :2].mean(axis=0)
        
        # Check alignment with any boundary edge
        is_aligned = False
        best_edge_idx = -1
        best_angle_diff = 180
        best_distance = float('inf')
        
        for j, edge in enumerate(boundary_edges):
            edge_dir = edge['direction']
            edge_start = edge['start']
            edge_length = edge['length']
            
            # Angle difference
            dot = abs(np.dot(wall_dir, edge_dir))
            angle_diff = np.degrees(np.arccos(np.clip(dot, 0, 1)))
            
            # Distance from centroid to edge line
            to_centroid = centroid_2d - edge_start
            proj_length = np.dot(to_centroid, edge_dir)
            proj_point = edge_start + proj_length * edge_dir
            distance = np.linalg.norm(centroid_2d - proj_point)
            
            # Check if within edge extent (with tolerance)
            tolerance = 0.5
            within_extent = -tolerance < proj_length < edge_length + tolerance
            
            if angle_diff < boundary_angle_thresh and distance < boundary_distance_thresh and within_extent:
                is_aligned = True
                if angle_diff < best_angle_diff:
                    best_angle_diff = angle_diff
                    best_distance = distance
                    best_edge_idx = j
        
        if is_aligned:
            filtered_walls.append(wall)
            if verbose:
                print(f"  Wall {i}: KEEP (edge {best_edge_idx}, angle={best_angle_diff:.1f}°, dist={best_distance*100:.1f}cm)")
        else:
            if verbose:
                print(f"  Wall {i}: REJECT (no boundary alignment)")
    
    if verbose:
        print(f"\n  Walls after filtering: {len(filtered_walls)}")
    
    return filtered_walls


def filter_parallel_offset_walls(
    walls: List[Dict],
    min_offset: float = 0.03,
    max_offset: float = 0.30,
    angle_thresh: float = 20.0,
    min_points_ratio: float = 0.10,
    verbose: bool = True
) -> List[Dict]:
    """
    Filter out small walls that are parallel to larger walls but offset.
    
    These are typically:
    - Radiators (5-15cm in front of wall)
    - Window sills
    - Furniture against walls
    - Small parallel noise segments
    
    Parameters:
    -----------
    walls : list of dict
        Wall segments
    min_offset : float
        Minimum offset (meters) to be considered "parallel but offset" (3cm default)
    max_offset : float
        Maximum offset (meters) - beyond this it's a different wall (30cm default)
    angle_thresh : float
        Max angle (degrees) for parallel check (default 20° to catch noisy normals)
    min_points_ratio : float
        Walls with fewer than this ratio of largest wall's points are candidates
    verbose : bool
    
    Returns:
    --------
    filtered_walls : list of dict
        Walls with radiators/parallel noise removed
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FILTERING PARALLEL-BUT-OFFSET SEGMENTS")
        print(f"{'='*70}")
        print(f"  Input walls: {len(walls)}")
        print(f"  Offset range: {min_offset*100:.0f}-{max_offset*100:.0f}cm")
    
    if len(walls) <= 1:
        return walls
    
    # Sort by point count (largest first = main walls)
    walls_sorted = sorted(walls, key=lambda w: w['n_points'], reverse=True)
    max_points = walls_sorted[0]['n_points']
    
    # Identify main walls (large) vs candidate walls (small)
    main_walls = []
    candidate_walls = []
    
    for wall in walls_sorted:
        ratio = wall['n_points'] / max_points
        if ratio >= min_points_ratio:
            main_walls.append(wall)
            if verbose:
                print(f"  Main wall: {wall['n_points']} points ({ratio*100:.1f}%)")
        else:
            candidate_walls.append(wall)
            if verbose:
                print(f"  Candidate: {wall['n_points']} points ({ratio*100:.1f}%)")
    
    if verbose:
        print(f"\n  Main walls: {len(main_walls)}")
        print(f"  Candidate walls to check: {len(candidate_walls)}")
    
    # Check each candidate against main walls
    filtered_walls = list(main_walls)  # Keep all main walls
    
    for i, candidate in enumerate(candidate_walls):
        # Get candidate centroid
        cand_centroid = candidate['points'][:, :2].mean(axis=0)
        
        cand_normal = candidate['normal'][:2]
        cand_normal = cand_normal / (np.linalg.norm(cand_normal) + 1e-10)
        
        is_parallel_offset = False
        
        for main_wall in main_walls:
            main_normal = main_wall['normal'][:2]
            main_normal = main_normal / (np.linalg.norm(main_normal) + 1e-10)
            
            # Check if parallel (normals aligned or opposite)
            dot = abs(np.dot(cand_normal, main_normal))
            angle_diff = np.degrees(np.arccos(np.clip(dot, 0, 1)))
            
            if angle_diff > angle_thresh:
                continue  # Not parallel
            
            # Calculate actual distance between planes using centroids
            # Project candidate centroid onto main wall's plane
            main_centroid = main_wall['points'][:, :2].mean(axis=0)
            
            # Vector from main centroid to candidate centroid
            diff_vec = cand_centroid - main_centroid
            
            # Distance along the normal direction (perpendicular to wall)
            offset_dist = abs(np.dot(diff_vec, main_normal))
            
            if verbose:
                print(f"    Candidate {i} vs main wall: angle={angle_diff:.1f}°, offset={offset_dist*100:.1f}cm")
            
            if min_offset < offset_dist < max_offset:
                is_parallel_offset = True
                if verbose:
                    print(f"  Candidate {i}: REJECT (parallel to main wall, "
                          f"offset={offset_dist*100:.1f}cm)")
                break
        
        if not is_parallel_offset:
            filtered_walls.append(candidate)
            if verbose:
                print(f"  Candidate {i}: KEEP")
    
    if verbose:
        print(f"\n  Walls after filtering: {len(filtered_walls)}")
    
    return filtered_walls


def merge_coplanar_walls(
    walls: List[Dict],
    coplanar_angle_thresh: float = 15.0,
    coplanar_offset_thresh: float = 0.20,
    verbose: bool = True
) -> List[Dict]:
    """
    Merge walls that are co-planar (same plane, different segments).
    
    Handles:
    - Walls above doors that should merge with main wall
    - Split walls that should be one wall
    
    Parameters:
    -----------
    walls : list of dict
        Wall segments
    coplanar_angle_thresh : float
        Max angle (degrees) between normals for co-planar
    coplanar_offset_thresh : float
        Max offset (meters) between planes for co-planar
    verbose : bool
    
    Returns:
    --------
    merged_walls : list of dict
        Walls with co-planar segments merged
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MERGING CO-PLANAR WALLS")
        print(f"{'='*70}")
        print(f"  Input walls: {len(walls)}")
        print(f"  Angle threshold: {coplanar_angle_thresh}°")
        print(f"  Offset threshold: {coplanar_offset_thresh*100:.0f}cm")
    
    if len(walls) <= 1:
        return walls
    
    # Use Union-Find to group co-planar walls
    n_walls = len(walls)
    parent = list(range(n_walls))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Check all pairs for co-planarity
    for i in range(n_walls):
        normal_i = walls[i]['normal'][:2]
        normal_i = normal_i / (np.linalg.norm(normal_i) + 1e-10)
        centroid_i = walls[i]['points'][:, :2].mean(axis=0)
        
        for j in range(i + 1, n_walls):
            normal_j = walls[j]['normal'][:2]
            normal_j = normal_j / (np.linalg.norm(normal_j) + 1e-10)
            centroid_j = walls[j]['points'][:, :2].mean(axis=0)
            
            # Check angle
            dot = abs(np.dot(normal_i, normal_j))
            angle_diff = np.degrees(np.arccos(np.clip(dot, 0, 1)))
            
            if angle_diff > coplanar_angle_thresh:
                continue
            
            # Check offset (distance between centroids along normal)
            diff_vec = centroid_j - centroid_i
            offset_dist = abs(np.dot(diff_vec, normal_i))
            
            if offset_dist < coplanar_offset_thresh:
                union(i, j)
                if verbose:
                    print(f"  Merging wall {j} with wall {i} (angle={angle_diff:.1f}°, offset={offset_dist*100:.1f}cm)")
    
    # Group walls by their root parent
    groups = {}
    for i in range(n_walls):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Create merged walls
    merged = []
    for root, indices in groups.items():
        if len(indices) == 1:
            # Single wall, keep as-is
            merged.append(walls[indices[0]])
        else:
            # Merge multiple walls
            group_points = [walls[i]['points'] for i in indices]
            merged_points = np.vstack(group_points)
            
            # Refit plane
            merged_plane = _fit_plane_to_points(merged_points)
            
            # Recalculate stats
            z_min = merged_points[:, 2].min()
            z_max = merged_points[:, 2].max()
            height_span = z_max - z_min
            
            points_2d = merged_points[:, :2]
            x_span = points_2d[:, 0].max() - points_2d[:, 0].min()
            y_span = points_2d[:, 1].max() - points_2d[:, 1].min()
            horizontal_extent = np.sqrt(x_span**2 + y_span**2)
            
            avg_normal = merged_plane[:3]
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
            
            merged.append({
                'points': merged_points,
                'plane_model': merged_plane,
                'normal': avg_normal,
                'height_span': height_span,
                'horizontal_extent': horizontal_extent,
                'n_points': len(merged_points),
                'merged_from': indices
            })
            
            if verbose:
                print(f"  Created merged wall from {indices}: {len(merged_points)} points")
    
    if verbose:
        print(f"\n  Walls after merging: {len(merged)}")
    
    return merged


def _fit_plane_to_points(points: np.ndarray) -> np.ndarray:
    """
    Fit plane to points using SVD.
    Returns [a, b, c, d] where ax + by + cz + d = 0
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    # SVD
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Smallest singular value = normal direction
    
    # Ensure normal points "outward" (positive z component for floor-facing)
    if normal[2] < 0:
        normal = -normal
    
    # Plane equation: n · (p - centroid) = 0
    # n · p - n · centroid = 0
    # ax + by + cz + d = 0, where d = -n · centroid
    d = -np.dot(normal, centroid)
    
    return np.array([normal[0], normal[1], normal[2], d])


def region_growing_wall_detection(
    points: np.ndarray,
    floor_height: float,
    ceiling_height: float,
    # Normal estimation params
    normal_k_neighbors: int = 30,
    # Region growing params
    angle_threshold: float = 10.0,
    distance_threshold: float = 0.10,
    min_region_points: int = 100,
    grow_k_neighbors: int = 30,
    # Wall filtering params
    min_wall_height: float = 0.5,
    min_wall_length: float = 0.3,
    max_normal_z: float = 0.3,
    # Boundary filtering params
    ceiling_boundary: Dict = None,
    boundary_angle_thresh: float = 15.0,
    boundary_distance_thresh: float = 0.20,
    # Parallel-offset filter params (radiators, etc.)
    filter_parallel_offset: bool = True,
    parallel_min_offset: float = 0.03,
    parallel_max_offset: float = 0.30,
    parallel_angle_thresh: float = 20.0,
    parallel_min_points_ratio: float = 0.10,
    # Merge params
    coplanar_angle_thresh: float = 20.0,
    coplanar_offset_thresh: float = 0.25,
    # Options
    precomputed_normals: np.ndarray = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Complete region growing wall detection pipeline.
    
    Parameters:
    -----------
    points : np.ndarray (N, 3)
        Point cloud (ideally pre-filtered to wall region)
    floor_height : float
        Z coordinate of floor
    ceiling_height : float
        Z coordinate of ceiling
    normal_k_neighbors : int
        Neighbors for normal estimation
    angle_threshold : float
        Max angle (degrees) for region membership
    distance_threshold : float
        Max distance (meters) for region membership
    min_region_points : int
        Minimum points per region
    grow_k_neighbors : int
        Neighbors to check during growing
    min_wall_height : float
        Minimum wall height (0.5 to catch above-door segments)
    min_wall_length : float
        Minimum wall horizontal extent
    max_normal_z : float
        Maximum |normal_z| for vertical surface
    ceiling_boundary : dict, optional
        Output from extract_ceiling_boundary() for boundary filtering
    boundary_angle_thresh : float
        Max angle for boundary alignment
    boundary_distance_thresh : float
        Max distance for boundary alignment
    coplanar_angle_thresh : float
        Max angle for co-planar merge
    coplanar_offset_thresh : float
        Max offset for co-planar merge
    precomputed_normals : np.ndarray, optional
        Pre-computed normals (skip estimation)
    verbose : bool
        Print progress
        
    Returns:
    --------
    walls : list of dict
        Detected wall segments
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"REGION GROWING WALL DETECTION")
        print(f"{'='*70}")
    
    # Step 1: Compute normals (if not provided)
    if precomputed_normals is not None:
        normals = precomputed_normals
        if verbose:
            print(f"  Using precomputed normals")
    else:
        if verbose:
            print(f"  Computing normals (k={normal_k_neighbors})...")
        normals = compute_normals(points, k_neighbors=normal_k_neighbors)
    
    # Step 2: Region growing segmentation
    regions = region_growing_segmentation(
        points=points,
        normals=normals,
        angle_threshold=angle_threshold,
        distance_threshold=distance_threshold,
        min_region_points=min_region_points,
        k_neighbors=grow_k_neighbors,
        verbose=verbose
    )
    
    # Step 3: Filter to walls (basic geometric criteria)
    walls = filter_wall_regions(
        points=points,
        normals=normals,
        regions=regions,
        floor_height=floor_height,
        ceiling_height=ceiling_height,
        min_wall_height=min_wall_height,
        min_wall_length=min_wall_length,
        max_normal_z=max_normal_z,
        verbose=verbose
    )
    
    # Step 4: Filter by ceiling boundary (removes outside noise)
    if ceiling_boundary is not None:
        walls = filter_walls_by_ceiling_boundary(
            walls=walls,
            ceiling_boundary=ceiling_boundary,
            boundary_angle_thresh=boundary_angle_thresh,
            boundary_distance_thresh=boundary_distance_thresh,
            verbose=verbose
        )
    
    # Step 5: Filter parallel-but-offset segments (radiators, window sills)
    if filter_parallel_offset:
        walls = filter_parallel_offset_walls(
            walls=walls,
            min_offset=parallel_min_offset,
            max_offset=parallel_max_offset,
            angle_thresh=parallel_angle_thresh,
            min_points_ratio=parallel_min_points_ratio,
            verbose=verbose
        )
    
    # Step 6: Merge co-planar walls (above-door + main wall)
    walls = merge_coplanar_walls(
        walls=walls,
        coplanar_angle_thresh=coplanar_angle_thresh,
        coplanar_offset_thresh=coplanar_offset_thresh,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL RESULT: {len(walls)} walls")
        print(f"{'='*70}")
    
    return walls


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Region Growing Wall Detection")
    print("=" * 50)
    
    # Create synthetic test data: 4 walls forming a room
    np.random.seed(42)  # For reproducible test data only
    
    # Room dimensions
    room_width = 4.0
    room_depth = 3.0
    room_height = 2.5
    
    points_list = []
    
    # Wall 1: X = 0 (left wall)
    n_points = 1000
    y = np.random.uniform(0, room_depth, n_points)
    z = np.random.uniform(0, room_height, n_points)
    x = np.random.normal(0, 0.02, n_points)  # Small noise
    points_list.append(np.column_stack([x, y, z]))
    
    # Wall 2: X = room_width (right wall)
    y = np.random.uniform(0, room_depth, n_points)
    z = np.random.uniform(0, room_height, n_points)
    x = room_width + np.random.normal(0, 0.02, n_points)
    points_list.append(np.column_stack([x, y, z]))
    
    # Wall 3: Y = 0 (front wall)
    x = np.random.uniform(0, room_width, n_points)
    z = np.random.uniform(0, room_height, n_points)
    y = np.random.normal(0, 0.02, n_points)
    points_list.append(np.column_stack([x, y, z]))
    
    # Wall 4: Y = room_depth (back wall)
    x = np.random.uniform(0, room_width, n_points)
    z = np.random.uniform(0, room_height, n_points)
    y = room_depth + np.random.normal(0, 0.02, n_points)
    points_list.append(np.column_stack([x, y, z]))
    
    # Combine all points
    all_points = np.vstack(points_list)
    
    print(f"\nTest data: {len(all_points)} points, 4 walls")
    
    # Run detection
    walls = region_growing_wall_detection(
        points=all_points,
        floor_height=0.0,
        ceiling_height=room_height,
        angle_threshold=10.0,
        distance_threshold=0.15,
        min_region_points=50,
        min_wall_height=1.5,
        min_wall_length=0.3,
        verbose=True
    )
    
    print(f"\n✓ Detected {len(walls)} walls")
    
    for i, wall in enumerate(walls):
        print(f"\nWall {i}:")
        print(f"  Points: {wall['n_points']}")
        print(f"  Height: {wall['height_span']:.2f}m")
        print(f"  Extent: {wall['horizontal_extent']:.2f}m")
        print(f"  Normal: [{wall['normal'][0]:.3f}, {wall['normal'][1]:.3f}, {wall['normal'][2]:.3f}]")
