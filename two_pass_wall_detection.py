import numpy as np
from typing import List, Dict, Tuple, Optional

# Try to import ceiling boundary functions
try:
    from wall_boundary_filter import (
        extract_ceiling_boundary,
        filter_walls_by_ceiling_boundary,
        compute_wall_line_2d as boundary_compute_wall_line_2d
    )
    BOUNDARY_FILTER_AVAILABLE = True
except ImportError:
    BOUNDARY_FILTER_AVAILABLE = False


def two_pass_wall_detection(
    vertical_points: np.ndarray,
    floor_height: float,
    ceiling_height: float,
    detect_walls_func,  # Your detect_walls_ransac_clustering or ensemble function
    strict_config: Dict,
    loose_config: Dict,
    ceiling_points_2d: Optional[np.ndarray] = None,  # NEW: for boundary filtering
    ceiling_boundary: Optional[Dict] = None,  # NEW: pre-computed boundary
    boundary_angle_thresh: float = 15.0,  # NEW
    boundary_distance_thresh: float = 0.20,  # NEW
    coplanar_angle_thresh: float = 5.0,
    coplanar_offset_thresh: float = 0.08,
    use_config_kwarg: bool = True,  # True for ensemble_ransac, False for detect_walls_ransac_clustering
    verbose: bool = True
) -> List[Dict]:
    """
    Two-pass wall detection with ceiling boundary filter and co-planar merge.
    
    Pass 1: Strict params → main walls (floor-to-ceiling)
    Pass 2: Loose params on remaining points → small segments
    Filter: Keep segments that align with ceiling boundary (reject outside room)
    Merge: Keep small segments only if co-planar with main wall
    
    Parameters:
    -----------
    vertical_points : np.ndarray (N, 3)
        All vertical surface points
    floor_height : float
    ceiling_height : float
    detect_walls_func : callable
        Your detect_walls_ransac_clustering or ensemble_ransac_wall_detection_v2
    strict_config : dict
        Config for Pass 1 (strict params)
    loose_config : dict
        Config for Pass 2 (loose params)
    ceiling_points_2d : np.ndarray (M, 2), optional
        2D projected ceiling points for boundary extraction
    ceiling_boundary : dict, optional
        Pre-computed ceiling boundary (from extract_ceiling_boundary)
        If provided, ceiling_points_2d is ignored
    boundary_angle_thresh : float
        Max angle between segment and boundary edge (degrees)
    boundary_distance_thresh : float
        Max distance from segment to boundary edge (meters)
    coplanar_angle_thresh : float
        Max angle between normals to consider co-planar (degrees)
    coplanar_offset_thresh : float
        Max plane offset difference to consider co-planar (meters)
    use_config_kwarg : bool
        If True: detect_walls_func(..., config=config)
        If False: detect_walls_func(..., **config)
    verbose : bool
    
    Returns:
    --------
    final_walls : list of wall dicts
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TWO-PASS WALL DETECTION")
        print(f"{'='*70}")
        print(f"Total vertical points: {len(vertical_points):,}")
        print(f"Room height: {ceiling_height - floor_height:.2f}m")
    
    # ========================================================================
    # PASS 1: Strict params → Main walls
    # ========================================================================
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PASS 1: STRICT PARAMS (main walls)")
        print(f"{'='*70}")
        print(f"Config: min_wall_height={strict_config.get('min_wall_height', 'N/A')}, "
              f"min_height_ratio={strict_config.get('min_height_ratio', 'N/A')}")
    
    if use_config_kwarg:
        # For ensemble_ransac_wall_detection_v2
        # Extract n_runs if present, default to 5
        n_runs = strict_config.pop('n_runs', 5)
        main_walls = detect_walls_func(
            vertical_points,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            n_runs=n_runs,
            config=strict_config
        )
        # Put n_runs back in case config is reused
        strict_config['n_runs'] = n_runs
    else:
        main_walls = detect_walls_func(
            vertical_points,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            **strict_config
        )
    
    if verbose:
        print(f"\n→ Pass 1 detected {len(main_walls)} main walls")
        for wall in main_walls:
            print(f"   Wall {wall['wall_id']}: {wall['num_points']:,} pts, "
                  f"L={wall['length']:.2f}m, H={wall['height']:.2f}m")
    
    if len(main_walls) == 0:
        if verbose:
            print("⚠️  No main walls detected, returning empty")
        return []
    
    # ========================================================================
    # Remove main wall points from pool
    # ========================================================================
    
    # Collect all indices used by main walls
    main_wall_indices = set()
    for wall in main_walls:
        if 'wall_indices' in wall:
            main_wall_indices.update(wall['wall_indices'])
    
    # Create mask for remaining points
    all_indices = np.arange(len(vertical_points))
    remaining_mask = ~np.isin(all_indices, list(main_wall_indices))
    remaining_points = vertical_points[remaining_mask]
    remaining_original_indices = all_indices[remaining_mask]
    
    if verbose:
        print(f"\n→ Removed {len(main_wall_indices):,} points used by main walls")
        print(f"→ Remaining points: {len(remaining_points):,}")
    
    if len(remaining_points) < loose_config.get('min_cluster_points', 50):
        if verbose:
            print("→ Not enough remaining points for Pass 2")
        return main_walls
    
    # ========================================================================
    # PASS 2: Loose params → Small segments
    # ========================================================================
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PASS 2: LOOSE PARAMS (small segments)")
        print(f"{'='*70}")
        print(f"Config: min_wall_height={loose_config.get('min_wall_height', 'N/A')}, "
              f"min_height_ratio={loose_config.get('min_height_ratio', 'N/A')}")
    
    if use_config_kwarg:
        # For ensemble_ransac_wall_detection_v2
        # Extract n_runs if present, default to 5
        n_runs = loose_config.pop('n_runs', 5)
        small_segments = detect_walls_func(
            remaining_points,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            n_runs=n_runs,
            config=loose_config
        )
        # Put n_runs back in case config is reused
        loose_config['n_runs'] = n_runs
    else:
        small_segments = detect_walls_func(
            remaining_points,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            **loose_config
        )
    
    if verbose:
        print(f"\n→ Pass 2 detected {len(small_segments)} small segments")
        for seg in small_segments:
            print(f"   Segment {seg['wall_id']}: {seg['num_points']:,} pts, "
                  f"L={seg['length']:.2f}m, H={seg['height']:.2f}m")
    
    if len(small_segments) == 0:
        if verbose:
            print("→ No small segments found")
        return main_walls
    
    # ========================================================================
    # Fix indices: map back to original vertical_points
    # ========================================================================
    
    for seg in small_segments:
        if 'wall_indices' in seg:
            # These indices are into remaining_points
            # Map back to original vertical_points indices
            local_indices = seg['wall_indices']
            original_indices = remaining_original_indices[local_indices]
            seg['wall_indices'] = original_indices
            # Also update points from original array
            seg['points'] = vertical_points[original_indices]
    
    # ========================================================================
    # FILTER BY CEILING BOUNDARY (reject segments outside room)
    # ========================================================================
    
    if ceiling_points_2d is not None or ceiling_boundary is not None:
        if verbose:
            print(f"\n{'='*70}")
            print(f"FILTER: CEILING BOUNDARY CHECK")
            print(f"{'='*70}")
        
        # Extract boundary if not provided
        if ceiling_boundary is None:
            if not BOUNDARY_FILTER_AVAILABLE:
                print("⚠️  wall_boundary_filter module not available, skipping boundary filter")
            else:
                ceiling_boundary = extract_ceiling_boundary(
                    ceiling_points_2d,
                    concavity_ratio=0.3,
                    simplify_tolerance=0.15
                )
        
        if ceiling_boundary is not None:
            # Filter segments by boundary
            boundary_filtered_segments = filter_segments_by_boundary(
                small_segments,
                ceiling_boundary,
                angle_thresh=boundary_angle_thresh,
                distance_thresh=boundary_distance_thresh,
                main_walls=main_walls,  # NEW: for offset check
                offset_thresh=0.15,     # Reject if 5-15cm offset from main wall
                verbose=verbose
            )
            
            if verbose:
                print(f"\n→ Boundary filter: {len(small_segments)} → {len(boundary_filtered_segments)} segments")
            
            small_segments = boundary_filtered_segments
    
    if len(small_segments) == 0:
        if verbose:
            print("→ No segments remaining after boundary filter")
        return main_walls
    
    # ========================================================================
    # MERGE: Check co-planarity and merge, OR keep as separate wall
    # ========================================================================
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MERGE / KEEP DECISION")
        print(f"{'='*70}")
        print(f"Thresholds: coplanar_angle={coplanar_angle_thresh}°, coplanar_offset={coplanar_offset_thresh*100:.1f}cm")
    
    final_walls = list(main_walls)  # Start with main walls
    
    for seg in small_segments:
        # First, try co-planar merge with existing main wall
        match = find_coplanar_wall(
            seg, 
            main_walls,
            angle_thresh=coplanar_angle_thresh,
            offset_thresh=coplanar_offset_thresh,
            verbose=verbose
        )
        
        if match is not None:
            # Merge segment into matching main wall
            wall_idx, match_info = match
            
            if verbose:
                print(f"\n  ✅ Segment {seg['wall_id']} → MERGE into Wall {main_walls[wall_idx]['wall_id']}")
                print(f"     Reason: Co-planar (angle={match_info['angle_diff']:.2f}°, "
                      f"offset={match_info['offset_diff']*100:.1f}cm)")
            
            # Find this wall in final_walls and merge
            for fw in final_walls:
                if fw['wall_id'] == main_walls[wall_idx]['wall_id']:
                    merge_segment_into_wall(fw, seg)
                    break
        
        elif 'matched_boundary_edge' in seg:
            # Not co-planar, but aligns with ceiling boundary
            # Keep as separate wall (e.g., chamfer wall)
            if verbose:
                print(f"\n  ✅ Segment {seg['wall_id']} → KEEP as separate wall")
                print(f"     Reason: Aligns with boundary edge {seg['matched_boundary_edge']} "
                      f"(not co-planar with main walls)")
            
            final_walls.append(seg)
        
        else:
            # Neither co-planar nor boundary-aligned
            if verbose:
                print(f"\n  ❌ Segment {seg['wall_id']} → REJECT")
                print(f"     Reason: Not co-planar with any main wall, not on boundary")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TWO-PASS COMPLETE")
        print(f"{'='*70}")
        print(f"Main walls (Pass 1): {len(main_walls)}")
        print(f"Small segments (Pass 2): {len(small_segments)}")
        print(f"Final walls: {len(final_walls)}")
        
        for wall in final_walls:
            merged_info = ""
            if 'merged_segments' in wall:
                merged_info = f" (merged {len(wall['merged_segments'])} segments)"
            print(f"   Wall {wall['wall_id']}: {wall['num_points']:,} pts, "
                  f"L={wall['length']:.2f}m{merged_info}")
    
    return final_walls


def filter_segments_by_boundary(
    segments: List[Dict],
    ceiling_boundary: Dict,
    angle_thresh: float = 15.0,
    distance_thresh: float = 0.20,
    main_walls: Optional[List[Dict]] = None,  # NEW: to check offset
    offset_thresh: float = 0.15,  # NEW: max offset from main wall plane
    verbose: bool = True
) -> List[Dict]:
    """
    Filter segments: keep only those aligned with ceiling boundary edges.
    Also rejects segments that are parallel to a main wall but offset (like radiator).
    
    Parameters:
    -----------
    segments : list of segment dicts
    ceiling_boundary : dict with 'edges' key
    angle_thresh : degrees
    distance_thresh : meters
    main_walls : list of main wall dicts (optional, for offset check)
    offset_thresh : meters - reject if parallel to main wall but offset more than this
    verbose : bool
    
    Returns:
    --------
    filtered_segments : list of segments that align with boundary
    """
    
    edges = ceiling_boundary['edges']
    filtered_segments = []
    
    for seg in segments:
        # Get 2D segment points
        seg_points_3d = seg['points']
        seg_points_2d = seg_points_3d[:, :2]
        
        # Compute segment line
        seg_line = compute_wall_line_2d(seg_points_2d)
        
        # NEW: Check if segment is parallel but offset from any main wall
        # This catches radiators, window sills, etc.
        if main_walls is not None:
            is_offset_from_main = check_parallel_but_offset(
                seg, main_walls, 
                angle_thresh=10.0,  # Must be parallel
                min_offset=0.05,    # At least 5cm offset
                max_offset=offset_thresh  # But not too far (would be different wall)
            )
            
            if is_offset_from_main:
                if verbose:
                    print(f"  ❌ Segment {seg['wall_id']}: REJECT (parallel but offset from main wall - likely radiator/sill)")
                continue
        
        # Check alignment with each edge
        is_aligned = False
        best_edge_idx = None
        best_distance = float('inf')
        
        for edge_idx, edge in enumerate(edges):
            aligned, match_info = segment_aligns_with_edge(
                seg_line, edge, angle_thresh, distance_thresh
            )
            
            if aligned and match_info['distance'] < best_distance:
                is_aligned = True
                best_edge_idx = edge_idx
                best_distance = match_info['distance']
        
        seg_angle = np.degrees(np.arctan2(
            seg_line['direction'][1],
            seg_line['direction'][0]
        ))
        
        if is_aligned:
            seg['matched_boundary_edge'] = best_edge_idx
            filtered_segments.append(seg)
            
            if verbose:
                print(f"  ✅ Segment {seg['wall_id']}: KEEP (matches edge {best_edge_idx}, "
                      f"dist={best_distance*100:.1f}cm)")
        else:
            if verbose:
                print(f"  ❌ Segment {seg['wall_id']}: REJECT (no matching boundary edge)")
    
    return filtered_segments


def check_parallel_but_offset(
    segment: Dict,
    main_walls: List[Dict],
    angle_thresh: float = 10.0,
    min_offset: float = 0.05,
    max_offset: float = 0.15
) -> bool:
    """
    Check if segment is parallel to a main wall but offset (like radiator).
    
    Returns True if segment should be REJECTED.
    """
    
    seg_normal = segment['normal'][:2]
    seg_normal = seg_normal / np.linalg.norm(seg_normal)
    seg_center = segment['center'][:2]
    seg_offset = np.dot(seg_normal, seg_center)
    
    for wall in main_walls:
        wall_normal = wall['normal'][:2]
        wall_normal = wall_normal / np.linalg.norm(wall_normal)
        wall_center = wall['center'][:2]
        wall_offset = np.dot(wall_normal, wall_center)
        
        # Check if parallel
        dot = np.abs(np.dot(seg_normal, wall_normal))
        angle = np.degrees(np.arccos(np.clip(dot, 0, 1)))
        
        if angle > angle_thresh:
            continue  # Not parallel
        
        # Check offset difference
        offset_diff = min(
            abs(seg_offset - wall_offset),
            abs(seg_offset + wall_offset)
        )
        
        # If parallel and offset is in "suspicious" range, reject
        if min_offset < offset_diff < max_offset:
            return True  # Likely radiator/sill
    
    return False


def segment_aligns_with_edge(
    seg_line: Dict,
    edge: Dict,
    angle_thresh: float = 15.0,
    distance_thresh: float = 0.20
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if segment aligns with a ceiling boundary edge.
    """
    
    # Check angle alignment
    seg_dir = seg_line['direction']
    edge_dir = edge['direction']
    
    dot = np.abs(np.dot(seg_dir, edge_dir))
    dot = np.clip(dot, 0, 1)
    angle = np.degrees(np.arccos(dot))
    
    if angle > angle_thresh:
        return False, None
    
    # Check distance (segment centroid to edge line)
    seg_centroid = seg_line['centroid']
    edge_start = edge['start']
    
    # Vector from edge start to segment centroid
    v = seg_centroid - edge_start
    
    # Project onto edge direction
    proj_length = np.dot(v, edge_dir)
    proj = proj_length * edge_dir
    
    # Perpendicular distance
    perp = v - proj
    distance = np.linalg.norm(perp)
    
    if distance > distance_thresh:
        return False, None
    
    # Check overlap (segment should be near edge, not far away along its length)
    tolerance = 0.5  # 50cm beyond edge is OK
    if proj_length < -tolerance or proj_length > edge['length'] + tolerance:
        return False, None
    
    return True, {
        'angle_diff': angle,
        'distance': distance
    }


def find_coplanar_wall(
    segment: Dict,
    main_walls: List[Dict],
    angle_thresh: float = 5.0,
    offset_thresh: float = 0.08,
    verbose: bool = False
) -> Optional[Tuple[int, Dict]]:
    """
    Find main wall that is co-planar with segment.
    
    Returns:
    --------
    (wall_index, match_info) if found, None otherwise
    """
    
    seg_normal = segment['normal'][:2]  # 2D normal (XY plane)
    seg_normal = seg_normal / np.linalg.norm(seg_normal)
    
    seg_center = segment['center'][:2]
    seg_offset = np.dot(seg_normal, seg_center)
    
    best_match = None
    best_score = float('inf')
    
    for idx, wall in enumerate(main_walls):
        wall_normal = wall['normal'][:2]
        wall_normal = wall_normal / np.linalg.norm(wall_normal)
        
        wall_center = wall['center'][:2]
        wall_offset = np.dot(wall_normal, wall_center)
        
        # Check angle (normals should be parallel or anti-parallel)
        dot = np.dot(seg_normal, wall_normal)
        angle_diff = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
        
        if angle_diff > angle_thresh:
            continue
        
        # Check offset (same plane)
        # Handle sign flip (normals could point opposite directions)
        offset_diff = min(
            abs(seg_offset - wall_offset),
            abs(seg_offset + wall_offset)
        )
        
        if offset_diff > offset_thresh:
            continue
        
        # Score: prefer smaller offset difference
        score = offset_diff
        
        if score < best_score:
            best_score = score
            best_match = (idx, {
                'angle_diff': angle_diff,
                'offset_diff': offset_diff
            })
    
    return best_match


def merge_segment_into_wall(wall: Dict, segment: Dict) -> None:
    """
    Merge segment into wall (in-place).
    """
    
    # Combine points
    combined_points = np.vstack([wall['points'], segment['points']])
    wall['points'] = combined_points
    wall['num_points'] = len(combined_points)
    
    # Combine indices
    if 'wall_indices' in wall and 'wall_indices' in segment:
        combined_indices = np.concatenate([wall['wall_indices'], segment['wall_indices']])
        wall['wall_indices'] = combined_indices
    
    # Update center
    wall['center'] = combined_points.mean(axis=0)
    
    # Track merged segments
    if 'merged_segments' not in wall:
        wall['merged_segments'] = []
    wall['merged_segments'].append(segment['wall_id'])
    
    # Recompute length (2D extent along wall direction)
    points_2d = combined_points[:, :2]
    wall_line = compute_wall_line_2d(points_2d)
    wall['length'] = wall_line['length']


def compute_wall_line_2d(wall_points_2d: np.ndarray) -> Dict:
    """
    Fit a 2D line to wall points using PCA.
    """
    
    centroid = wall_points_2d.mean(axis=0)
    centered = wall_points_2d - centroid
    
    # PCA to find principal direction
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Principal direction (largest eigenvalue)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Extent along direction
    projections = centered @ direction
    extent = (projections.min(), projections.max())
    
    return {
        'direction': direction,
        'centroid': centroid,
        'extent': extent,
        'length': extent[1] - extent[0]
    }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_default_configs(room_height: float) -> Tuple[Dict, Dict]:
    """
    Create default strict and loose configs based on room height.
    
    Returns:
    --------
    (strict_config, loose_config)
    """
    
    strict_config = {
        'distance_threshold': 0.03,
        'cluster_eps': 0.15,
        'min_cluster_points': 100,
        'min_wall_height': max(1.5, room_height * 0.5),  # At least 50% of room
        'min_wall_length': 0.50,
        'max_wall_thickness': 0.25,
        'min_height_ratio': 0.50,
        'max_iterations': 30
    }
    
    loose_config = {
        'distance_threshold': 0.03,
        'cluster_eps': 0.15,
        'min_cluster_points': 50,
        'min_wall_height': 0.30,  # Can be very short
        'min_wall_length': 0.20,
        'max_wall_thickness': 0.25,
        'min_height_ratio': 0.05,  # Can be small ratio
        'max_iterations': 20
    }
    
    return strict_config, loose_config


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Two-pass wall detection module")
    print("=" * 50)
    print("\nUsage:")
    print("""
    from two_pass_wall_detection import two_pass_wall_detection, create_default_configs
    
    # Create configs
    room_height = ceiling_height - floor_height
    strict_config, loose_config = create_default_configs(room_height)
    
    # Run two-pass detection
    final_walls = two_pass_wall_detection(
        vertical_points,
        floor_height,
        ceiling_height,
        detect_walls_func=detect_walls_ransac_clustering,  # Your function
        strict_config=strict_config,
        loose_config=loose_config,
        coplanar_angle_thresh=5.0,    # degrees
        coplanar_offset_thresh=0.08,  # meters (8cm)
        verbose=True
    )
    """)
    
    # Quick test of helper functions
    print("\nTesting compute_wall_line_2d...")
    test_points = np.array([
        [0, 0],
        [1, 0.1],
        [2, -0.05],
        [3, 0.08],
        [4, 0]
    ])
    
    result = compute_wall_line_2d(test_points)
    print(f"  Direction: [{result['direction'][0]:.3f}, {result['direction'][1]:.3f}]")
    print(f"  Length: {result['length']:.3f}m")
    print(f"  Expected: ~4m along X axis")
    
    print("\n✓ Module ready")
