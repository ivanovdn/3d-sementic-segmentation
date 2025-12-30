import numpy as np
from shapely.geometry import MultiPoint, LineString, Point
from shapely import concave_hull
from typing import List, Dict, Tuple, Optional


def extract_ceiling_boundary(ceiling_points_2d: np.ndarray, 
                              concavity_ratio: float = 0.3,
                              simplify_tolerance: float = 0.1) -> Dict:
    """
    Extract ceiling boundary using Shapely concave hull + simplification.
    
    Parameters:
    -----------
    ceiling_points_2d : np.ndarray (N, 2)
        2D projected ceiling points
    concavity_ratio : float
        Shapely concave_hull ratio parameter (0 = convex, 1 = very concave)
    simplify_tolerance : float
        Douglas-Peucker simplification tolerance in meters
        
    Returns:
    --------
    dict with:
        - polygon: Shapely Polygon (simplified)
        - edges: list of edge dicts with 'start', 'end', 'direction', 'length'
        - vertices: np.ndarray of boundary vertices
    """
    
    # Create MultiPoint and compute concave hull
    multipoint = MultiPoint(ceiling_points_2d)
    polygon = concave_hull(multipoint, ratio=concavity_ratio)
    
    # Simplify with Douglas-Peucker
    polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
    
    # Extract vertices (exterior ring)
    if polygon.geom_type == 'Polygon':
        coords = np.array(polygon.exterior.coords)
    else:
        raise ValueError(f"Expected Polygon, got {polygon.geom_type}")
    
    # Remove duplicate last vertex (closed polygon)
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    
    # Extract edges
    edges = []
    n_vertices = len(coords)
    
    for i in range(n_vertices):
        start = coords[i]
        end = coords[(i + 1) % n_vertices]
        
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length > 1e-6:
            direction = direction / length
        
        edges.append({
            'start': start,
            'end': end,
            'direction': direction,
            'length': length,
            'midpoint': (start + end) / 2
        })
    
    print(f"Ceiling boundary: {n_vertices} vertices, {len(edges)} edges")
    for i, edge in enumerate(edges):
        angle = np.degrees(np.arctan2(edge['direction'][1], edge['direction'][0]))
        print(f"  Edge {i}: length={edge['length']:.3f}m, angle={angle:.1f}°")
    
    return {
        'polygon': polygon,
        'edges': edges,
        'vertices': coords
    }


def compute_wall_line_2d(wall_points_2d: np.ndarray) -> Dict:
    """
    Fit a 2D line to wall points using PCA.
    
    Returns:
    --------
    dict with:
        - direction: unit vector along wall
        - normal: unit vector perpendicular to wall
        - centroid: center point
        - extent: (min_proj, max_proj) along direction
    """
    
    centroid = wall_points_2d.mean(axis=0)
    centered = wall_points_2d - centroid
    
    # PCA to find principal direction
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Principal direction (largest eigenvalue)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Ensure consistent direction (positive x component)
    if direction[0] < 0:
        direction = -direction
    
    # Normal (perpendicular)
    normal = np.array([-direction[1], direction[0]])
    
    # Extent along direction
    projections = centered @ direction
    extent = (projections.min(), projections.max())
    
    return {
        'direction': direction,
        'normal': normal,
        'centroid': centroid,
        'extent': extent,
        'length': extent[1] - extent[0]
    }


def angle_between_directions(dir1: np.ndarray, dir2: np.ndarray) -> float:
    """
    Compute angle between two directions (0-90°, ignores sign).
    """
    dot = np.abs(np.dot(dir1, dir2))
    dot = np.clip(dot, 0, 1)
    return np.degrees(np.arccos(dot))


def point_to_line_distance(point: np.ndarray, 
                           line_point: np.ndarray, 
                           line_dir: np.ndarray) -> float:
    """
    Compute perpendicular distance from point to infinite line.
    """
    # Vector from line point to target point
    v = point - line_point
    
    # Project onto line direction
    proj_length = np.dot(v, line_dir)
    proj = proj_length * line_dir
    
    # Perpendicular component
    perp = v - proj
    
    return np.linalg.norm(perp)


def wall_aligns_with_edge(wall_line: Dict, 
                          edge: Dict,
                          angle_thresh: float = 15.0,
                          distance_thresh: float = 0.20) -> Tuple[bool, Optional[Dict]]:
    """
    Check if wall aligns with a ceiling boundary edge.
    
    Parameters:
    -----------
    wall_line : dict from compute_wall_line_2d
    edge : dict from extract_ceiling_boundary
    angle_thresh : float (degrees)
    distance_thresh : float (meters)
    
    Returns:
    --------
    (is_aligned, match_info)
    """
    
    # Check angle alignment
    angle = angle_between_directions(wall_line['direction'], edge['direction'])
    
    if angle > angle_thresh:
        return False, None
    
    # Check distance (wall centroid to edge line)
    distance = point_to_line_distance(
        wall_line['centroid'],
        edge['start'],
        edge['direction']
    )
    
    if distance > distance_thresh:
        return False, None
    
    # Check overlap (wall should be near the edge, not far away)
    # Project wall centroid onto edge line
    v = wall_line['centroid'] - edge['start']
    proj_length = np.dot(v, edge['direction'])
    
    # Allow some tolerance beyond edge endpoints
    tolerance = 0.5  # 50cm beyond edge is OK
    
    if proj_length < -tolerance or proj_length > edge['length'] + tolerance:
        return False, None
    
    return True, {
        'angle_diff': angle,
        'distance': distance,
        'edge_idx': None  # Will be filled by caller
    }


def filter_walls_by_ceiling_boundary(walls: List[Dict],
                                      ceiling_boundary: Dict,
                                      angle_thresh: float = 15.0,
                                      distance_thresh: float = 0.20,
                                      verbose: bool = True) -> List[Dict]:
    """
    Filter walls: keep only those aligned with ceiling boundary edges.
    
    Parameters:
    -----------
    walls : list of wall dicts from RANSAC
    ceiling_boundary : dict from extract_ceiling_boundary
    angle_thresh : degrees
    distance_thresh : meters
    verbose : print details
    
    Returns:
    --------
    filtered_walls : list of wall dicts that align with boundary
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FILTERING WALLS BY CEILING BOUNDARY")
        print(f"{'='*70}")
        print(f"Input walls: {len(walls)}")
        print(f"Ceiling edges: {len(ceiling_boundary['edges'])}")
        print(f"Thresholds: angle={angle_thresh}°, distance={distance_thresh}m")
    
    edges = ceiling_boundary['edges']
    filtered_walls = []
    
    for wall in walls:
        # Get 2D wall points
        wall_points_3d = wall['points']
        wall_points_2d = wall_points_3d[:, :2]  # Project to XY
        
        # Compute wall line
        wall_line = compute_wall_line_2d(wall_points_2d)
        
        # Check alignment with each edge
        best_match = None
        best_edge_idx = None
        
        for edge_idx, edge in enumerate(edges):
            is_aligned, match_info = wall_aligns_with_edge(
                wall_line, edge, angle_thresh, distance_thresh
            )
            
            if is_aligned:
                if best_match is None or match_info['distance'] < best_match['distance']:
                    best_match = match_info
                    best_edge_idx = edge_idx
        
        wall_angle = np.degrees(np.arctan2(
            wall_line['direction'][1], 
            wall_line['direction'][0]
        ))
        
        if best_match is not None:
            # Wall aligns with boundary
            wall['matched_edge_idx'] = best_edge_idx
            wall['match_info'] = best_match
            wall['wall_line_2d'] = wall_line
            filtered_walls.append(wall)
            
            if verbose:
                print(f"\n  ✅ Wall {wall['wall_id']}: KEEP")
                print(f"     Length: {wall['length']:.2f}m, Angle: {wall_angle:.1f}°")
                print(f"     Matched edge {best_edge_idx}: "
                      f"angle_diff={best_match['angle_diff']:.1f}°, "
                      f"distance={best_match['distance']*100:.1f}cm")
        else:
            if verbose:
                print(f"\n  ❌ Wall {wall['wall_id']}: REJECT (no matching edge)")
                print(f"     Length: {wall['length']:.2f}m, Angle: {wall_angle:.1f}°")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULT: {len(filtered_walls)}/{len(walls)} walls kept")
        print(f"{'='*70}")
    
    return filtered_walls


def merge_coplanar_walls(walls: List[Dict],
                         angle_thresh: float = 5.0,
                         distance_thresh: float = 0.08,
                         verbose: bool = True) -> List[Dict]:
    """
    Merge walls that lie on the same plane (e.g., wall segments split by door).
    
    Parameters:
    -----------
    walls : list of wall dicts
    angle_thresh : degrees - max angle difference between normals
    distance_thresh : meters - max distance between planes
    
    Returns:
    --------
    merged_walls : list of merged wall dicts
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MERGING CO-PLANAR WALLS")
        print(f"{'='*70}")
    
    if len(walls) == 0:
        return []
    
    # Track which walls have been merged
    merged_flags = [False] * len(walls)
    merged_walls = []
    
    for i, wall_i in enumerate(walls):
        if merged_flags[i]:
            continue
        
        # Start a new merged wall group
        group = [wall_i]
        merged_flags[i] = True
        
        normal_i = wall_i['normal'][:2]  # 2D normal
        normal_i = normal_i / np.linalg.norm(normal_i)
        
        # Plane offset (distance from origin)
        center_i = wall_i['center'][:2]
        offset_i = np.dot(normal_i, center_i)
        
        for j, wall_j in enumerate(walls):
            if merged_flags[j]:
                continue
            
            normal_j = wall_j['normal'][:2]
            normal_j = normal_j / np.linalg.norm(normal_j)
            
            center_j = wall_j['center'][:2]
            offset_j = np.dot(normal_j, center_j)
            
            # Check angle (normals should be parallel)
            angle = angle_between_directions(normal_i, normal_j)
            
            if angle > angle_thresh:
                continue
            
            # Check offset (same plane)
            # Handle sign flip (normals could be opposite)
            offset_diff = min(
                abs(offset_i - offset_j),
                abs(offset_i + offset_j)
            )
            
            if offset_diff > distance_thresh:
                continue
            
            # Merge!
            group.append(wall_j)
            merged_flags[j] = True
            
            if verbose:
                print(f"  Merging wall {wall_j['wall_id']} into wall {wall_i['wall_id']}")
                print(f"    Angle diff: {angle:.2f}°, Offset diff: {offset_diff*100:.1f}cm")
        
        # Create merged wall
        if len(group) == 1:
            merged_walls.append(group[0])
        else:
            merged_wall = merge_wall_group(group)
            merged_walls.append(merged_wall)
            
            if verbose:
                print(f"  → Merged wall {merged_wall['wall_id']}: "
                      f"{merged_wall['num_points']} points, "
                      f"{merged_wall['length']:.2f}m")
    
    if verbose:
        print(f"\nResult: {len(walls)} walls → {len(merged_walls)} walls")
    
    return merged_walls


def merge_wall_group(walls: List[Dict]) -> Dict:
    """
    Merge a group of walls into single wall.
    """
    
    # Combine points
    all_points = np.vstack([w['points'] for w in walls])
    all_indices = np.concatenate([w['wall_indices'] for w in walls])
    
    # Use first wall as base, update with combined data
    merged = walls[0].copy()
    
    merged['points'] = all_points
    merged['wall_indices'] = all_indices
    merged['num_points'] = len(all_points)
    merged['center'] = all_points.mean(axis=0)
    
    # Recompute geometry
    points_2d = all_points[:, :2]
    wall_line = compute_wall_line_2d(points_2d)
    merged['length'] = wall_line['length']
    
    # Update wall_id to indicate merge
    merged['wall_id'] = walls[0]['wall_id']
    merged['merged_from'] = [w['wall_id'] for w in walls]
    
    return merged


# =============================================================================
# VISUALIZATION (optional)
# =============================================================================

def plot_filtering_result(ceiling_points_2d: np.ndarray,
                          ceiling_boundary: Dict,
                          all_walls: List[Dict],
                          filtered_walls: List[Dict],
                          save_path: Optional[str] = None):
    """
    Visualize ceiling boundary and wall filtering result.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Before filtering
    ax1 = axes[0]
    ax1.set_title(f"Before Filtering ({len(all_walls)} walls)")
    
    # Ceiling points
    ax1.scatter(ceiling_points_2d[:, 0], ceiling_points_2d[:, 1], 
                s=1, alpha=0.3, c='lightblue', label='Ceiling')
    
    # Ceiling boundary
    vertices = ceiling_boundary['vertices']
    vertices_closed = np.vstack([vertices, vertices[0]])
    ax1.plot(vertices_closed[:, 0], vertices_closed[:, 1], 
             'b-', linewidth=2, label='Boundary')
    
    # All walls
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_walls)))
    for wall, color in zip(all_walls, colors):
        pts = wall['points'][:, :2]
        ax1.scatter(pts[:, 0], pts[:, 1], s=3, c=[color], 
                    label=f"Wall {wall['wall_id']}")
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Right: After filtering
    ax2 = axes[1]
    ax2.set_title(f"After Filtering ({len(filtered_walls)} walls)")
    
    # Ceiling boundary
    ax2.plot(vertices_closed[:, 0], vertices_closed[:, 1], 
             'b-', linewidth=2, label='Boundary')
    
    # Mark edges
    for i, edge in enumerate(ceiling_boundary['edges']):
        mid = edge['midpoint']
        ax2.annotate(f'E{i}', mid, fontsize=8, ha='center')
    
    # Filtered walls
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_walls)))
    for wall, color in zip(filtered_walls, colors):
        pts = wall['points'][:, :2]
        ax2.scatter(pts[:, 0], pts[:, 1], s=3, c=[color], 
                    label=f"Wall {wall['wall_id']} → E{wall.get('matched_edge_idx', '?')}")
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def filter_walls_pipeline(walls: List[Dict],
                          ceiling_points_2d: np.ndarray,
                          concavity_ratio: float = 0.3,
                          simplify_tolerance: float = 0.15,
                          angle_thresh: float = 15.0,
                          distance_thresh: float = 0.20,
                          merge_walls: bool = True,
                          verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Complete pipeline: ceiling boundary → filter walls → merge co-planar.
    
    Parameters:
    -----------
    walls : list of wall dicts from RANSAC
    ceiling_points_2d : np.ndarray (N, 2)
    concavity_ratio : for Shapely concave_hull
    simplify_tolerance : for Douglas-Peucker simplification (meters)
    angle_thresh : degrees for alignment check
    distance_thresh : meters for alignment check
    merge_walls : whether to merge co-planar walls
    verbose : print details
    
    Returns:
    --------
    (filtered_walls, ceiling_boundary)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"WALL FILTERING PIPELINE")
        print(f"{'='*70}")
    
    # Step 1: Extract ceiling boundary
    ceiling_boundary = extract_ceiling_boundary(
        ceiling_points_2d, 
        concavity_ratio=concavity_ratio,
        simplify_tolerance=simplify_tolerance
    )
    
    # Step 2: Filter walls by boundary
    filtered_walls = filter_walls_by_ceiling_boundary(
        walls,
        ceiling_boundary,
        angle_thresh=angle_thresh,
        distance_thresh=distance_thresh,
        verbose=verbose
    )
    
    # Step 3: Merge co-planar walls (optional)
    if merge_walls and len(filtered_walls) > 0:
        filtered_walls = merge_coplanar_walls(
            filtered_walls,
            verbose=verbose
        )
    
    return filtered_walls, ceiling_boundary


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Create synthetic test data
    np.random.seed(42)
    
    # Simulated ceiling boundary (pentagon like your room)
    true_vertices = np.array([
        [1.8, -2.2],
        [-1.8, -2.2],
        [-1.8, 2.2],
        [1.0, 2.2],
        [1.8, 1.5],
    ])
    
    # Generate ceiling points along boundary
    ceiling_points = []
    for i in range(len(true_vertices)):
        start = true_vertices[i]
        end = true_vertices[(i + 1) % len(true_vertices)]
        
        # Points along edge
        t = np.linspace(0, 1, 100)
        edge_points = start + np.outer(t, end - start)
        
        # Add noise
        noise = np.random.randn(len(edge_points), 2) * 0.02
        edge_points += noise
        
        ceiling_points.append(edge_points)
    
    ceiling_points_2d = np.vstack(ceiling_points)
    
    # Test boundary extraction
    boundary = extract_ceiling_boundary(ceiling_points_2d, 
                                         concavity_ratio=0.3,
                                         simplify_tolerance=0.15)
    
    print(f"\nExtracted {len(boundary['vertices'])} vertices")
    print(f"Original had {len(true_vertices)} vertices")
