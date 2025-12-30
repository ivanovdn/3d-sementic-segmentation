"""
Between-Walls Opening Detection

Detects doorways and passages that exist as gaps BETWEEN walls,
not as holes within a single wall.

Approach:
1. Extract wall line segments (2D, top-down view)
2. Find wall endpoints
3. Detect gaps between endpoints that are:
   - Close enough to be connected (< max_gap)
   - But not touching (> min_gap)
4. Classify gaps by size (door, passage, etc.)
5. Measure opening dimensions

"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def extract_wall_line_2d(wall: Dict) -> Dict:
    """
    Extract 2D line segment from wall points.
    
    Parameters:
    -----------
    wall : dict
        Wall with 'points' (N, 3)
        
    Returns:
    --------
    line : dict
        Line segment with endpoints, direction, length
    """
    
    points = wall['points']
    points_2d = points[:, :2]  # XY only
    
    # Get wall direction from normal or PCA
    if 'normal' in wall:
        normal = wall['normal'][:2]
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        direction = np.array([-normal[1], normal[0]])  # Perpendicular to normal
    else:
        # Use PCA to find main direction
        centroid = points_2d.mean(axis=0)
        centered = points_2d - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Project points onto direction to find endpoints
    centroid = points_2d.mean(axis=0)
    projections = np.dot(points_2d - centroid, direction)
    
    min_proj = projections.min()
    max_proj = projections.max()
    
    # Endpoints
    p1 = centroid + min_proj * direction
    p2 = centroid + max_proj * direction
    
    length = np.linalg.norm(p2 - p1)
    
    # Also get height info
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    
    return {
        'p1': p1,
        'p2': p2,
        'direction': direction,
        'normal': np.array([-direction[1], direction[0]]),
        'centroid': centroid,
        'length': length,
        'z_min': z_min,
        'z_max': z_max,
        'n_points': len(points)
    }


def find_wall_endpoint_gaps(wall_lines: List[Dict],
                            min_gap: float = 0.5,
                            max_gap: float = 2.0,
                            max_perpendicular_offset: float = 0.3,
                            verbose: bool = False) -> List[Dict]:
    """
    Find gaps between wall endpoints that could be openings.
    
    Parameters:
    -----------
    wall_lines : list of dict
        Wall line segments with endpoints
    min_gap : float
        Minimum gap to be considered opening (not just noise)
    max_gap : float
        Maximum gap (larger = different wall, not opening)
    max_perpendicular_offset : float
        Maximum perpendicular offset between endpoints
        (they should be roughly aligned)
    verbose : bool
        Print debug info
        
    Returns:
    --------
    gaps : list of dict
        Detected gaps with endpoints, width, and wall indices
    """
    
    gaps = []
    n_walls = len(wall_lines)
    
    if verbose:
        print(f"\n  Checking {n_walls} walls for endpoint gaps...")
    
    # Collect all endpoints
    endpoints = []
    for i, line in enumerate(wall_lines):
        endpoints.append({
            'point': line['p1'],
            'wall_idx': i,
            'endpoint_id': 'p1',
            'direction': line['direction'],
            'normal': line['normal'],
            'wall_length': line['length']
        })
        endpoints.append({
            'point': line['p2'],
            'wall_idx': i,
            'endpoint_id': 'p2',
            'direction': line['direction'],
            'normal': line['normal'],
            'wall_length': line['length']
        })
    
    # Check all endpoint pairs
    checked_pairs = set()
    
    for i, ep1 in enumerate(endpoints):
        for j, ep2 in enumerate(endpoints):
            if i >= j:
                continue
            
            # Skip endpoints from same wall
            if ep1['wall_idx'] == ep2['wall_idx']:
                continue
            
            # Skip if already checked this wall pair
            wall_pair = tuple(sorted([ep1['wall_idx'], ep2['wall_idx']]))
            endpoint_pair = (ep1['endpoint_id'], ep2['endpoint_id'])
            pair_key = (wall_pair, endpoint_pair)
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)
            
            # Calculate distance
            p1 = ep1['point']
            p2 = ep2['point']
            gap_vector = p2 - p1
            distance = np.linalg.norm(gap_vector)
            
            # Check if within gap range
            if distance < min_gap or distance > max_gap:
                continue
            
            # Check alignment - endpoints should be roughly aligned
            # (perpendicular offset should be small)
            gap_direction = gap_vector / distance
            
            # Use average wall direction to check alignment
            avg_direction = (ep1['direction'] + ep2['direction']) / 2
            avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-10)
            
            # Perpendicular offset
            perp_component = abs(np.dot(gap_vector, np.array([-avg_direction[1], avg_direction[0]])))
            
            if perp_component > max_perpendicular_offset:
                if verbose:
                    print(f"    Walls {ep1['wall_idx']}-{ep2['wall_idx']}: "
                          f"gap={distance:.2f}m, perp_offset={perp_component:.2f}m (too large)")
                continue
            
            # Check if walls are roughly collinear or at an angle
            dot_directions = abs(np.dot(ep1['direction'], ep2['direction']))
            angle_between = np.degrees(np.arccos(np.clip(dot_directions, 0, 1)))
            
            # Valid gap found
            gap = {
                'p1': p1,
                'p2': p2,
                'width': distance,
                'wall_idx_1': ep1['wall_idx'],
                'wall_idx_2': ep2['wall_idx'],
                'endpoint_1': ep1['endpoint_id'],
                'endpoint_2': ep2['endpoint_id'],
                'direction': gap_direction,
                'perpendicular_offset': perp_component,
                'wall_angle': angle_between,
                'midpoint': (p1 + p2) / 2,
                'header_wall': None  # Will be filled by corner detection
            }
            
            gaps.append(gap)
            
            if verbose:
                print(f"    Gap found: walls {ep1['wall_idx']}-{ep2['wall_idx']}, "
                      f"width={distance:.2f}m, wall_angle={angle_between:.1f}°")
    
    return gaps


def find_corner_openings(wall_lines: List[Dict],
                         min_gap: float = 0.5,
                         max_gap: float = 2.0,
                         max_header_length: float = 2.0,
                         min_header_length: float = 0.3,
                         endpoint_tolerance: float = 0.3,
                         verbose: bool = False) -> List[Dict]:
    """
    Find 3-wall corner openings where a small wall (header) connects two walls
    above a doorway.
    
    Configuration:
        Wall A (e.g., Orange)
        ━━━━━━━━━━━━━━━━●
                        │
                        │ ← DOOR GAP
                        │
                  ●━━━━━● Wall B (header, e.g., Green)
                  │
                  │
                  ● Wall C (e.g., Dark Blue)
    
    Parameters:
    -----------
    wall_lines : list of dict
        Wall line segments with endpoints
    min_gap : float
        Minimum door width
    max_gap : float
        Maximum door width
    max_header_length : float
        Maximum length of header wall (small wall above door)
    min_header_length : float
        Minimum header length
    endpoint_tolerance : float
        Tolerance for endpoint matching
    verbose : bool
        Print debug info
        
    Returns:
    --------
    corner_gaps : list of dict
        Detected 3-wall corner openings
    """
    
    corner_gaps = []
    n_walls = len(wall_lines)
    
    if verbose:
        print(f"\n  Checking for 3-wall corner openings...")
    
    # Find potential header walls (short walls)
    header_candidates = []
    main_walls = []
    
    for i, line in enumerate(wall_lines):
        if min_header_length <= line['length'] <= max_header_length:
            header_candidates.append((i, line))
        if line['length'] > max_header_length:
            main_walls.append((i, line))
    
    if verbose:
        print(f"    Header candidates: {len(header_candidates)}")
        print(f"    Main walls: {len(main_walls)}")
    
    # For each header candidate, check if it connects two main walls
    for header_idx, header in header_candidates:
        header_p1 = header['p1']
        header_p2 = header['p2']
        
        # Find main walls connected to header endpoints
        connected_to_p1 = []
        connected_to_p2 = []
        
        for main_idx, main_wall in main_walls:
            # Check distance from header endpoints to main wall endpoints
            for main_ep_name, main_ep in [('p1', main_wall['p1']), ('p2', main_wall['p2'])]:
                dist_to_header_p1 = np.linalg.norm(main_ep - header_p1)
                dist_to_header_p2 = np.linalg.norm(main_ep - header_p2)
                
                if dist_to_header_p1 < endpoint_tolerance:
                    connected_to_p1.append({
                        'wall_idx': main_idx,
                        'wall': main_wall,
                        'endpoint': main_ep_name,
                        'point': main_ep,
                        'distance': dist_to_header_p1
                    })
                
                if dist_to_header_p2 < endpoint_tolerance:
                    connected_to_p2.append({
                        'wall_idx': main_idx,
                        'wall': main_wall,
                        'endpoint': main_ep_name,
                        'point': main_ep,
                        'distance': dist_to_header_p2
                    })
        
        # If header connects two different main walls, we have a corner opening
        for conn1 in connected_to_p1:
            for conn2 in connected_to_p2:
                if conn1['wall_idx'] == conn2['wall_idx']:
                    continue  # Same wall, skip
                
                # Find the OTHER endpoints of connected walls (the door gap)
                wall1 = conn1['wall']
                wall2 = conn2['wall']
                
                # Get the endpoints NOT connected to header
                if conn1['endpoint'] == 'p1':
                    door_ep1 = wall1['p2']
                else:
                    door_ep1 = wall1['p1']
                
                if conn2['endpoint'] == 'p1':
                    door_ep2 = wall2['p2']
                else:
                    door_ep2 = wall2['p1']
                
                # Calculate gap width (distance between non-header endpoints)
                gap_vector = door_ep2 - door_ep1
                gap_width = np.linalg.norm(gap_vector)
                
                # Also consider the header length as part of door width measurement
                # The actual door width is the perpendicular distance
                header_vector = header_p2 - header_p1
                header_length = np.linalg.norm(header_vector)
                
                if verbose:
                    print(f"    Checking header {header_idx} (len={header_length:.2f}m) "
                          f"connecting walls {conn1['wall_idx']} and {conn2['wall_idx']}")
                    print(f"      Gap between far endpoints: {gap_width:.2f}m")
                
                # The door width is approximately the header length
                # (header spans across the top of the door)
                door_width = header_length
                
                if min_gap <= door_width <= max_gap:
                    # Valid corner opening found
                    corner_gap = {
                        'p1': header_p1,
                        'p2': header_p2,
                        'width': door_width,
                        'wall_idx_1': conn1['wall_idx'],
                        'wall_idx_2': conn2['wall_idx'],
                        'header_wall': header_idx,
                        'header_length': header_length,
                        'endpoint_1': conn1['endpoint'],
                        'endpoint_2': conn2['endpoint'],
                        'direction': header_vector / header_length if header_length > 0 else np.array([1, 0]),
                        'midpoint': (header_p1 + header_p2) / 2,
                        'wall_angle': 0,  # Will be calculated
                        'perpendicular_offset': 0,
                        'is_corner_opening': True
                    }
                    
                    # Calculate angle between the two main walls
                    dot_dirs = abs(np.dot(wall1['direction'], wall2['direction']))
                    corner_gap['wall_angle'] = np.degrees(np.arccos(np.clip(dot_dirs, 0, 1)))
                    
                    corner_gaps.append(corner_gap)
                    
                    if verbose:
                        print(f"    → Corner opening found: walls {conn1['wall_idx']}-{conn2['wall_idx']} "
                              f"with header {header_idx}, width={door_width:.2f}m")
    
    return corner_gaps


def find_all_endpoint_gaps(wall_lines: List[Dict],
                           min_gap: float = 0.5,
                           max_gap: float = 2.0,
                           max_perpendicular_offset: float = 0.3,
                           max_header_length: float = 2.0,
                           min_header_length: float = 0.3,
                           endpoint_tolerance: float = 0.3,
                           verbose: bool = False) -> List[Dict]:
    """
    Find ALL endpoint gaps: both 2-wall gaps and 3-wall corner openings.
    
    Parameters:
    -----------
    wall_lines : list of dict
        Wall line segments
    min_gap, max_gap : float
        Door width range
    max_perpendicular_offset : float
        For 2-wall gaps alignment check
    max_header_length, min_header_length : float
        Header wall size range for corner detection
    endpoint_tolerance : float
        Tolerance for endpoint matching in corner detection
    verbose : bool
        Print debug info
        
    Returns:
    --------
    all_gaps : list of dict
        All detected gaps (2-wall and 3-wall)
    """
    
    # Find 2-wall gaps
    two_wall_gaps = find_wall_endpoint_gaps(
        wall_lines,
        min_gap=min_gap,
        max_gap=max_gap,
        max_perpendicular_offset=max_perpendicular_offset,
        verbose=verbose
    )
    
    # Find 3-wall corner openings
    corner_gaps = find_corner_openings(
        wall_lines,
        min_gap=min_gap,
        max_gap=max_gap,
        max_header_length=max_header_length,
        min_header_length=min_header_length,
        endpoint_tolerance=endpoint_tolerance,
        verbose=verbose
    )
    
    # Mark 2-wall gaps
    for gap in two_wall_gaps:
        gap['is_corner_opening'] = False
    
    # Combine, avoiding duplicates
    # A corner opening might overlap with a 2-wall gap detection
    all_gaps = list(two_wall_gaps)
    
    for corner_gap in corner_gaps:
        # Check if this corner opening duplicates a 2-wall gap
        is_duplicate = False
        for existing in two_wall_gaps:
            # Same walls involved
            walls_match = (
                {existing['wall_idx_1'], existing['wall_idx_2']} == 
                {corner_gap['wall_idx_1'], corner_gap['wall_idx_2']}
            )
            # Similar width
            width_similar = abs(existing['width'] - corner_gap['width']) < 0.3
            
            if walls_match and width_similar:
                # Update existing gap with header info instead of adding duplicate
                existing['header_wall'] = corner_gap['header_wall']
                existing['is_corner_opening'] = True
                is_duplicate = True
                break
        
        if not is_duplicate:
            all_gaps.append(corner_gap)
    
    if verbose:
        print(f"\n  Total gaps found: {len(all_gaps)}")
        print(f"    2-wall gaps: {len(two_wall_gaps)}")
        print(f"    3-wall corner openings: {len(corner_gaps)}")
    
    return all_gaps


def get_opening_height(walls: List[Dict],
                       gap: Dict,
                       floor_height: float,
                       ceiling_height: float,
                       header_percentile: float = 10.0,  # 10th percentile to filter more noise
                       verbose: bool = False) -> Tuple[float, float, float]:
    """
    Determine opening height by analyzing wall heights near the gap.
    
    For 3-wall corner openings, uses the BOTTOM of the header wall.
    For 2-wall openings, uses the TOP of the adjacent walls.
    
    Parameters:
    -----------
    walls : list of dict
        Original wall data with points
    gap : dict
        Gap info with wall indices
    floor_height : float
        Floor Z coordinate
    ceiling_height : float
        Ceiling Z coordinate
    header_percentile : float
        Percentile for header bottom (filters noise). Default 10.0.
        Higher = more aggressive noise filtering.
    verbose : bool
        Print debug info
        
    Returns:
    --------
    opening_z_min : float
        Bottom of opening
    opening_z_max : float
        Top of opening
    height : float
        Opening height
    """
    
    wall1 = walls[gap['wall_idx_1']]
    wall2 = walls[gap['wall_idx_2']]
    
    # Get Z range of both main walls
    z_min_1 = wall1['points'][:, 2].min()
    z_max_1 = wall1['points'][:, 2].max()
    z_min_2 = wall2['points'][:, 2].min()
    z_max_2 = wall2['points'][:, 2].max()
    
    # Opening bottom is at floor
    opening_z_min = floor_height
    
    # Opening top depends on whether there's a header wall
    header_wall_idx = gap.get('header_wall', None)
    
    if header_wall_idx is not None:
        # 3-wall corner opening: door top is at BOTTOM of header wall
        header_wall = walls[header_wall_idx]
        header_z_values = header_wall['points'][:, 2]
        
        # Use percentile to filter out noise points
        header_z_min = np.min(header_z_values)
        header_z_pct = np.percentile(header_z_values, header_percentile)
        header_z_max = np.max(header_z_values)
        
        if verbose:
            print(f"    Header wall {header_wall_idx}:")
            print(f"      z_min={header_z_min:.2f}, z_{header_percentile:.0f}%={header_z_pct:.2f}, z_max={header_z_max:.2f}")
        
        # Door top is at the bottom of the header (using percentile to filter noise)
        # But if header goes close to floor (bad data), fall back to wall tops
        if header_z_pct <= floor_height + 0.5:
            # Header too close to floor - use wall tops instead
            opening_z_max = min(z_max_1, z_max_2, ceiling_height)
            if verbose:
                print(f"      Header too close to floor, using wall tops: {opening_z_max:.2f}")
        else:
            opening_z_max = header_z_pct
            if verbose:
                print(f"      Using {header_percentile:.0f}th percentile as door top: {opening_z_max:.2f}")
        
    else:
        # 2-wall opening: door top is at top of walls
        opening_z_max = min(z_max_1, z_max_2, ceiling_height)
    
    height = opening_z_max - opening_z_min
    
    if verbose:
        print(f"    Opening height: {opening_z_min:.2f} to {opening_z_max:.2f} = {height:.2f}m")
    
    return opening_z_min, opening_z_max, height


def classify_between_wall_openings(gaps: List[Dict],
                                   walls: List[Dict],
                                   floor_height: float,
                                   ceiling_height: float,
                                   min_door_width: float = 0.6,
                                   max_door_width: float = 1.5,
                                   min_door_height: float = 1.8,
                                   min_passage_width: float = 0.8,
                                   max_passage_width: float = 3.0,
                                   verbose: bool = False) -> List[Dict]:
    """
    Classify gaps as doors, passages, or other openings.
    
    Parameters:
    -----------
    gaps : list of dict
        Detected gaps
    walls : list of dict
        Original wall data
    floor_height : float
        Floor Z
    ceiling_height : float  
        Ceiling Z
    min_door_width, max_door_width : float
        Door width range
    min_door_height : float
        Minimum door height
    min_passage_width, max_passage_width : float
        Passage width range (wider than door)
        
    Returns:
    --------
    openings : list of dict
        Classified openings
    """
    
    openings = []
    
    for gap in gaps:
        width = gap['width']
        
        # Get height
        z_min, z_max, height = get_opening_height(
            walls, gap, floor_height, ceiling_height, verbose=verbose
        )
        
        opening = {
            'p1': gap['p1'],
            'p2': gap['p2'],
            'width': width,
            'height': height,
            'z_min': z_min,
            'z_max': z_max,
            'midpoint': gap['midpoint'],
            'wall_idx_1': gap['wall_idx_1'],
            'wall_idx_2': gap['wall_idx_2'],
            'wall_angle': gap['wall_angle'],
            'location': 'between_walls',
            'header_wall': gap.get('header_wall', None),
            'is_corner_opening': gap.get('is_corner_opening', False)
        }
        
        # Classify
        if min_door_width <= width <= max_door_width:
            if height >= min_door_height:
                opening['type'] = 'door'
                opening['confidence'] = 'high'
            else:
                opening['type'] = 'door'
                opening['confidence'] = 'medium'
        
        elif max_door_width < width <= max_passage_width:
            opening['type'] = 'passage'
            opening['confidence'] = 'high' if height >= min_door_height else 'medium'
        
        elif width < min_door_width:
            opening['type'] = 'narrow_gap'
            opening['confidence'] = 'low'
        
        else:
            opening['type'] = 'wide_opening'
            opening['confidence'] = 'low'
        
        openings.append(opening)
    
    return openings


def detect_between_wall_openings(walls: List[Dict],
                                  floor_height: float,
                                  ceiling_height: float,
                                  # Gap detection params
                                  min_gap: float = 0.5,
                                  max_gap: float = 2.0,
                                  max_perpendicular_offset: float = 0.3,
                                  # Corner opening params
                                  max_header_length: float = 2.0,
                                  min_header_length: float = 0.3,
                                  endpoint_tolerance: float = 0.3,
                                  # Classification params
                                  min_door_width: float = 0.6,
                                  max_door_width: float = 1.5,
                                  min_door_height: float = 1.8,
                                  min_passage_width: float = 0.8,
                                  max_passage_width: float = 3.0,
                                  verbose: bool = True) -> List[Dict]:
    """
    Detect openings between walls (doorways, passages).
    
    Handles both:
    - 2-wall gaps: simple gap between two wall endpoints
    - 3-wall corner openings: gap with header wall above (chamfer/lintel)
    
    Parameters:
    -----------
    walls : list of dict
        Wall segments with 'points'
    floor_height : float
        Z coordinate of floor
    ceiling_height : float
        Z coordinate of ceiling
    min_gap : float
        Minimum gap width to consider
    max_gap : float
        Maximum gap width (larger = separate walls, not opening)
    max_perpendicular_offset : float
        Maximum perpendicular offset between endpoints
    max_header_length : float
        Maximum length of header wall (wall above door)
    min_header_length : float
        Minimum header length
    endpoint_tolerance : float
        Tolerance for endpoint matching in corner detection
    min_door_width, max_door_width : float
        Door width range
    min_door_height : float
        Minimum door height
    min_passage_width, max_passage_width : float
        Passage width range
    verbose : bool
        Print progress
        
    Returns:
    --------
    openings : list of dict
        Detected between-wall openings
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"BETWEEN-WALLS OPENING DETECTION")
        print(f"{'='*70}")
        print(f"  Walls: {len(walls)}")
        print(f"  Gap range: {min_gap}m - {max_gap}m")
        print(f"  Header wall range: {min_header_length}m - {max_header_length}m")
    
    # Extract wall line segments
    wall_lines = []
    for i, wall in enumerate(walls):
        line = extract_wall_line_2d(wall)
        line['wall_idx'] = i
        wall_lines.append(line)
        
        if verbose:
            print(f"  Wall {i}: length={line['length']:.2f}m, "
                  f"p1=[{line['p1'][0]:.2f}, {line['p1'][1]:.2f}], "
                  f"p2=[{line['p2'][0]:.2f}, {line['p2'][1]:.2f}]")
    
    # Find ALL gaps (2-wall and 3-wall corner)
    gaps = find_all_endpoint_gaps(
        wall_lines,
        min_gap=min_gap,
        max_gap=max_gap,
        max_perpendicular_offset=max_perpendicular_offset,
        max_header_length=max_header_length,
        min_header_length=min_header_length,
        endpoint_tolerance=endpoint_tolerance,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n  Found {len(gaps)} potential gaps")
    
    # Classify openings
    openings = classify_between_wall_openings(
        gaps,
        walls,
        floor_height=floor_height,
        ceiling_height=ceiling_height,
        min_door_width=min_door_width,
        max_door_width=max_door_width,
        min_door_height=min_door_height,
        min_passage_width=min_passage_width,
        max_passage_width=max_passage_width,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"BETWEEN-WALL OPENINGS: {len(openings)}")
        for op in openings:
            header_info = ""
            if op.get('header_wall') is not None:
                header_info = f", header wall={op['header_wall']}"
            print(f"  {op['type']} ({op['confidence']}): "
                  f"W={op['width']:.2f}m × H={op['height']:.2f}m "
                  f"between walls {op['wall_idx_1']}-{op['wall_idx_2']}{header_info}")
        print(f"{'='*70}")
    
    return openings


def visualize_between_wall_openings(walls: List[Dict],
                                     openings: List[Dict],
                                     title: str = "Between-Wall Openings",
                                     save_path: str = None):
    """
    Visualize walls and between-wall openings (top-down view).
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Draw walls
    for i, wall in enumerate(walls):
        line = extract_wall_line_2d(wall)
        color = colors[i % len(colors)]
        
        # Draw wall line
        ax.plot([line['p1'][0], line['p2'][0]], 
                [line['p1'][1], line['p2'][1]],
                linewidth=4, color=color, label=f'Wall {i}')
        
        # Mark endpoints
        ax.scatter([line['p1'][0], line['p2'][0]], 
                   [line['p1'][1], line['p2'][1]],
                   s=100, color=color, zorder=5)
        
        # Label wall
        ax.annotate(f'W{i}', line['centroid'], fontsize=10, ha='center')
    
    # Draw openings
    opening_colors = {'door': 'red', 'passage': 'orange', 
                      'narrow_gap': 'gray', 'wide_opening': 'purple'}
    
    for op in openings:
        color = opening_colors.get(op['type'], 'black')
        
        # Draw opening line
        ax.plot([op['p1'][0], op['p2'][0]], 
                [op['p1'][1], op['p2'][1]],
                linewidth=6, color=color, linestyle='--', alpha=0.7)
        
        # Label
        ax.annotate(
            f"{op['type']}\n{op['width']:.2f}m",
            op['midpoint'],
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# COMBINED DETECTION: In-wall + Between-wall openings
# =============================================================================

def detect_all_openings_combined(walls: List[Dict],
                                  floor_height: float,
                                  ceiling_height: float,
                                  # In-wall detection params
                                  cell_size: float = 0.05,
                                  density_threshold: float = 0.1,
                                  min_gap_width: float = 0.5,
                                  min_gap_height: float = 0.5,
                                  # Between-wall detection params
                                  min_between_gap: float = 0.5,
                                  max_between_gap: float = 2.0,
                                  max_perpendicular_offset: float = 0.3,
                                  # Classification params
                                  floor_tolerance: float = 0.15,
                                  min_door_height: float = 1.8,
                                  max_door_height: float = 2.5,
                                  min_door_width: float = 0.6,
                                  max_door_width: float = 1.5,
                                  min_window_height: float = 0.5,
                                  min_window_width: float = 0.4,
                                  verbose: bool = True) -> Dict:
    """
    Detect ALL openings: both within walls and between walls.
    
    Parameters:
    -----------
    walls : list of dict
        Wall segments
    floor_height, ceiling_height : float
        Room heights
    ... (see individual functions for params)
    
    Returns:
    --------
    result : dict
        'in_wall': openings within walls (windows, doors in wall)
        'between_walls': openings between walls (doorways)
        'all': combined list
    """
    
    # Import in-wall detection
    try:
        from opening_detection import detect_all_openings
        
        in_wall_openings = detect_all_openings(
            walls=walls,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            cell_size=cell_size,
            density_threshold=density_threshold,
            min_gap_width=min_gap_width,
            min_gap_height=min_gap_height,
            floor_tolerance=floor_tolerance,
            min_door_height=min_door_height,
            max_door_height=max_door_height,
            min_door_width=min_door_width,
            max_door_width=max_door_width,
            min_window_height=min_window_height,
            min_window_width=min_window_width,
            verbose=verbose
        )
        
        # Mark location
        for op in in_wall_openings:
            op['location'] = 'in_wall'
            
    except ImportError:
        if verbose:
            print("Warning: opening_detection module not found, skipping in-wall detection")
        in_wall_openings = []
    
    # Detect between-wall openings
    between_wall_openings = detect_between_wall_openings(
        walls=walls,
        floor_height=floor_height,
        ceiling_height=ceiling_height,
        min_gap=min_between_gap,
        max_gap=max_between_gap,
        max_perpendicular_offset=max_perpendicular_offset,
        min_door_width=min_door_width,
        max_door_width=max_door_width,
        min_door_height=min_door_height,
        verbose=verbose
    )
    
    # Combine
    all_openings = in_wall_openings + between_wall_openings
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"COMBINED RESULTS")
        print(f"{'='*70}")
        print(f"  In-wall openings: {len(in_wall_openings)}")
        print(f"  Between-wall openings: {len(between_wall_openings)}")
        print(f"  Total: {len(all_openings)}")
        print(f"{'='*70}")
    
    return {
        'in_wall': in_wall_openings,
        'between_walls': between_wall_openings,
        'all': all_openings
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Between-Walls Opening Detection")
    print("=" * 50)
    
    # Create synthetic room with both 2-wall gap and 3-wall corner opening
    np.random.seed(42)
    
    floor_z = 0.0
    ceiling_z = 2.5
    
    print("\n--- Test 1: Simple 2-wall gap ---")
    
    # Wall 1: Left wall (full)
    n_points = 2000
    x1 = np.zeros(n_points)
    y1 = np.random.uniform(0, 4, n_points)
    z1 = np.random.uniform(floor_z, ceiling_z, n_points)
    wall1_points = np.column_stack([x1, y1, z1])
    
    # Wall 2: Top wall (full)
    x2 = np.random.uniform(0, 5, n_points)
    y2 = np.ones(n_points) * 4
    z2 = np.random.uniform(floor_z, ceiling_z, n_points)
    wall2_points = np.column_stack([x2, y2, z2])
    
    # Wall 3: Right wall, part 1 (before door)
    n_part = 1000
    x3a = np.ones(n_part) * 5
    y3a = np.random.uniform(0, 1.5, n_part)
    z3a = np.random.uniform(floor_z, ceiling_z, n_part)
    
    # Wall 4: Right wall, part 2 (after door) - gap is 1.5 to 2.4 (0.9m door)
    x3b = np.ones(n_part) * 5
    y3b = np.random.uniform(2.4, 4, n_part)
    z3b = np.random.uniform(floor_z, ceiling_z, n_part)
    
    # Wall 5: Bottom wall (full)
    x4 = np.random.uniform(0, 5, n_points)
    y4 = np.zeros(n_points)
    z4 = np.random.uniform(floor_z, ceiling_z, n_points)
    wall4_points = np.column_stack([x4, y4, z4])
    
    walls_2wall = [
        {'points': wall1_points, 'normal': np.array([-1, 0, 0])},
        {'points': wall2_points, 'normal': np.array([0, 1, 0])},
        {'points': np.column_stack([x3a, y3a, z3a]), 'normal': np.array([1, 0, 0])},
        {'points': np.column_stack([x3b, y3b, z3b]), 'normal': np.array([1, 0, 0])},
        {'points': wall4_points, 'normal': np.array([0, -1, 0])},
    ]
    
    print(f"\nSynthetic room: {len(walls_2wall)} walls")
    print(f"  Expected: 0.9m door gap between walls 2 and 3")
    
    openings = detect_between_wall_openings(
        walls=walls_2wall,
        floor_height=floor_z,
        ceiling_height=ceiling_z,
        min_gap=0.5,
        max_gap=2.0,
        verbose=True
    )
    
    print(f"\n✓ Test 1 result: {len(openings)} openings detected")
    
    # ==========================================================================
    
    print("\n\n--- Test 2: 3-wall corner opening (with header) ---")
    
    # Simulating your room: Orange - Green (header) - Dark Blue
    #
    #     Orange (wall 0)
    #     ━━━━━━━━━━━━━━━●
    #                    ╲
    #                     ● Green (wall 1, header above door)
    #                      ╲
    #                       ●━━━━━━━━ Dark Blue (wall 2)
    
    n_points = 2000
    
    # Wall 0: Orange (top-left, going right)
    # endpoints approximately at (0,3) to (3,3)
    x0 = np.random.uniform(0, 3, n_points)
    y0 = 3 + np.random.normal(0, 0.02, n_points)
    z0 = np.random.uniform(floor_z, ceiling_z, n_points)
    
    # Wall 1: Green (header/chamfer, short wall, ~1m)
    # endpoints approximately at (3,3) to (3.7,2.3)
    n_header = 500
    t = np.random.uniform(0, 1, n_header)
    x1 = 3 + t * 0.7
    y1 = 3 - t * 0.7 + np.random.normal(0, 0.02, n_header)
    z1 = np.random.uniform(floor_z, ceiling_z, n_header)
    
    # Wall 2: Dark Blue (right wall, going down)
    # endpoints approximately at (3.7,2.3) to (3.7,0)
    x2 = 3.7 + np.random.normal(0, 0.02, n_points)
    y2 = np.random.uniform(0, 2.3, n_points)
    z2 = np.random.uniform(floor_z, ceiling_z, n_points)
    
    # Wall 3: Bottom wall
    x3 = np.random.uniform(0, 3.7, n_points)
    y3 = np.random.normal(0, 0.02, n_points)
    z3 = np.random.uniform(floor_z, ceiling_z, n_points)
    
    # Wall 4: Left wall
    x4 = np.random.normal(0, 0.02, n_points)
    y4 = np.random.uniform(0, 3, n_points)
    z4 = np.random.uniform(floor_z, ceiling_z, n_points)
    
    walls_3wall = [
        {'points': np.column_stack([x0, y0, z0]), 'normal': np.array([0, 1, 0])},   # Orange
        {'points': np.column_stack([x1, y1, z1]), 'normal': np.array([0.7, 0.7, 0])},  # Green (header)
        {'points': np.column_stack([x2, y2, z2]), 'normal': np.array([1, 0, 0])},   # Dark Blue
        {'points': np.column_stack([x3, y3, z3]), 'normal': np.array([0, -1, 0])},  # Bottom
        {'points': np.column_stack([x4, y4, z4]), 'normal': np.array([-1, 0, 0])},  # Left
    ]
    
    print(f"\nSynthetic room with chamfer: {len(walls_3wall)} walls")
    print(f"  Wall 0 (Orange): top wall")
    print(f"  Wall 1 (Green): header/chamfer above door (~1m)")
    print(f"  Wall 2 (Dark Blue): right wall")
    print(f"  Expected: Door with header between walls 0-2, header=1")
    
    openings = detect_between_wall_openings(
        walls=walls_3wall,
        floor_height=floor_z,
        ceiling_height=ceiling_z,
        min_gap=0.5,
        max_gap=2.0,
        max_header_length=1.5,
        min_header_length=0.3,
        endpoint_tolerance=0.5,
        verbose=True
    )
    
    print(f"\n✓ Test 2 result: {len(openings)} openings detected")
    for op in openings:
        if op.get('header_wall') is not None:
            print(f"  → 3-wall corner opening detected! Header wall: {op['header_wall']}")
