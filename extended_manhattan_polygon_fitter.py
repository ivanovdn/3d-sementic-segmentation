"""
Extended Manhattan World Polygon Fitter

Extends the Manhattan World assumption to handle non-rectangular rooms:
- N walls (not just 4)
- Multiple dominant directions (0°, 90°, 45°, etc.)
- Automatic wall adjacency detection
- Precise corner intersection

Usage:
    wall_lines = [extract_wall_line_2d(wall) for wall in merged_walls]
    wall_points = [wall["points"] for wall in merged_walls]
    
    fitter = ExtendedManhattanPolygonFitter(wall_lines, wall_points)
    result = fitter.fit(verbose=True)
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class ExtendedManhattanPolygonFitter:
    """
    Fits a polygon to wall segments using Extended Manhattan World assumption.
    
    Unlike standard Manhattan (2 perpendicular directions), this allows:
    - N dominant directions (detected automatically)
    - Common architectural angles: 0°, 45°, 90°, 135°
    - Or any angles present in the data
    """

    def __init__(self, wall_lines: List[Dict], wall_points: List[np.ndarray]):
        """
        Parameters:
        -----------
        wall_lines : list of dict
            Each dict has 'direction', 'centroid', 'start', 'end', 'length'
            (output from extract_wall_line_2d)
        wall_points : list of np.ndarray
            Point clouds for each wall, shape (N, 3) or (N, 2)
        """
        self.wall_lines = wall_lines
        self.wall_points = wall_points
        self.n_walls = len(wall_lines)
        
        # Results
        self.dominant_directions = None
        self.snapped_lines = None
        self.adjacency = None
        self.corners = None
        self.polygon = None

    def fit(self, 
            n_directions: Optional[int] = None,
            angle_snap_threshold: float = 15.0,
            use_standard_angles: bool = True,
            standard_angles: List[float] = [0, 45, 90, 135],
            verbose: bool = True) -> Optional[Dict]:
        """
        Fit polygon to walls.
        
        Parameters:
        -----------
        n_directions : int, optional
            Number of dominant directions. If None, auto-detect.
        angle_snap_threshold : float
            Max angle deviation for snapping (degrees)
        use_standard_angles : bool
            If True, prefer snapping to standard architectural angles
        standard_angles : list of float
            Standard angles to consider (degrees, 0-180 range)
        verbose : bool
        
        Returns:
        --------
        dict with polygon info, or None if fitting fails
        """
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"EXTENDED MANHATTAN WORLD POLYGON FITTING")
            print(f"{'='*70}")
            print(f"Number of walls: {self.n_walls}")

        # ====================================================================
        # STEP 1: Extract wall directions
        # ====================================================================
        
        directions, angles = self._extract_directions(verbose)
        
        # ====================================================================
        # STEP 2: Find dominant directions
        # ====================================================================
        
        self.dominant_directions = self._find_dominant_directions(
            angles,
            n_directions=n_directions,
            use_standard_angles=use_standard_angles,
            standard_angles=standard_angles,
            verbose=verbose
        )
        
        # ====================================================================
        # STEP 3: Snap walls to dominant directions
        # ====================================================================
        
        self.snapped_lines = self._snap_walls_to_directions(
            directions,
            angle_snap_threshold=angle_snap_threshold,
            verbose=verbose
        )
        
        # ====================================================================
        # STEP 4: Refit line offsets through original points
        # ====================================================================
        
        self._refit_line_offsets(verbose)
        
        # ====================================================================
        # STEP 5: Determine wall adjacency (which walls share corners)
        # ====================================================================
        
        self.adjacency = self._find_wall_adjacency(verbose)
        
        if self.adjacency is None:
            print("❌ Failed to determine wall adjacency")
            return None
        
        # ====================================================================
        # STEP 6: Find corners by intersecting adjacent walls
        # ====================================================================
        
        self.corners = self._find_corners(verbose)
        
        if self.corners is None or len(self.corners) < 3:
            print("❌ Failed to find valid corners")
            return None
        
        # ====================================================================
        # STEP 7: Build and validate polygon
        # ====================================================================
        
        self.polygon = self._build_polygon(verbose)
        
        return self.polygon

    def _extract_directions(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Extract direction vectors and angles from wall lines."""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 1: EXTRACT WALL DIRECTIONS")
            print(f"{'='*70}")
        
        directions = []
        angles = []
        
        for i, wall in enumerate(self.wall_lines):
            direction = np.array(wall["direction"])
            direction = direction / np.linalg.norm(direction)  # Ensure unit
            
            angle = np.arctan2(direction[1], direction[0])
            
            directions.append(direction)
            angles.append(angle)
            
            if verbose:
                print(f"  Wall {i}: dir=[{direction[0]:6.3f}, {direction[1]:6.3f}], "
                      f"angle={np.degrees(angle):7.2f}°, "
                      f"length={wall.get('length', 0):.2f}m")
        
        return np.array(directions), np.array(angles)

    def _find_dominant_directions(self,
                                   angles: np.ndarray,
                                   n_directions: Optional[int],
                                   use_standard_angles: bool,
                                   standard_angles: List[float],
                                   verbose: bool) -> List[Dict]:
        """Find dominant wall directions."""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 2: FIND DOMINANT DIRECTIONS")
            print(f"{'='*70}")
        
        # Normalize angles to [0, 180) range (walls are bidirectional)
        angles_normalized = angles.copy()
        for i in range(len(angles_normalized)):
            while angles_normalized[i] < 0:
                angles_normalized[i] += np.pi
            while angles_normalized[i] >= np.pi:
                angles_normalized[i] -= np.pi
        
        if verbose:
            print(f"\nNormalized angles (0-180°):")
            for i, (orig, norm) in enumerate(zip(angles, angles_normalized)):
                print(f"  Wall {i}: {np.degrees(orig):7.2f}° → {np.degrees(norm):7.2f}°")
        
        # ====================================================================
        # Option A: Use standard architectural angles
        # ====================================================================
        
        if use_standard_angles:
            if verbose:
                print(f"\nUsing standard angles: {standard_angles}°")
            
            standard_rads = [np.radians(a) for a in standard_angles]
            
            # Find which standard angles are present in data
            dominant = []
            
            for std_angle in standard_rads:
                # Check if any wall is close to this angle
                for wall_angle in angles_normalized:
                    angle_diff = self._angle_difference(wall_angle, std_angle)
                    
                    if angle_diff < np.radians(20):  # Within 20° of standard
                        # This standard angle is used
                        dominant.append({
                            'angle': std_angle,
                            'direction': np.array([np.cos(std_angle), np.sin(std_angle)]),
                            'walls': []
                        })
                        break
            
            if verbose:
                print(f"  Found {len(dominant)} standard directions in use")
        
        # ====================================================================
        # Option B: Cluster to find natural directions
        # ====================================================================
        
        else:
            # Auto-detect number of directions if not specified
            if n_directions is None:
                # Estimate from data using silhouette or elbow method
                n_directions = self._estimate_n_directions(angles_normalized)
            
            if verbose:
                print(f"\nClustering into {n_directions} directions")
            
            km = KMeans(n_clusters=n_directions, random_state=42, n_init=10)
            labels = km.fit_predict(angles_normalized.reshape(-1, 1))
            
            dominant = []
            for cluster_id in range(n_directions):
                cluster_angle = km.cluster_centers_[cluster_id, 0]
                dominant.append({
                    'angle': cluster_angle,
                    'direction': np.array([np.cos(cluster_angle), np.sin(cluster_angle)]),
                    'walls': []
                })
        
        # Sort by angle
        dominant.sort(key=lambda x: x['angle'])
        
        if verbose:
            print(f"\nDominant directions:")
            for i, d in enumerate(dominant):
                print(f"  Direction {i}: {np.degrees(d['angle']):7.2f}° "
                      f"[{d['direction'][0]:6.3f}, {d['direction'][1]:6.3f}]")
        
        return dominant

    def _angle_difference(self, a1: float, a2: float) -> float:
        """Compute smallest angle between two angles (handles wraparound)."""
        diff = abs(a1 - a2)
        if diff > np.pi / 2:
            diff = np.pi - diff
        return diff

    def _estimate_n_directions(self, angles: np.ndarray) -> int:
        """Estimate number of dominant directions from data."""
        # Simple heuristic: use number of unique angles within tolerance
        unique_count = 1
        sorted_angles = np.sort(angles)
        
        for i in range(1, len(sorted_angles)):
            if sorted_angles[i] - sorted_angles[i-1] > np.radians(20):
                unique_count += 1
        
        return min(unique_count, len(angles))

    def _snap_walls_to_directions(self,
                                   directions: np.ndarray,
                                   angle_snap_threshold: float,
                                   verbose: bool) -> List[Dict]:
        """Snap each wall to nearest dominant direction."""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 3: SNAP WALLS TO DOMINANT DIRECTIONS")
            print(f"{'='*70}")
        
        snapped_lines = []
        
        for i, direction in enumerate(directions):
            wall_angle = np.arctan2(direction[1], direction[0])
            
            # Normalize to [0, π)
            wall_angle_norm = wall_angle
            if wall_angle_norm < 0:
                wall_angle_norm += np.pi
            if wall_angle_norm >= np.pi:
                wall_angle_norm -= np.pi
            
            # Find best matching dominant direction
            best_match = None
            best_diff = float('inf')
            
            for dom_idx, dom in enumerate(self.dominant_directions):
                diff = self._angle_difference(wall_angle_norm, dom['angle'])
                
                if diff < best_diff:
                    best_diff = diff
                    best_match = dom_idx
            
            # Check if within threshold
            if np.degrees(best_diff) > angle_snap_threshold:
                if verbose:
                    print(f"\n  ⚠️ Wall {i}: No good match (best diff: {np.degrees(best_diff):.1f}°)")
                    print(f"     Keeping original direction")
                
                snapped_dir = direction
                dom_idx = -1  # No dominant direction
            else:
                # Snap to dominant direction
                dom = self.dominant_directions[best_match]
                snapped_dir = dom['direction'].copy()
                
                # Preserve original direction sign
                if np.dot(direction, snapped_dir) < 0:
                    snapped_dir = -snapped_dir
                
                dom_idx = best_match
                dom['walls'].append(i)
            
            snapped_lines.append({
                'wall_idx': i,
                'original_direction': direction,
                'direction': snapped_dir,
                'dominant_idx': dom_idx,
                'snap_error': np.degrees(best_diff),
                'centroid': self.wall_lines[i]['centroid'],
                'length': self.wall_lines[i].get('length', 0)
            })
            
            if verbose:
                orig_angle = np.degrees(np.arctan2(direction[1], direction[0]))
                snap_angle = np.degrees(np.arctan2(snapped_dir[1], snapped_dir[0]))
                print(f"\n  Wall {i}:")
                print(f"    Original: {orig_angle:7.2f}°")
                print(f"    Snapped:  {snap_angle:7.2f}° (dominant group {dom_idx})")
                print(f"    Error:    {np.degrees(best_diff):5.2f}°")
        
        return snapped_lines

    def _refit_line_offsets(self, verbose: bool) -> None:
        """Refit line offsets through original points with snapped directions."""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 4: REFIT LINE OFFSETS")
            print(f"{'='*70}")
        
        for i, line in enumerate(self.snapped_lines):
            direction = line['direction']
            wall_pts = self.wall_points[i]
            
            # Use 2D points
            if wall_pts.shape[1] > 2:
                wall_pts = wall_pts[:, :2]
            
            # Normal is perpendicular to direction
            normal = np.array([-direction[1], direction[0]])
            
            # Offset: n·x + d = 0, so d = -n·centroid
            centroid = wall_pts.mean(axis=0)
            offset = -np.dot(normal, centroid)
            
            # Calculate fit quality
            distances = np.abs(wall_pts.dot(normal) + offset)
            rms = np.sqrt(np.mean(distances**2))
            
            # Update line info
            line['normal'] = normal
            line['offset'] = offset
            line['centroid'] = centroid
            line['rms_error'] = rms
            
            # Recompute extent along direction
            centered = wall_pts - centroid
            projections = centered @ direction
            line['extent'] = (projections.min(), projections.max())
            line['length'] = projections.max() - projections.min()
            
            if verbose:
                print(f"\n  Wall {i}:")
                print(f"    Normal: [{normal[0]:6.3f}, {normal[1]:6.3f}]")
                print(f"    Offset: {offset:7.3f}")
                print(f"    Length: {line['length']:.3f}m")
                print(f"    RMS error: {rms*100:.2f}cm")

    def _find_wall_adjacency(self, verbose: bool) -> Optional[List[Tuple[int, int]]]:
        """
        Determine which walls are adjacent (share a corner).
        
        Uses endpoint proximity and direction compatibility.
        """
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 5: FIND WALL ADJACENCY")
            print(f"{'='*70}")
        
        # Compute endpoints for each wall
        for line in self.snapped_lines:
            centroid = line['centroid']
            direction = line['direction']
            ext_min, ext_max = line['extent']
            
            line['start'] = centroid + ext_min * direction
            line['end'] = centroid + ext_max * direction
        
        # Find adjacencies based on endpoint proximity
        adjacencies = []
        used_walls = set()
        
        # Start from wall 0 and find connected walls
        current_wall = 0
        ordered_walls = [current_wall]
        used_walls.add(current_wall)
        
        while len(ordered_walls) < self.n_walls:
            current_line = self.snapped_lines[current_wall]
            current_end = current_line['end']
            
            # Find nearest wall endpoint
            best_next = None
            best_dist = float('inf')
            best_endpoint = None  # 'start' or 'end' of next wall
            
            for j in range(self.n_walls):
                if j in used_walls:
                    continue
                
                next_line = self.snapped_lines[j]
                
                # Check distance to start and end of candidate wall
                dist_to_start = np.linalg.norm(current_end - next_line['start'])
                dist_to_end = np.linalg.norm(current_end - next_line['end'])
                
                if dist_to_start < best_dist:
                    best_dist = dist_to_start
                    best_next = j
                    best_endpoint = 'start'
                
                if dist_to_end < best_dist:
                    best_dist = dist_to_end
                    best_next = j
                    best_endpoint = 'end'
            
            if best_next is None:
                if verbose:
                    print(f"  ⚠️ Could not find next wall after wall {current_wall}")
                break
            
            # If connecting to 'end', flip the wall direction
            if best_endpoint == 'end':
                next_line = self.snapped_lines[best_next]
                next_line['start'], next_line['end'] = next_line['end'], next_line['start']
                next_line['direction'] = -next_line['direction']
                next_line['extent'] = (-next_line['extent'][1], -next_line['extent'][0])
            
            adjacencies.append((current_wall, best_next))
            ordered_walls.append(best_next)
            used_walls.add(best_next)
            current_wall = best_next
            
            if verbose:
                print(f"  Wall {adjacencies[-1][0]} → Wall {adjacencies[-1][1]} "
                      f"(dist: {best_dist*100:.1f}cm)")
        
        # Close the polygon
        if len(ordered_walls) == self.n_walls:
            adjacencies.append((ordered_walls[-1], ordered_walls[0]))
            if verbose:
                print(f"  Wall {ordered_walls[-1]} → Wall {ordered_walls[0]} (closing)")
        
        if verbose:
            print(f"\n  Ordered walls: {ordered_walls}")
            print(f"  Adjacencies: {adjacencies}")
        
        # Store ordered walls
        self.ordered_walls = ordered_walls
        
        return adjacencies

    def _find_corners(self, verbose: bool) -> Optional[List[Dict]]:
        """Find corners by intersecting adjacent walls."""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 6: FIND CORNERS")
            print(f"{'='*70}")
        
        corners = []
        
        for wall_i, wall_j in self.adjacency:
            line_i = self.snapped_lines[wall_i]
            line_j = self.snapped_lines[wall_j]
            
            # Intersect lines: n·x + d = 0
            corner = self._intersect_lines(line_i, line_j)
            
            if corner is None:
                if verbose:
                    print(f"  ⚠️ Failed to intersect Wall {wall_i} and Wall {wall_j}")
                continue
            
            # Calculate corner angle
            dir_i = line_i['direction']
            dir_j = line_j['direction']
            
            # Angle between walls (interior angle)
            dot = np.dot(dir_i, dir_j)
            angle = np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))
            interior_angle = 180 - angle  # Interior angle of polygon
            
            corners.append({
                'point': corner,
                'walls': (wall_i, wall_j),
                'angle': interior_angle
            })
            
            if verbose:
                print(f"  Corner {len(corners)}: Wall {wall_i} ∩ Wall {wall_j}")
                print(f"    Position: [{corner[0]:7.3f}, {corner[1]:7.3f}]")
                print(f"    Angle: {interior_angle:.1f}°")
        
        return corners

    def _intersect_lines(self, line1: Dict, line2: Dict) -> Optional[np.ndarray]:
        """Intersect two lines defined by normal and offset."""
        
        n1, d1 = line1['normal'], line1['offset']
        n2, d2 = line2['normal'], line2['offset']
        
        A = np.vstack([n1, n2])
        b = -np.array([d1, d2])
        
        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            return None
        
        return np.linalg.solve(A, b)

    def _build_polygon(self, verbose: bool) -> Dict:
        """Build final polygon from corners."""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 7: BUILD POLYGON")
            print(f"{'='*70}")
        
        # Extract corner points in order
        corner_points = np.array([c['point'] for c in self.corners])
        n_corners = len(corner_points)
        
        # Calculate edge lengths
        edges = []
        for i in range(n_corners):
            p1 = corner_points[i]
            p2 = corner_points[(i + 1) % n_corners]
            length = np.linalg.norm(p2 - p1)
            edges.append(length)
        
        # Calculate perimeter
        perimeter = sum(edges)
        
        # Calculate area (shoelace formula)
        x = corner_points[:, 0]
        y = corner_points[:, 1]
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % n_corners] - x[(i + 1) % n_corners] * y[i] 
                             for i in range(n_corners)))
        
        # Calculate centroid
        centroid = corner_points.mean(axis=0)
        
        # Corner angles
        corner_angles = [c['angle'] for c in self.corners]
        
        # Validation
        angle_sum = sum(corner_angles)
        expected_angle_sum = (n_corners - 2) * 180  # Sum of interior angles
        angle_error = abs(angle_sum - expected_angle_sum)
        
        if verbose:
            print(f"\nPolygon with {n_corners} corners:")
            
            print(f"\nCorners:")
            for i, corner in enumerate(corner_points):
                print(f"  C{i+1}: [{corner[0]:7.3f}, {corner[1]:7.3f}]")
            
            print(f"\nEdges:")
            for i, edge in enumerate(edges):
                wall_idx = self.ordered_walls[i]
                print(f"  E{i+1} (Wall {wall_idx}): {edge:.4f}m")
            
            print(f"\nCorner angles:")
            for i, angle in enumerate(corner_angles):
                deviation = ""
                if abs(angle - 90) < 5:
                    deviation = "(~90°)"
                elif abs(angle - 135) < 5:
                    deviation = "(~135°)"
                elif abs(angle - 45) < 5:
                    deviation = "(~45°)"
                print(f"  C{i+1}: {angle:.2f}° {deviation}")
            
            print(f"\nAngle sum: {angle_sum:.1f}° (expected: {expected_angle_sum}°, error: {angle_error:.1f}°)")
            
            print(f"\n{'='*70}")
            print(f"POLYGON SUMMARY")
            print(f"{'='*70}")
            print(f"Corners: {n_corners}")
            print(f"Perimeter: {perimeter:.4f}m")
            print(f"Area: {area:.4f}m²")
            print(f"Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}]")
        
        polygon = {
            'corners': corner_points,
            'corner_info': self.corners,
            'edges': edges,
            'edge_walls': self.ordered_walls,
            'n_corners': n_corners,
            'perimeter': perimeter,
            'area': area,
            'centroid': centroid,
            'corner_angles': corner_angles,
            'angle_sum': angle_sum,
            'angle_error': angle_error,
            'snapped_lines': self.snapped_lines,
            'dominant_directions': self.dominant_directions,
            'adjacency': self.adjacency
        }
        
        return polygon


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_polygon_result(polygon: Dict, 
                              wall_points: Optional[List[np.ndarray]] = None,
                              save_path: Optional[str] = None):
    """Visualize the fitted polygon."""
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corners = polygon['corners']
    edges = polygon['edges']
    n_corners = polygon['n_corners']
    
    # Plot wall points if provided
    if wall_points is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(wall_points)))
        for i, (pts, color) in enumerate(zip(wall_points, colors)):
            pts_2d = pts[:, :2] if pts.shape[1] > 2 else pts
            ax.scatter(pts_2d[:, 0], pts_2d[:, 1], s=1, c=[color], alpha=0.3)
    
    # Plot polygon edges
    corners_closed = np.vstack([corners, corners[0]])
    ax.plot(corners_closed[:, 0], corners_closed[:, 1], 
            'b-', linewidth=3, label='Fitted polygon')
    
    # Plot corners
    ax.scatter(corners[:, 0], corners[:, 1], 
               c='red', s=200, zorder=10, edgecolors='black', linewidths=2)
    
    # Label corners and edges
    for i in range(n_corners):
        # Corner label
        ax.annotate(f'C{i+1}', corners[i], fontsize=12, fontweight='bold',
                   ha='center', va='center', color='white')
        
        # Edge label
        mid = (corners[i] + corners[(i+1) % n_corners]) / 2
        edge_vec = corners[(i+1) % n_corners] - corners[i]
        perp = np.array([-edge_vec[1], edge_vec[0]])
        perp = perp / np.linalg.norm(perp) * 0.2
        
        ax.annotate(f'{edges[i]:.2f}m', mid + perp, fontsize=10,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"Polygon: {n_corners} corners, "
                f"Perimeter: {polygon['perimeter']:.2f}m, "
                f"Area: {polygon['area']:.2f}m²")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Extended Manhattan World Polygon Fitter")
    print("=" * 50)
    
    # Create synthetic test data: pentagon (like your room)
    np.random.seed(42)
    
    # 5 walls forming a pentagon-like shape
    # Similar to RoomPlan output: 4 axis-aligned + 1 diagonal
    true_corners = np.array([
        [1.8, -2.2],
        [-1.8, -2.2],
        [-1.8, 2.2],
        [1.0, 2.2],
        [1.8, 1.5],
    ])
    
    n_walls = len(true_corners)
    wall_lines = []
    wall_points = []
    
    for i in range(n_walls):
        start = true_corners[i]
        end = true_corners[(i + 1) % n_walls]
        
        # Direction
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length
        
        # Centroid
        centroid = (start + end) / 2
        
        # Generate noisy points along wall
        t = np.linspace(0, 1, 100)
        points = start + np.outer(t, end - start)
        noise = np.random.randn(len(points), 2) * 0.02
        points += noise
        points_3d = np.column_stack([points, np.zeros(len(points))])
        
        wall_lines.append({
            'direction': direction,
            'centroid': centroid,
            'start': start,
            'end': end,
            'length': length
        })
        wall_points.append(points_3d)
    
    # Fit polygon
    fitter = ExtendedManhattanPolygonFitter(wall_lines, wall_points)
    result = fitter.fit(verbose=True)
    
    if result:
        print("\n✓ Fitting successful!")
        
        # Compare with ground truth
        print(f"\nGround truth corners:")
        for i, corner in enumerate(true_corners):
            print(f"  {i+1}: [{corner[0]:7.3f}, {corner[1]:7.3f}]")
        
        # Visualize
        visualize_polygon_result(result, wall_points)
