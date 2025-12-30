"""
Comparison module for polygon fitting results vs RoomPlan ground truth.

Creates pandas DataFrames comparing:
- Edge lengths (with automatic alignment)
- Corner positions
- Corner angles
- Overall metrics (perimeter, area)

Supports optional ground truth measurements for validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json


def extract_roomplan_values(json_path: str) -> Dict:
    """
    Extract polygon values from RoomPlan JSON file.
    
    Uses floor polygonCorners for accurate corner positions.
    
    Parameters:
    -----------
    json_path : str
        Path to RoomPlan JSON file
        
    Returns:
    --------
    dict with:
        - corners: np.ndarray of corner positions
        - edges: list of edge lengths
        - corner_angles: list of corner angles
        - perimeter: total perimeter
        - area: total area
        - n_corners: number of corners
        - wall_lengths: list of wall lengths from wall dimensions
    """
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get floor polygon corners (most accurate source)
    floors = data.get('floors', [])
    if not floors:
        raise ValueError("No floors found in RoomPlan JSON")
    
    floor = floors[0]
    polygon_corners = floor.get('polygonCorners', [])
    
    if not polygon_corners:
        raise ValueError("No polygonCorners found in floor data")
    
    # Convert to numpy array (use X and Y, ignore Z which is ~0)
    corners = np.array([[c[0], c[1]] for c in polygon_corners])
    n_corners = len(corners)
    
    # Calculate edge lengths from corners
    edges = []
    for i in range(n_corners):
        p1 = corners[i]
        p2 = corners[(i + 1) % n_corners]
        length = np.linalg.norm(p2 - p1)
        edges.append(float(length))
    
    # Calculate corner angles
    corner_angles = []
    for i in range(n_corners):
        p_prev = corners[(i - 1) % n_corners]
        p_curr = corners[i]
        p_next = corners[(i + 1) % n_corners]
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        corner_angles.append(float(angle))
    
    # Calculate perimeter
    perimeter = sum(edges)
    
    # Calculate area using shoelace formula
    x = corners[:, 0]
    y = corners[:, 1]
    area = 0.5 * abs(sum(x[i] * y[(i + 1) % n_corners] - x[(i + 1) % n_corners] * y[i] 
                         for i in range(n_corners)))
    
    # Also get wall lengths from wall dimensions (for reference)
    walls = data.get('walls', [])
    wall_lengths = []
    for wall in walls:
        dimensions = wall.get('dimensions', [0, 0, 0])
        if isinstance(dimensions, list) and len(dimensions) > 0:
            wall_lengths.append(float(dimensions[0]))  # First element is length
    
    return {
        'corners': corners,
        'edges': edges,
        'corner_angles': corner_angles,
        'perimeter': float(perimeter),
        'area': float(area),
        'n_corners': n_corners,
        'wall_lengths': wall_lengths
    }


def _find_best_edge_alignment(edges1: List[float], edges2: List[float]) -> Tuple[int, bool]:
    """
    Find best rotation offset to align two edge lists.
    
    Returns:
    --------
    (offset, reversed): offset to apply to edges1, whether to reverse edges1 first
    """
    n = len(edges1)
    best_offset = 0
    best_error = float('inf')
    best_reversed = False
    
    # Try normal order
    for offset in range(n):
        rotated = edges1[offset:] + edges1[:offset]
        error = sum(abs(float(a) - float(b)) for a, b in zip(rotated, edges2))
        if error < best_error:
            best_error = error
            best_offset = offset
            best_reversed = False
    
    # Try reversed order
    edges1_rev = edges1[::-1]
    for offset in range(n):
        rotated = edges1_rev[offset:] + edges1_rev[:offset]
        error = sum(abs(float(a) - float(b)) for a, b in zip(rotated, edges2))
        if error < best_error:
            best_error = error
            best_offset = offset
            best_reversed = True
    
    return best_offset, best_reversed


def _align_corners(corners1: np.ndarray, corners2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two corner arrays by finding best rotation offset.
    """
    n1, n2 = len(corners1), len(corners2)
    
    if n1 != n2:
        return corners1, corners2
    
    best_offset = 0
    best_error = float('inf')
    best_reversed = False
    
    # Try normal order
    for offset in range(n1):
        rotated = np.roll(corners2, offset, axis=0)
        error = np.sum(np.linalg.norm(corners1 - rotated, axis=1))
        if error < best_error:
            best_error = error
            best_offset = offset
            best_reversed = False
    
    # Try reversed order
    corners2_rev = corners2[::-1]
    for offset in range(n1):
        rotated = np.roll(corners2_rev, offset, axis=0)
        error = np.sum(np.linalg.norm(corners1 - rotated, axis=1))
        if error < best_error:
            best_error = error
            best_offset = offset
            best_reversed = True
    
    if best_reversed:
        aligned2 = np.roll(corners2[::-1], best_offset, axis=0)
    else:
        aligned2 = np.roll(corners2, best_offset, axis=0)
    
    return corners1, aligned2


def compare_polygons(fitted: Dict, 
                     roomplan: Dict,
                     ground_truth: Optional[Dict] = None,
                     decimals: int = 3,
                     align_edges: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Compare fitted polygon with RoomPlan ground truth.
    
    Parameters:
    -----------
    fitted : dict
        Output from ExtendedManhattanPolygonFitter.fit()
    roomplan : dict
        Output from extract_roomplan_values()
    ground_truth : dict, optional
        Manual ground truth measurements with keys:
        - 'edges': list of edge lengths (in same order as roomplan)
        - 'angles': list of corner angles
        - 'perimeter': float
        - 'area': float
    decimals : int
        Round to this many decimal places
    align_edges : bool
        If True, align edges by finding best rotation offset
        
    Returns:
    --------
    dict with DataFrames:
        - 'edges': Edge length comparison
        - 'angles': Corner angle comparison
        - 'summary': Overall metrics comparison
    """
    
    results = {}
    
    # ========================================================================
    # Align edges if needed
    # ========================================================================
    
    fitted_edges = [float(e) for e in fitted.get('edges', [])]
    roomplan_edges = [float(e) for e in roomplan.get('edges', [])]
    fitted_angles = [float(a) for a in fitted.get('corner_angles', [])]
    roomplan_angles = [float(a) for a in roomplan.get('corner_angles', [])]
    
    if align_edges and len(fitted_edges) == len(roomplan_edges) and len(fitted_edges) > 0:
        # Find best rotation to align edges
        best_offset, best_reversed = _find_best_edge_alignment(fitted_edges, roomplan_edges)
        
        # Apply alignment to fitted data
        if best_reversed:
            fitted_edges = fitted_edges[::-1]
            fitted_angles = fitted_angles[::-1]
        
        fitted_edges = fitted_edges[best_offset:] + fitted_edges[:best_offset]
        fitted_angles = fitted_angles[best_offset:] + fitted_angles[:best_offset]
    
    # ========================================================================
    # Edge Comparison
    # ========================================================================
    
    n_edges = max(len(fitted_edges), len(roomplan_edges))
    gt_edges = ground_truth.get('edges', []) if ground_truth else []
    
    edge_data = []
    for i in range(n_edges):
        row = {'Edge': f'E{i+1}'}
        
        fitted_val = fitted_edges[i] if i < len(fitted_edges) else None
        roomplan_val = roomplan_edges[i] if i < len(roomplan_edges) else None
        gt_val = gt_edges[i] if i < len(gt_edges) else None
        
        row['Fitted (m)'] = round(fitted_val, decimals) if fitted_val is not None else None
        row['RoomPlan (m)'] = round(roomplan_val, decimals) if roomplan_val is not None else None
        
        if gt_val is not None:
            row['GroundTruth (m)'] = round(float(gt_val), decimals)
        
        # Diff vs RoomPlan
        if fitted_val is not None and roomplan_val is not None:
            diff = fitted_val - roomplan_val
            row['Diff_RP (cm)'] = round(diff * 100, 1)
            row['Diff_RP (%)'] = round((diff / roomplan_val) * 100, 2) if roomplan_val != 0 else 0
        else:
            row['Diff_RP (cm)'] = None
            row['Diff_RP (%)'] = None
        
        # Diff vs Ground Truth
        if gt_val is not None and fitted_val is not None:
            diff_gt = fitted_val - float(gt_val)
            row['Diff_GT (cm)'] = round(diff_gt * 100, 1)
        
        edge_data.append(row)
    
    results['edges'] = pd.DataFrame(edge_data)
    
    # ========================================================================
    # Corner Angle Comparison
    # ========================================================================
    
    n_angles = max(len(fitted_angles), len(roomplan_angles))
    gt_angles = ground_truth.get('angles', []) if ground_truth else []
    
    angle_data = []
    for i in range(n_angles):
        row = {'Corner': f'C{i+1}'}
        
        fitted_val = fitted_angles[i] if i < len(fitted_angles) else None
        roomplan_val = roomplan_angles[i] if i < len(roomplan_angles) else None
        gt_val = gt_angles[i] if i < len(gt_angles) else None
        
        row['Fitted (°)'] = round(fitted_val, decimals) if fitted_val is not None else None
        row['RoomPlan (°)'] = round(roomplan_val, decimals) if roomplan_val is not None else None
        
        if gt_val is not None:
            row['GroundTruth (°)'] = round(float(gt_val), decimals)
        
        # Diff vs RoomPlan
        if fitted_val is not None and roomplan_val is not None:
            diff = fitted_val - roomplan_val
            row['Diff_RP (°)'] = round(diff, decimals)
        else:
            row['Diff_RP (°)'] = None
        
        # Diff vs Ground Truth
        if gt_val is not None and fitted_val is not None:
            diff_gt = fitted_val - float(gt_val)
            row['Diff_GT (°)'] = round(diff_gt, decimals)
        
        angle_data.append(row)
    
    results['angles'] = pd.DataFrame(angle_data)
    
    # ========================================================================
    # Summary Comparison
    # ========================================================================
    
    summary_data = []
    
    # Perimeter
    f_perim = float(fitted.get('perimeter', 0))
    r_perim = float(roomplan.get('perimeter', 0))
    gt_perim = float(ground_truth.get('perimeter', 0)) if ground_truth and ground_truth.get('perimeter') else None
    
    perim_row = {
        'Metric': 'Perimeter (m)',
        'Fitted': round(f_perim, decimals),
        'RoomPlan': round(r_perim, decimals),
        'Diff_RP': round(f_perim - r_perim, decimals),
        'Diff_RP (%)': round((f_perim - r_perim) / r_perim * 100, 2) if r_perim != 0 else 0
    }
    if gt_perim:
        perim_row['GroundTruth'] = round(gt_perim, decimals)
        perim_row['Diff_GT'] = round(f_perim - gt_perim, decimals)
    summary_data.append(perim_row)
    
    # Area
    f_area = float(fitted.get('area', 0))
    r_area = float(roomplan.get('area', 0))
    gt_area = float(ground_truth.get('area', 0)) if ground_truth and ground_truth.get('area') else None
    
    area_row = {
        'Metric': 'Area (m²)',
        'Fitted': round(f_area, decimals),
        'RoomPlan': round(r_area, decimals),
        'Diff_RP': round(f_area - r_area, decimals),
        'Diff_RP (%)': round((f_area - r_area) / r_area * 100, 2) if r_area != 0 else 0
    }
    if gt_area:
        area_row['GroundTruth'] = round(gt_area, decimals)
        area_row['Diff_GT'] = round(f_area - gt_area, decimals)
    summary_data.append(area_row)
    
    # Number of corners
    f_corners = fitted.get('n_corners', len(fitted.get('corners', [])))
    r_corners = roomplan.get('n_corners', len(roomplan.get('corners', [])))
    
    corners_row = {
        'Metric': 'Corners',
        'Fitted': f_corners,
        'RoomPlan': r_corners,
        'Diff_RP': f_corners - r_corners,
        'Diff_RP (%)': None
    }
    summary_data.append(corners_row)
    
    results['summary'] = pd.DataFrame(summary_data)
    
    return results


def compare_with_roomplan_json(
    fitted: Dict,
    json_path: str,
    ground_truth: Optional[Dict] = None,
    decimals: int = 3,
    align_edges: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Compare fitted polygon with RoomPlan JSON file.
    """
    roomplan = extract_roomplan_values(json_path)
    return compare_polygons(fitted, roomplan, ground_truth, decimals, align_edges)


def print_comparison(comparison: Dict[str, pd.DataFrame], title: str = "Polygon Comparison"):
    """Pretty print comparison results."""
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    if 'summary' in comparison:
        print(f"\n📊 SUMMARY")
        print(comparison['summary'].to_string(index=False))
    
    if 'edges' in comparison:
        print(f"\n📏 EDGES")
        print(comparison['edges'].to_string(index=False))
    
    if 'angles' in comparison:
        print(f"\n📐 CORNER ANGLES")
        print(comparison['angles'].to_string(index=False))
    
    # Calculate statistics
    if 'edges' in comparison:
        diff_col = 'Diff_RP (cm)' if 'Diff_RP (cm)' in comparison['edges'].columns else 'Diff (cm)'
        if diff_col in comparison['edges'].columns:
            edge_diffs = comparison['edges'][diff_col].dropna()
            if len(edge_diffs) > 0:
                print(f"\n📈 EDGE STATISTICS (vs RoomPlan)")
                print(f"   Mean absolute error: {edge_diffs.abs().mean():.2f} cm")
                print(f"   Max error: {edge_diffs.abs().max():.2f} cm")
                print(f"   RMS error: {np.sqrt((edge_diffs**2).mean()):.2f} cm")
        
        # Ground truth stats if available
        if 'Diff_GT (cm)' in comparison['edges'].columns:
            gt_diffs = comparison['edges']['Diff_GT (cm)'].dropna()
            if len(gt_diffs) > 0:
                print(f"\n📈 EDGE STATISTICS (vs Ground Truth)")
                print(f"   Mean absolute error: {gt_diffs.abs().mean():.2f} cm")
                print(f"   Max error: {gt_diffs.abs().max():.2f} cm")
                print(f"   RMS error: {np.sqrt((gt_diffs**2).mean()):.2f} cm")


def export_to_dataframe(comparison: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Export all comparison results to a single DataFrame.
    
    Returns combined DataFrame with all metrics.
    """
    
    dfs = []
    
    for name, df in comparison.items():
        df_copy = df.copy()
        df_copy['Category'] = name.capitalize()
        dfs.append(df_copy)
    
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Polygon Comparison Module")
    print("=" * 50)
    
    # Test with room.json
    json_path = "/mnt/user-data/uploads/room.json"
    
    try:
        roomplan = extract_roomplan_values(json_path)
        
        print("\nRoomPlan extracted values:")
        print(f"  Corners: {roomplan['n_corners']}")
        print(f"  Edges: {[round(e, 3) for e in roomplan['edges']]}")
        print(f"  Corner angles: {[round(a, 1) for a in roomplan['corner_angles']]}")
        print(f"  Perimeter: {roomplan['perimeter']:.3f}m")
        print(f"  Area: {roomplan['area']:.3f}m²")
        
        # Example with ground truth
        ground_truth = {
            'edges': [3.59, 4.51, 2.87, 0.99, 3.82],
            'angles': [90, 90, 90, 136, 134],
            'perimeter': 15.78,
            'area': 15.94
        }
        
        # Simulated fitted result (edges in different order)
        fitted = {
            'edges': [4.50, 2.88, 1.00, 3.80, 3.58],  # Rotated order
            'corner_angles': [90.0, 90.0, 135.5, 134.5, 90.0],
            'perimeter': 15.76,
            'area': 15.90,
            'n_corners': 5
        }
        
        # Compare (should auto-align edges)
        comparison = compare_polygons(fitted, roomplan, ground_truth, decimals=3)
        print_comparison(comparison, "Test Comparison (with auto-alignment)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
