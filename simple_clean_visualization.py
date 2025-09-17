#!/usr/bin/env python3
"""
Simple script to create clean visualization from your existing results
Run this directly with your labeled point cloud
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def create_clean_visualization(labeled_points, hide_clutter=True, emphasis_mode='furniture'):
    """
    Create clean visualization emphasizing important elements
    
    Args:
        labeled_points: Your labeled point cloud (N, 7) [x, y, z, r, g, b, label]
        hide_clutter: Remove clutter points (class 12)
        emphasis_mode: 'furniture', 'structural', or 'balanced'
    """
    
    class_names = [
        "ceiling", "floor", "wall", "beam", "column", 
        "window", "door", "table", "chair", "sofa", 
        "bookcase", "board", "clutter"
    ]
    
    # Enhanced colors for better visibility
    if emphasis_mode == 'furniture':
        # Emphasize furniture, tone down structure
        colors = np.array([
            [0.9, 0.9, 0.9],   # ceiling - very light (de-emphasized)
            [0.3, 0.2, 0.1],   # floor - dark brown (context)
            [0.6, 0.6, 0.5],   # wall - muted beige (context)
            [0.4, 0.2, 0.0],   # beam - dark brown
            [0.5, 0.5, 0.5],   # column - gray
            [0.0, 0.6, 1.0],   # window - bright blue (important)
            [1.0, 0.6, 0.0],   # door - bright orange (important)
            [0.9, 0.1, 0.1],   # table - bright red (EMPHASIS)
            [1.0, 0.0, 0.0],   # chair - bright red (EMPHASIS)
            [0.0, 0.8, 0.0],   # sofa - bright green (EMPHASIS)
            [0.7, 0.0, 0.7],   # bookcase - bright purple (EMPHASIS)
            [0.2, 0.2, 0.2],   # board - dark
            [0.8, 0.8, 0.8],   # clutter - light gray (will be hidden)
        ])
        
    elif emphasis_mode == 'structural':
        # Emphasize architectural elements
        colors = np.array([
            [0.8, 0.8, 0.9],   # ceiling - light blue-gray (visible)
            [0.4, 0.2, 0.1],   # floor - brown (EMPHASIS)
            [0.7, 0.7, 0.6],   # wall - beige (EMPHASIS)
            [0.5, 0.3, 0.1],   # beam - brown (EMPHASIS)
            [0.6, 0.6, 0.6],   # column - gray (EMPHASIS)
            [0.0, 0.5, 1.0],   # window - blue (EMPHASIS)
            [1.0, 0.5, 0.0],   # door - orange (EMPHASIS)
            [0.6, 0.3, 0.3],   # table - muted red
            [0.6, 0.3, 0.3],   # chair - muted red
            [0.3, 0.6, 0.3],   # sofa - muted green
            [0.5, 0.3, 0.5],   # bookcase - muted purple
            [0.3, 0.3, 0.3],   # board - dark gray
            [0.8, 0.8, 0.8],   # clutter - light gray
        ])
        
    else:  # balanced
        colors = np.array([
            [0.85, 0.85, 0.85], # ceiling - light gray
            [0.4, 0.2, 0.1],    # floor - brown
            [0.65, 0.65, 0.55], # wall - beige
            [0.3, 0.15, 0.0],   # beam - dark brown
            [0.6, 0.6, 0.6],    # column - gray
            [0.0, 0.6, 1.0],    # window - blue
            [1.0, 0.5, 0.0],    # door - orange
            [0.8, 0.2, 0.2],    # table - red
            [0.9, 0.1, 0.1],    # chair - bright red
            [0.1, 0.7, 0.1],    # sofa - green
            [0.6, 0.1, 0.6],    # bookcase - purple
            [0.2, 0.2, 0.2],    # board - dark
            [0.7, 0.7, 0.7],    # clutter - gray
        ])
    
    # Extract points and labels
    points = labeled_points[:, :3]
    labels = labeled_points[:, 6].astype(int)
    
    # Filter points
    if hide_clutter:
        mask = labels != 12  # Remove clutter
        points = points[mask]
        labels = labels[mask]
        print(f"Removed {np.sum(~mask)} clutter points")
    
    # Create colored point cloud
    point_colors = np.zeros((len(points), 3))
    for i, label in enumerate(labels):
        if 0 <= label < len(colors):
            point_colors[i] = colors[label]
        else:
            point_colors[i] = [0.5, 0.5, 0.5]  # Gray for unknown
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    return pcd, colors

def visualize_clean_scene(labeled_points, save_path=None, emphasis_mode='furniture'):
    """
    Create and display clean visualization
    """
    
    print(f"Creating clean visualization with {emphasis_mode} emphasis...")
    
    # Create clean point cloud
    pcd, colors = create_clean_visualization(
        labeled_points, 
        hide_clutter=True, 
        emphasis_mode=emphasis_mode
    )
    
    # Save if requested
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Clean point cloud saved to: {save_path}")
    
    # Display
    print("Opening 3D viewer...")
    print("Controls: Mouse to rotate, scroll to zoom, ESC to close")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Clean Semantic Segmentation - {emphasis_mode.title()} Mode", 
        width=1400, 
        height=900
    )
    vis.add_geometry(pcd)
    
    # Enhanced rendering settings
    render_option = vis.get_render_option()
    render_option.point_size = 2.5
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Dark background
    
    # Better viewing angle
    view_control = vis.get_view_control()
    view_control.set_front([0.2, -0.5, -0.3])
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()
    
    return pcd

def create_clean_legend(labeled_points, emphasis_mode='furniture', save_path=None):
    """
    Create legend showing only visible/important classes
    """
    
    class_names = [
        "ceiling", "floor", "wall", "beam", "column", 
        "window", "door", "table", "chair", "sofa", 
        "bookcase", "board", "clutter"
    ]
    
    # Get colors used
    _, colors = create_clean_visualization(labeled_points, hide_clutter=True, emphasis_mode=emphasis_mode)
    
    # Count points per class (excluding clutter)
    labels = labeled_points[:, 6].astype(int)
    mask = labels != 12  # Exclude clutter
    filtered_labels = labels[mask]
    
    class_counts = {}
    total_visible = len(filtered_labels)
    
    for i, class_name in enumerate(class_names):
        if i == 12:  # Skip clutter
            continue
        count = np.sum(filtered_labels == i)
        if count > 0:
            percentage = (count / total_visible) * 100
            class_counts[class_name] = {'count': count, 'percentage': percentage}
    
    # Create legend
    active_classes = [(name, stats) for name, stats in class_counts.items()]
    active_classes.sort(key=lambda x: x[1]['count'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (class_name, stats) in enumerate(active_classes):
        class_idx = class_names.index(class_name)
        color = colors[class_idx]
        
        # Format count
        count_text = f"{stats['count']:,}"
        if stats['count'] >= 1000:
            count_text = f"{stats['count']/1000:.1f}K"
        
        # Create bar
        ax.barh(i, 1, color=color, alpha=0.9, edgecolor='black', linewidth=1)
        
        # Add label
        label = f"{class_name.capitalize()}: {count_text} points ({stats['percentage']:.1f}%)"
        text_color = 'white' if np.mean(color) < 0.5 else 'black'
        ax.text(0.03, i, label, va='center', fontsize=12, fontweight='bold', color=text_color)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(active_classes) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'Clean Semantic Segmentation - {emphasis_mode.title()} Mode (Clutter Removed)', 
                fontsize=16, fontweight='bold', pad=20)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Clean legend saved to: {save_path}")
    
    plt.show()

def create_comparison_view(labeled_points, save_path=None):
    """
    Create side-by-side comparison: original vs clean
    """
    
    # Original with clutter
    pcd_original, _ = create_clean_visualization(labeled_points, hide_clutter=False, emphasis_mode='balanced')
    
    # Clean without clutter
    pcd_clean, _ = create_clean_visualization(labeled_points, hide_clutter=True, emphasis_mode='furniture')
    
    # Move clean version to the side
    bbox = pcd_clean.get_axis_aligned_bounding_box()
    width = bbox.max_bound[0] - bbox.min_bound[0]
    pcd_clean.translate([width * 1.2, 0, 0])
    
    # Combine for visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Comparison: Original vs Clean", width=1600, height=800)
    vis.add_geometry(pcd_original)
    vis.add_geometry(pcd_clean)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    
    print("Left: Original with clutter | Right: Clean without clutter")
    vis.run()
    vis.destroy_window()
    
    if save_path:
        # Save separate files
        o3d.io.write_point_cloud(save_path.replace('.ply', '_original.ply'), pcd_original)
        o3d.io.write_point_cloud(save_path.replace('.ply', '_clean.ply'), pcd_clean)

def interactive_clean_visualization(labeled_points, output_dir="clean_visualization"):
    """
    Interactive menu for different visualization modes
    """
    
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    while True:
        print("\n" + "="*50)
        print("CLEAN VISUALIZATION OPTIONS")
        print("="*50)
        print("1. Furniture Focus (bright furniture, muted structure)")
        print("2. Structural Focus (walls, doors, windows emphasized)")
        print("3. Balanced View (everything visible)")
        print("4. Comparison View (original vs clean)")
        print("5. Create all views and save")
        print("6. Exit")
        
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == '1':
            pcd = visualize_clean_scene(labeled_points, 
                                      save_path=output_path / "furniture_focus.ply",
                                      emphasis_mode='furniture')
            create_clean_legend(labeled_points, 'furniture', output_path / "legend_furniture.png")
            
        elif choice == '2':
            pcd = visualize_clean_scene(labeled_points,
                                      save_path=output_path / "structural_focus.ply", 
                                      emphasis_mode='structural')
            create_clean_legend(labeled_points, 'structural', output_path / "legend_structural.png")
            
        elif choice == '3':
            pcd = visualize_clean_scene(labeled_points,
                                      save_path=output_path / "balanced_view.ply",
                                      emphasis_mode='balanced')
            create_clean_legend(labeled_points, 'balanced', output_path / "legend_balanced.png")
            
        elif choice == '4':
            create_comparison_view(labeled_points, save_path=output_path / "comparison.ply")
            
        elif choice == '5':
            print("Creating all views...")
            
            # Create all three modes
            for mode in ['furniture', 'structural', 'balanced']:
                print(f"Creating {mode} view...")
                pcd = visualize_clean_scene(labeled_points,
                                          save_path=output_path / f"{mode}_view.ply",
                                          emphasis_mode=mode)
                create_clean_legend(labeled_points, mode, output_path / f"legend_{mode}.png")
            
            # Create comparison
            create_comparison_view(labeled_points, save_path=output_path / "comparison.ply")
            
            print(f"All views saved to: {output_dir}")
            break
            
        elif choice == '6':
            break
            
        else:
            print("Invalid choice. Please try again.")

# Quick functions for direct use
def quick_furniture_view(labeled_points, save_path="furniture_view.ply"):
    """Quick function to create furniture-focused view"""
    return visualize_clean_scene(labeled_points, save_path, 'furniture')

def quick_clean_view(labeled_points, save_path="clean_view.ply"):
    """Quick function to create clean balanced view"""
    return visualize_clean_scene(labeled_points, save_path, 'balanced')

def quick_structural_view(labeled_points, save_path="structural_view.ply"):
    """Quick function to create structural-focused view"""
    return visualize_clean_scene(labeled_points, save_path, 'structural')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean visualization without clutter")
    parser.add_argument("--labeled_points", type=str, required=True, 
                       help="Path to labeled_pointcloud.npy file")
    parser.add_argument("--mode", type=str, choices=['furniture', 'structural', 'balanced', 'interactive'], 
                       default='interactive', help="Visualization mode")
    parser.add_argument("--output_dir", type=str, default="clean_visualization", 
                       help="Output directory for saved files")
    parser.add_argument("--save_only", action="store_true", 
                       help="Save files without opening viewer")
    
    args = parser.parse_args()
    
    # Load labeled points
    try:
        labeled_points = np.load(args.labeled_points)
        print(f"Loaded {len(labeled_points)} labeled points")
        
        # Check data format
        if labeled_points.shape[1] != 7:
            raise ValueError(f"Expected 7 columns [x,y,z,r,g,b,label], got {labeled_points.shape[1]}")
            
    except Exception as e:
        print(f"Error loading labeled points: {e}")
        print("Make sure the file exists and contains the correct format")
        exit(1)
    
    # Run visualization based on mode
    if args.mode == 'interactive':
        interactive_clean_visualization(labeled_points, args.output_dir)
        
    elif args.save_only:
        from pathlib import Path
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pcd, _ = create_clean_visualization(labeled_points, hide_clutter=True, emphasis_mode=args.mode)
        save_file = output_path / f"{args.mode}_clean.ply"
        o3d.io.write_point_cloud(str(save_file), pcd)
        print(f"Clean visualization saved to: {save_file}")
        
        create_clean_legend(labeled_points, args.mode, output_path / f"legend_{args.mode}.png")
        
    else:
        from pathlib import Path
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualize_clean_scene(labeled_points, 
                            save_path=output_path / f"{args.mode}_view.ply",
                            emphasis_mode=args.mode)
        create_clean_legend(labeled_points, args.mode, output_path / f"legend_{args.mode}.png")


"""
QUICK USAGE EXAMPLES:

1. Interactive menu (recommended):
   python simple_clean_visualization.py --labeled_points labeled_pointcloud.npy

2. Direct furniture-focused view:
   python simple_clean_visualization.py --labeled_points labeled_pointcloud.npy --mode furniture

3. Save all views without opening viewer:
   python simple_clean_visualization.py --labeled_points labeled_pointcloud.npy --mode furniture --save_only

4. Use in your existing code:
   
   from simple_clean_visualization import quick_furniture_view, quick_clean_view
   
   # Quick furniture-focused view (bright furniture, muted structure)
   quick_furniture_view(labeled_points, "furniture_focus.ply")
   
   # Clean balanced view (no clutter, everything visible)
   quick_clean_view(labeled_points, "clean_room.ply")

WHAT THIS DOES:
✅ Removes all clutter points (19.5% → 0%)
✅ Bright colors for furniture (tables, chairs, sofas)
✅ Muted but visible architectural elements
✅ Clean legends showing only visible classes
✅ Multiple viewing modes for different analysis needs
✅ Dark background for better contrast
✅ Enhanced point sizes for better visibility

MODES:
- furniture: Emphasizes furniture, mutes structure
- structural: Emphasizes walls/doors/windows, mutes furniture  
- balanced: Everything visible and balanced
- interactive: Menu to try all modes
"""