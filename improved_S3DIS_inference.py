import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

class ImprovedS3DISInference:
    """
    Improved inference class with better preprocessing and prediction quality
    """
    
    def __init__(self, model_path, model_type="pointnet2", num_classes=13):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.class_names = [
            "ceiling", "floor", "wall", "beam", "column", 
            "window", "door", "table", "chair", "sofa", 
            "bookcase", "board", "clutter"
        ]
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        """Load the pretrained model"""
        if self.model_type == "pointnet2":
            from pointnet2_sem_seg import get_model as get_pointnet2_model
            model = get_pointnet2_model(self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        return model
    
    def improved_preprocessing(self, pcd_path, block_size=1.0, stride=0.3, num_point=4096, 
                             min_points_per_block=512, use_adaptive_sampling=True):
        """
        Improved preprocessing with better coverage and quality
        
        Args:
            pcd_path: Path to point cloud
            block_size: Size of each block (try smaller for better coverage)
            stride: Stride for sliding window (smaller = more overlap = better coverage)
            num_point: Points per block
            min_points_per_block: Minimum points to consider a block valid
            use_adaptive_sampling: Use adaptive sampling based on point density
        """
        
        from mesh_preprocessing import read_pcd_and_extract_points
        
        print("Loading point cloud...")
        points = read_pcd_and_extract_points(pcd_path)
        original_points = points.copy()
        
        print(f"Original point cloud: {len(points)} points")
        
        # Step 1: Normalize coordinates
        xyz_min = np.min(points[:, :3], axis=0)
        points[:, :3] -= xyz_min
        coord_max = np.max(points[:, :3], axis=0)
        
        print(f"Point cloud bounds: {coord_max}")
        
        # Step 2: Ensure colors are in [0,1] range
        if np.max(points[:, 3:6]) > 1.0:
            points[:, 3:6] = points[:, 3:6] / 255.0
        
        # Step 3: Improved block generation with better coverage
        data_blocks, block_indices = self._create_improved_blocks(
            points, coord_max, block_size, stride, num_point, 
            min_points_per_block, use_adaptive_sampling
        )
        
        print(f"Created {len(data_blocks)} blocks with improved coverage")
        
        # Step 4: Verify coverage
        covered_points = set()
        for indices in block_indices:
            covered_points.update(indices)
        
        coverage_percentage = (len(covered_points) / len(points)) * 100
        print(f"Point coverage: {len(covered_points)}/{len(points)} ({coverage_percentage:.1f}%)")
        
        return data_blocks, block_indices, original_points
    
    def _create_improved_blocks(self, points, coord_max, block_size, stride, num_point, 
                               min_points_per_block, use_adaptive_sampling):
        """
        Create blocks with improved coverage and quality
        """
        
        # Calculate grid with better coverage
        grid_x = int(np.ceil(coord_max[0] / stride))
        grid_y = int(np.ceil(coord_max[1] / stride))
        
        print(f"Grid size: {grid_x} x {grid_y} = {grid_x * grid_y} potential blocks")
        
        data_blocks = []
        block_indices = []
        valid_blocks = 0
        
        # Add some randomness to improve coverage
        padding = 0.05  # 5cm padding
        
        for index_y in range(grid_y):
            for index_x in range(grid_x):
                # Calculate block boundaries with overlap
                s_x = index_x * stride
                e_x = min(s_x + block_size, coord_max[0])
                
                s_y = index_y * stride
                e_y = min(s_y + block_size, coord_max[1])
                
                # Expand boundaries slightly for better coverage
                s_x = max(0, s_x - padding)
                s_y = max(0, s_y - padding)
                e_x = min(coord_max[0], e_x + padding)
                e_y = min(coord_max[1], e_y + padding)
                
                # Find points in block
                point_idxs = np.where(
                    (points[:, 0] >= s_x) & (points[:, 0] <= e_x) & 
                    (points[:, 1] >= s_y) & (points[:, 1] <= e_y)
                )[0]
                
                if len(point_idxs) < min_points_per_block:
                    continue
                
                # Adaptive sampling based on point density
                if use_adaptive_sampling:
                    selected_points, selected_indices = self._adaptive_sampling(
                        points[point_idxs], point_idxs, num_point
                    )
                else:
                    # Standard sampling
                    if len(point_idxs) >= num_point:
                        selected_idx = np.random.choice(point_idxs, num_point, replace=False)
                    else:
                        selected_idx = np.random.choice(point_idxs, num_point, replace=True)
                    
                    selected_points = points[selected_idx]
                    selected_indices = selected_idx
                
                # Create 9-channel format
                current_points = np.zeros((num_point, 9))
                
                # Normalized coordinates (channels 6,7,8)
                current_points[:, 6] = selected_points[:, 0] / coord_max[0]
                current_points[:, 7] = selected_points[:, 1] / coord_max[1]
                current_points[:, 8] = selected_points[:, 2] / coord_max[2]
                
                # Center the block
                block_center_x = (s_x + e_x) / 2.0
                block_center_y = (s_y + e_y) / 2.0
                
                centered_points = selected_points.copy()
                centered_points[:, 0] -= block_center_x
                centered_points[:, 1] -= block_center_y
                
                # Copy xyz and rgb (channels 0-5)
                current_points[:, 0:6] = centered_points
                
                data_blocks.append(current_points)
                block_indices.append(selected_indices)
                valid_blocks += 1
        
        print(f"Valid blocks created: {valid_blocks}")
        return data_blocks, block_indices
    
    def _adaptive_sampling(self, block_points, point_indices, num_point):
        """
        Adaptive sampling that tries to preserve important geometric features
        """
        
        if len(block_points) <= num_point:
            # If not enough points, duplicate with some noise
            selected_idx = np.random.choice(len(block_points), num_point, replace=True)
            selected_points = block_points[selected_idx]
            selected_indices = point_indices[selected_idx]
            return selected_points, selected_indices
        
        # Strategy 1: Keep points with diverse heights (to capture walls, ceiling, floor)
        z_values = block_points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        
        if z_max - z_min > 0.5:  # If there's significant height variation
            # Sample from different height levels
            n_levels = 5
            samples_per_level = num_point // n_levels
            
            selected_indices_list = []
            
            for i in range(n_levels):
                z_start = z_min + i * (z_max - z_min) / n_levels
                z_end = z_min + (i + 1) * (z_max - z_min) / n_levels
                
                level_mask = (z_values >= z_start) & (z_values <= z_end)
                level_indices = np.where(level_mask)[0]
                
                if len(level_indices) > 0:
                    n_samples = min(samples_per_level, len(level_indices))
                    if len(level_indices) >= n_samples:
                        sampled = np.random.choice(level_indices, n_samples, replace=False)
                    else:
                        sampled = np.random.choice(level_indices, n_samples, replace=True)
                    selected_indices_list.extend(sampled)
            
            # Fill remaining slots randomly
            remaining = num_point - len(selected_indices_list)
            if remaining > 0:
                available = list(set(range(len(block_points))) - set(selected_indices_list))
                if available:
                    additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                    selected_indices_list.extend(additional)
            
            # Ensure we have exactly num_point points
            if len(selected_indices_list) < num_point:
                additional_needed = num_point - len(selected_indices_list)
                additional = np.random.choice(len(block_points), additional_needed, replace=True)
                selected_indices_list.extend(additional)
            elif len(selected_indices_list) > num_point:
                selected_indices_list = selected_indices_list[:num_point]
            
            selected_idx = np.array(selected_indices_list)
        else:
            # Standard random sampling for flat areas
            selected_idx = np.random.choice(len(block_points), num_point, replace=False)
        
        selected_points = block_points[selected_idx]
        selected_indices = point_indices[selected_idx]
        
        return selected_points, selected_indices
    
    def run_improved_inference(self, pcd_path, output_path=None, 
                              block_size=1.0, stride=0.3, confidence_threshold=0.3,
                              use_tta=True):
        """
        Run inference with improved preprocessing and post-processing
        
        Args:
            pcd_path: Path to point cloud
            output_path: Output directory
            block_size: Block size (smaller = better coverage)
            stride: Stride (smaller = more overlap)
            confidence_threshold: Minimum confidence for predictions
            use_tta: Use test-time augmentation
        """
        
        print(f"Running improved inference on {pcd_path}")
        print(f"Settings: block_size={block_size}, stride={stride}, confidence_threshold={confidence_threshold}")
        
        # Improved preprocessing
        data_blocks, block_indices, original_points = self.improved_preprocessing(
            pcd_path, block_size=block_size, stride=stride
        )
        
        if len(data_blocks) == 0:
            raise ValueError("No valid blocks created from point cloud")
        
        # Run inference with confidence scoring
        all_predictions = []
        all_probabilities = []
        all_confidences = []
        
        with torch.no_grad():
            for i, block in enumerate(data_blocks):
                if i % 10 == 0:
                    print(f"Processing block {i+1}/{len(data_blocks)}")
                
                # Convert to tensor
                block_tensor = torch.FloatTensor(block).transpose(0, 1).unsqueeze(0)
                block_tensor = block_tensor.to(self.device)
                
                if use_tta:
                    # Test-time augmentation: multiple predictions with slight variations
                    predictions_tta = []
                    probabilities_tta = []
                    
                    for tta_iter in range(3):  # 3 TTA iterations
                        # Add small random noise for TTA
                        if tta_iter > 0:
                            noise = torch.randn_like(block_tensor[:, :3, :]) * 0.01
                            block_tta = block_tensor.clone()
                            block_tta[:, :3, :] += noise
                        else:
                            block_tta = block_tensor
                        
                        # Forward pass
                        pred, _ = self.model(block_tta)
                        pred = pred.contiguous().view(-1, self.num_classes)
                        pred_probs = F.softmax(pred, dim=1)
                        
                        predictions_tta.append(pred_probs.cpu().numpy())
                    
                    # Average TTA predictions
                    avg_probs = np.mean(predictions_tta, axis=0)
                    pred_choice = np.argmax(avg_probs, axis=1)
                    
                    # Calculate confidence as max probability
                    confidences = np.max(avg_probs, axis=1)
                    
                else:
                    # Standard inference
                    pred, _ = self.model(block_tensor)
                    pred = pred.contiguous().view(-1, self.num_classes)
                    pred_probs = F.softmax(pred, dim=1)
                    
                    avg_probs = pred_probs.cpu().numpy()
                    pred_choice = pred_probs.data.max(1)[1].cpu().numpy()
                    confidences = np.max(avg_probs, axis=1)
                
                # Apply confidence threshold - mark low confidence as clutter
                low_confidence_mask = confidences < confidence_threshold
                pred_choice[low_confidence_mask] = 12  # clutter
                
                all_predictions.append(pred_choice)
                all_probabilities.append(avg_probs)
                all_confidences.append(confidences)
        
        # Improved reconstruction with confidence-weighted voting
        labeled_points = self._improved_reconstruction(
            original_points, all_predictions, all_confidences, block_indices
        )
        
        # Analyze results
        class_counts = self._analyze_predictions(all_predictions)
        
        # Save results
        if output_path:
            self._save_improved_results(
                all_predictions, all_probabilities, all_confidences,
                output_path, class_counts, labeled_points
            )
        
        return all_predictions, class_counts, labeled_points
    
    def _improved_reconstruction(self, original_points, predictions, confidences, block_indices):
        """
        Improved reconstruction using confidence-weighted voting
        """
        
        print("Reconstructing with confidence-weighted voting...")
        
        num_points = len(original_points)
        vote_weights = np.zeros((num_points, self.num_classes), dtype=np.float32)
        
        # Accumulate confidence-weighted votes
        for block_pred, block_conf, block_idx in zip(predictions, confidences, block_indices):
            for i, point_idx in enumerate(block_idx):
                if point_idx < num_points:
                    predicted_class = block_pred[i]
                    confidence = block_conf[i]
                    
                    # Weight vote by confidence
                    vote_weights[point_idx, predicted_class] += confidence
        
        # Assign labels based on weighted votes
        point_labels = np.argmax(vote_weights, axis=1)
        
        # Handle points with no votes
        no_votes_mask = np.sum(vote_weights, axis=1) == 0
        point_labels[no_votes_mask] = 12  # clutter
        
        print(f"Points with no votes: {np.sum(no_votes_mask)} ({np.sum(no_votes_mask)/num_points*100:.1f}%)")
        
        # Combine with original points
        labeled_points = np.column_stack([original_points, point_labels])
        
        return labeled_points
    
    def _analyze_predictions(self, predictions):
        """Analyze prediction results"""
        all_preds = np.concatenate(predictions)
        class_counts = {}
        total_points = len(all_preds)
        
        for class_idx in range(self.num_classes):
            count = np.sum(all_preds == class_idx)
            percentage = (count / total_points) * 100
            class_counts[self.class_names[class_idx]] = {
                "count": int(count),
                "percentage": float(percentage),
            }
        
        return class_counts
    
    def _save_improved_results(self, predictions, probabilities, confidences, 
                              output_path, class_counts, labeled_points):
        """Save enhanced results"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / "predictions.npy", predictions)
        np.save(output_dir / "probabilities.npy", probabilities)
        np.save(output_dir / "confidences.npy", confidences)
        np.save(output_dir / "labeled_pointcloud.npy", labeled_points)
        
        import json
        with open(output_dir / "class_analysis.json", "w") as f:
            json.dump(class_counts, f, indent=2)
        
        print(f"Enhanced results saved to {output_path}")


# Usage example
def run_high_quality_inference(pcd_path, model_path, output_dir="improved_results"):
    """
    Run high-quality inference with improved settings
    """
    
    # Initialize improved inference
    inference = ImprovedS3DISInference(model_path=model_path, model_type='pointnet2')
    
    # Run with optimized settings
    predictions, class_counts, labeled_points = inference.run_improved_inference(
        pcd_path=pcd_path,
        output_path=output_dir,
        block_size=0.8,          # Smaller blocks for better coverage
        stride=0.2,              # Much smaller stride for more overlap
        confidence_threshold=0.4, # Higher confidence threshold
        use_tta=True             # Test-time augmentation
    )
    
    # Print results
    print("\n" + "="*50)
    print("IMPROVED SEMANTIC SEGMENTATION RESULTS")
    print("="*50)
    
    for class_name, stats in class_counts.items():
        print(f"{class_name:12s}: {stats['count']:8d} points ({stats['percentage']:5.1f}%)")
    
    return labeled_points, class_counts

    def print_results(self, class_counts):
        """Print inference results"""

        print("\n" + "=" * 50)
        print("SEMANTIC SEGMENTATION RESULTS")
        print("=" * 50)

        for class_name, stats in class_counts.items():
            print(
                f"{class_name:12s}: {stats['count']:8d} points ({stats['percentage']:5.1f}%)"
            )

        print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-quality semantic segmentation")
    parser.add_argument("--pcd_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="improved_results")
    parser.add_argument("--block_size", type=float, default=0.8)
    parser.add_argument("--stride", type=float, default=0.2)
    parser.add_argument("--confidence_threshold", type=float, default=0.4)
    
    args = parser.parse_args()
    
    labeled_points, class_counts = run_high_quality_inference(
        args.pcd_path, args.model_path, args.output_dir
    )