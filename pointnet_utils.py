import numpy as np
import torch
import torch.nn.functional as F

from pointnet2_sem_seg import get_model as get_pointnet2_model

device = torch.device("mps" if torch.mps.is_available() else "cpu")


def get_model():
    model = get_pointnet2_model(13)
    checkpoint = torch.load(
        "./models/best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def inference(data_blocks, type="wall"):
    model = get_model()
    all_predictions = []
    all_confidences = []
    with torch.no_grad():
        for i, block in enumerate(data_blocks):
            if i % 5 == 0:
                print(f"Processing furniture block {i+1}/{len(data_blocks)}")

            block_tensor = torch.FloatTensor(block).transpose(0, 1).unsqueeze(0)
            block_tensor = block_tensor.to(device)

            pred, _ = model(block_tensor)
            pred = pred.contiguous().view(-1, 13)
            pred_probs = F.softmax(pred, dim=1)

            confidences = torch.max(pred_probs, dim=1)[0]
            pred_choice = pred_probs.data.max(1)[1]

            if type == "wall":
                low_confidence_mask = confidences < 0.3
                pred_choice[low_confidence_mask] = 2  # wall
            else:
                low_confidence_mask = confidences < 0.3
                pred_choice[low_confidence_mask] = 12  # clutter

            all_predictions.append(pred_choice.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())
    return all_predictions, all_confidences


def create_pointnet_blocks(points, block_size, stride):
    """Create blocks for  segmentation"""

    xyz_min = np.min(points[:, :3], axis=0)
    points[:, :3] -= xyz_min

    if np.max(points[:, 3:6]) > 1.0:
        points[:, 3:6] = points[:, 3:6].astype(np.float32) / 255.0
    else:
        points[:, 3:6] = points[:, 3:6].astype(np.float32)

    coord_max = np.max(points[:, :3], axis=0)

    grid_x = int(np.ceil(coord_max[0] / stride))
    grid_y = int(np.ceil(coord_max[1] / stride))

    data_blocks = []
    block_indices = []

    for index_y in range(grid_y):
        for index_x in range(grid_x):
            s_x = index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_y = index_y * stride
            e_y = min(s_y + block_size, coord_max[1])

            point_idxs = np.where(
                (points[:, 0] >= s_x)
                & (points[:, 0] <= e_x)
                & (points[:, 1] >= s_y)
                & (points[:, 1] <= e_y)
            )[0]

            if len(point_idxs) < 100:
                continue

            num_point = min(4096, len(point_idxs) * 2)
            if len(point_idxs) >= num_point:
                selected_idxs = np.random.choice(point_idxs, num_point, replace=False)
            else:
                selected_idxs = np.random.choice(point_idxs, num_point, replace=True)

            selected_points = points[selected_idxs, :]
            current_points = np.zeros((num_point, 9))

            current_points[:, 6] = selected_points[:, 0] / coord_max[0]
            current_points[:, 7] = selected_points[:, 1] / coord_max[1]
            current_points[:, 8] = selected_points[:, 2] / coord_max[2]

            selected_points[:, 0] -= (s_x + e_x) / 2.0
            selected_points[:, 1] -= (s_y + e_y) / 2.0

            current_points[:, 0:6] = selected_points

            data_blocks.append(current_points)
            block_indices.append(selected_idxs)

    return data_blocks, block_indices


def reconstruct_furniture_labels(points, predictions, confidences, block_indices):
    """Reconstruct labels for furniture points using weighted voting"""

    num_points = len(points)
    vote_weights = np.zeros((num_points, 13), dtype=np.float32)

    # Accumulate votes
    for block_pred, block_conf, block_idx in zip(
        predictions, confidences, block_indices
    ):
        for i, point_idx in enumerate(block_idx):
            if point_idx < num_points:
                predicted_class = block_pred[i]
                confidence = block_conf[i]
                vote_weights[point_idx, predicted_class] += confidence

    # Assign labels
    point_labels = np.argmax(vote_weights, axis=1)

    # Handle points with no votes
    no_votes_mask = np.sum(vote_weights, axis=1) == 0
    point_labels[no_votes_mask] = 2  # clutter

    return point_labels
