import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class SegmentationMetrics:
    def __init__(self, gt_labels, pred_labels, class_names):
        self.gt = gt_labels
        self.pred = pred_labels
        self.classes = class_names
        self.structural_only = [0, 1, 2, 3, 4, 5, 6]

    def compute_metrics(self):

        results = {}

        mask = self.gt >= 0
        results["overall_acc"] = np.mean(self.pred[mask] == self.gt[mask])

        results["iou_per_class"] = self.compute_iou()

        struct_mask = np.isin(self.gt, self.structural_only)
        results["structural_acc"] = np.mean(
            self.pred[struct_mask] == self.gt[struct_mask]
        )

        results["confusion"] = confusion_matrix(self.gt[mask], self.pred[mask])

        # results['classification_report'] = classification_report(
        #     self.gt[mask], self.pred[mask],
        #     target_names=self.classes,
        #     output_dict=True
        # )

        return results

    def compute_iou(self):

        ious = {}
        for class_id, class_name in enumerate(self.classes):

            gt_mask = self.gt == class_id
            pred_mask = self.pred == class_id

            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)

            ious[class_name] = intersection / union if union > 0 else 0

        return ious
