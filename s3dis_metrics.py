import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class SegmentationMetrics:
    def __init__(self, elements):
        self.classes = {
            "ceiling": 0,
            "floor": 1,
            "wall": 2,
            "beam": 3,
            "column": 4,
            "window": 5,
            "door": 6,
            "table": 7,
            "chair": 8,
            "sofa": 9,
            "bookcase": 10,
            "board": 11,
            "clutter": 12,
        }
        self.elements = elements
        self.counter = 0
        self.mean_iou = 0.0
        self.results = {k: 0.0 for k in self.elements}

    def compute_metrics(self, labels, preds):
        results = {}
        mask = labels >= 0
        results["overall_acc"] = np.mean(preds[mask] == labels[mask])

        results["iou_per_class"] = self.compute_iou(labels, preds)

        results["confusion"] = confusion_matrix(labels[mask], preds[mask])

        # results['classification_report'] = classification_report(
        #     labels[mask], preds[mask],
        #     target_names=self.classes,
        #     output_dict=True
        # )
        self.counter += 1
        return results

    def compute_iou(self, labels, preds):
        ious = {}
        for class_id, class_name in enumerate(self.classes):

            gt_mask = labels == class_id
            pred_mask = preds == class_id

            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)

            ious[class_name] = intersection / union if union > 0 else 0
            if class_name in self.elements:
                self.results[class_name] += intersection / union if union > 0 else 0

        return ious

    def compute_mean_iou(self):
        self.results = {k: v / self.counter for k, v in self.results.items()}
        self.mean_iou = np.sum([i for i in self.results.values()]) / len(self.elements)
