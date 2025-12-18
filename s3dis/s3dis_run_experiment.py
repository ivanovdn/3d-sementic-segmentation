import yaml
from tqdm import tqdm

import wandb
from pcd_dataset import PointCloudDataset
from plane_detector import StructuralRANSAC
from pointnet_utils import PointnetInference
from s3dis_metrics import SegmentationMetrics
from semantic_segmentation import SemanticSegmentor

token = "415dd2b149285b382a2e695e3023f569695b8398"
wandb.login(key=token)

rooms = ["office_1", "office_2", "office_3", "office_4", "office_5", "office_6"]

with open("config.yaml") as f:
    config = yaml.safe_load(f)

s_val = PointCloudDataset("../s3DIS/Stanford3dDataset_v1.2_Aligned_Version/")

metrics = SegmentationMetrics(config["elements"])

for room in tqdm(rooms):
    config["room"] = room
    sem_seg = SemanticSegmentor(s_val, StructuralRANSAC, PointnetInference, config)
    if config["approach"] == "ransac-pointnet":
        sem_seg.ransac_segment()
        sem_seg.refind_walls()
        sem_seg.refined_with_pointnet()
        predictions = sem_seg.map_ransac_to_s3dis()
    else:
        predictions = sem_seg.pointnet_segmentation(sem_seg.points, "all")

    results = metrics.compute_metrics(sem_seg.labels, predictions)
    print(f"Overall Accuracy: {results['overall_acc']:.2%}")

    print("\nPer-class IoU:")
    for class_name, iou in results["iou_per_class"].items():
        print(f"  {class_name}: {iou:.2%}")

metrics.compute_mean_iou()

config["room"] = rooms
run = wandb.init(
    project="point-cloud-sem-seg",
    config=config,
)
run.log({"mean_iou": metrics.mean_iou})
run.log(metrics.results)
run.finish()
