# File for simple count evaluation between ground truth and predictions

import json

GT = "/storage/soltau/data/model_outputs/dino_output/tile_0_0.json"
PRED = "/storage/soltau/data/prototype_results/dino_output/detections.geojson"

with open(GT) as f:
    gt = json.load(f)

with open(PRED) as f:
    pred = json.load(f)

gt_count = len(gt["shapes"])

pred_count = len(pred)

print("/n==== COUNT EVALUATION ==== /n")
print(f"GT objects: {gt_count}")
print(f"Pred objects: {pred_count}")
print(f"Difference: {pred_count - gt_count}")