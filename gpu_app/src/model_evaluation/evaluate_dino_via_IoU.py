# File to evaluate DINO model outputs using IoU metric --> Overkill for this project but useful for future reference


import json
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------

GROUND_TRUTH = "/storage/soltau/data/model_outputs/dino_output/tile_0_0.json"
PREDICTIONS = "/storage/soltau/data/prototype_results/dino_output/detections.geojson"

IOU_THRESHOLD = 0.2

# --------------------------------------


from shapely.geometry import shape
import json
import numpy as np



def load_geojson_boxes(path):
    with open(path) as f:
        geo = json.load(f)

    boxes = []

    if "shapes" in geo:
        featurs = geo["shapes"]
    elif "features" in geo:
        features = geo["features"]
    elif isinstance(geo, list):
        featurs = geo
    else:
        raise RuntimeError("Unknown annotation format")


    for feat in featurs:

        if "points" in feat:
            pts = feat["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
        
        else:
             geom = shape(feat["geometry"])
             minx, miny, maxx, maxy = geom.bounds
        
        boxes.append([minx, miny, maxx, maxy])

    return np.array(boxes)


def load_pred_boxes(path):
    with open(path) as f:
        data = json.load(f)

    boxes = []

    for feat in data["features"]:
        coords = np.array(feat["geometry"]["coordinates"][0])

        xs = coords[:, 0]
        ys = coords[:, 1]

        xmin = xs.min()
        xmax = xs.max()
        ymin = ys.min()
        ymax = ys.max()

        boxes.append([xmin, ymin, xmax, ymax])

    return np.array(boxes)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def evaluate(gt, preds):
    matched = set()
    TP = 0

    for g in gt:
        best = 0
        best_j = -1

        for j, p in enumerate(preds):
            if j in matched:
                continue
            v = iou(g, p)
            if v > best:
                best = v
                best_j = j

        if best > IOU_THRESHOLD:
            TP += 1
            matched.add(best_j)

    FP = len(preds) - TP
    FN = len(gt) - TP

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return TP, FP, FN, precision, recall, f1


if __name__ == "__main__":
    gt = load_geojson_boxes(GROUND_TRUTH)
    preds = load_pred_boxes(PREDICTIONS)

    print("GT sample:", gt[:3])
    print("Pred sample:", preds[:3])   
    TP, FP, FN, p, r, f1 = evaluate(gt, preds)

    print("\n====== DINO Evaluation ======\n")
    print(f"GT boxes: {len(gt)}")
    print(f"Pred boxes: {len(preds)}")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print("")
    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"F1 Score:  {f1:.3f}")