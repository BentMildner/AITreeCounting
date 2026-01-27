from samgeo import SamGeo2
from test_bboxes import bounding_data
import os

# Paths
image_path = "/storage/soltau/data/tiles/tile_test.tif"
raw_mask = "mask_raw.tif"
clean_mask = "tree_masks.tif"
out_vector = "tree_vector.geojson"

# 1. extract bboxes from provided geojson
def get_bbox_list(geojson):
    boxes = []
    for feature in geojson['features']:
        coords = feature['geometry']['coordinates'][0]
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return boxes

input_boxes = get_bbox_list(bounding_data)

# 2. SAM2 Prediction
sam = SamGeo2(model_id="sam2-hiera-large", automatic=False)
sam.set_image(image_path)
sam.predict(
    boxes=input_boxes, 
    point_crs="EPSG:4326", 
    output=raw_mask, 
    dtype="uint8"
)

# 3. Post-Processing: Region-Growing und Speichern
sam.region_groups(
    raw_mask, 
    min_size=200, 
    out_vector=out_vector, 
    out_image=clean_mask
)

print(f"Post-processing abgeschlossen. Dateien gespeichert: {clean_mask}, {out_vector}")