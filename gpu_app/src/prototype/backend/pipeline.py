import os
from config import *
from tree_segmentation_service import TreeSegmentationService
from geotools import save_dino_as_geojson, save_sam_as_geojson



def run_pipeline(service: TreeSegmentationService):
    logger.info(
        f"Environment prepared on GPU {GPU_ID} with "
        f"{service.free_mem/1024**2:.0f} MiB free."
    )
    logger.info(f"LangSAM with {MODEL_TYPE} initialized.")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find input directory: {INPUT_PATH}"
        )

    boxes, logits, phrases, pil_img = service.run_prediction_dino(
        image_path=INPUT_PATH,
        text_prompt=PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        output_dir=DINO_OUTPUT_DIR
    )

    final_mask_np = None  # Default setzen

    if boxes is not None and len(boxes) > 0:
        masks_obj, final_mask_np = service.run_prediction_sam(
            pil_img=pil_img,
            boxes=boxes,
            output_dir=SAM_OUTPUT_DIR,
            image_path=INPUT_PATH
        )

    dino_geojson = os.path.join(DINO_OUTPUT_DIR, "detections.geojson")
    sam_geojson = os.path.join(SAM_OUTPUT_DIR, "masks.geojson") 

    # DINO immer speichern (auch wenn 0)
    save_dino_as_geojson(boxes, INPUT_PATH, dino_geojson)

    # SAM nur speichern, wenn Masken existieren
    if final_mask_np is not None:
        save_sam_as_geojson(final_mask_np, INPUT_PATH, sam_geojson)
    else:
        logger.warning("No SAM mask to save as GeoJSON.")

    logger.info(f"DINO found {len(boxes)} objects.")
    return {
        "count": len(boxes),
        "dino_geojson": dino_geojson,
        "sam_geojson": sam_geojson if final_mask_np is not None else None
    }
