import os 
import torch
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
from config import * 
from geotools import save_dino_geotiff, save_sam_geotiff
from samgeo.text_sam import LangSAM




class TreeSegmentationService:

    def __init__(self, gpu_id, model_type):
        self.device_id, self.free_mem = self.prepare_environment(gpu_id)
        self.model = self.load_models(model_type)

    # Prepares the environment by selecting the GPU and checking memory
    def prepare_environment(self, gpu_id):
        # Selects the free GPU with the Id 1 --> turns it to only usable GPU with new index 0
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        # Check if CUDA is available before loading the model
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ERROR: CUDA is not available. LangSAM with SAM 2 Large requires a GPU with sufficient VRAM."
            )


        current_device = torch.cuda.current_device()
        free_mem, total_mem = torch.cuda.mem_get_info(current_device)

        # model needs sufficient VRAM
        if free_mem < 4000: 
            logger.warning("WARNING: application might crash due to insufficient VRAM.")

        return current_device, free_mem


    # Initializes LangSAM class and automatically builds SAM and GroundingDino with the official weights --> foundation models
    def load_models(self, model_type):
        model = LangSAM(model_type=model_type)
        return model

    # Runs prediction with GroundingDINO and applies filtering
    def run_prediction_dino(self, image_path, text_prompt, box_threshold, text_threshold, output_dir):
        logger.info(f"DINO Prediction started with prompt: {text_prompt}")

        # Load image
        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img)

        # DINO prediction
        boxes, logits, phrases = self.model.predict_dino(
            image=pil_img,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )


        if len(boxes) == 0:
            logger.warning("No boxes found.")
            return boxes, logits, phrases, pil_img

        # temporary lists for filtered boxes
        filtered_indices = []

        # Size of boxes to filter out (in pixels)
        MAX_BOX_SIZE = 250

        # Filter Function, iterates over all box outputs of dino and applies size and shadow filter
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.cpu().numpy() # takes upper left and lower right corner of gpu tensor box
            w, h = x2 - x1, y2 - y1 

            # Size Filter, everything larger than MAX_BOX_SIZE will be ignored
            if w > MAX_BOX_SIZE or h > MAX_BOX_SIZE:
                continue

            # Shadow/Asphalt Filter using Excess Green Index (ExG)
            crop = img_np[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
            if crop.size > 0:
                r, g, b = crop[:,:,0], crop[:,:,1], crop[:,:,2]
                # Excess Green: 2*G - R - B
                exg = 2.0 * g.astype(float) - r.astype(float) - b.astype(float)

                # if the average ExG is below a threshold, it's likely a shadow or asphalt
                if np.mean(exg) < 20: 
                    continue

            filtered_indices.append(i)

        # Recreate boxes
        if filtered_indices:
            boxes = boxes[filtered_indices]
            logits = logits[filtered_indices]
            phrases = [phrases[i] for i in filtered_indices]
            
            # Non-Maximum Suppression (NMS)
            # iou_threshold 0.5 bedeutet: Wenn zwei Boxen zu 50% überlappen, bleibt nur die bessere
            if logits.dim() > 1:
                scores = logits.max(dim=1).values
            else:
                scores = logits

            # NMS ausführen
            nms_idx = ops.nms(boxes, scores, iou_threshold=0.20)
            
            boxes = boxes[nms_idx]
            logits = logits[nms_idx]
            phrases = [phrases[i] for i in nms_idx]
        else:
            logger.warning("All boxes were filtered out.")
            return torch.tensor([]), torch.tensor([]), [], pil_img


    # Save results
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Save as png with annotated boxes
        save_path_png = os.path.join(output_dir, f"{base_name}_dino_filtered.png")
        
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(pil_img)
        for box, phrase, logit in zip(boxes, phrases, logits):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{phrase} ({logit:.2f})", color='white', backgroundcolor='red', fontsize=8)
        
        plt.axis('off')
        plt.savefig(save_path_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        logger.info(f"DINO result saved in: {save_path_png}")

        # Save as GeoTIFF via helper function
        save_path_tif = os.path.join(output_dir, f"{base_name}_dino_georef.tif")
        save_dino_geotiff(image_path, boxes, save_path_tif)

        return boxes, logits, phrases, pil_img

    # Runs prediction with SAM and refines masks
    def run_prediction_sam(self, pil_img, boxes, output_dir, image_path):
        if boxes is None or len(boxes) == 0:
            logger.warning("No boxes provided for SAM prediction.")
            return None

        logger.info(f"SAM starts segmentation for {len(boxes)} objects...")

        # SAM prediction
        masks = self.model.predict_sam(image=pil_img, boxes=boxes)

        # convert into numpy for further processing
        masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else masks
        if masks_np.ndim == 4:
            masks_np = np.squeeze(masks_np, axis=1)

        # Combine all masks into one, value 255 for mask, 0 for background
        combined_mask = np.any(masks_np, axis=0).astype(np.uint8) * 255
        
        # improves mask quality with another ExG filter 
        img_np = np.array(pil_img).astype(float)
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        exg = 2.0 * g - r - b
        green_mask = (exg > 20).astype(np.uint8) * 255
        refined_mask = cv2.bitwise_and(combined_mask, green_mask)

        # fills holes in the mask using contours
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(refined_mask)
        for cnt in contours:
            cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # eliminates small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
        final_mask = np.zeros_like(filled_mask)
        MIN_AREA = 150 
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
                final_mask[labels == i] = 255

        # save results
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]

        # save as png
        mask_save_path = os.path.join(output_dir, f"{base_name}_mask_final.png")
        
        Image.fromarray(final_mask).save(mask_save_path)
        logger.info(f"Optimierte Maske gespeichert: {mask_save_path}")

        # save as GeoTIFF via helper function
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]
        geotiff_save_path = os.path.join(output_dir, f"{base_name}_mask_final.tif")
        save_sam_geotiff(final_mask, image_path, geotiff_save_path)

        return masks, final_mask
      

def main():
    try:
        service = TreeSegmentationService(GPU_ID, MODEL_TYPE)

        logger.info(f"Environment prepared on GPU {GPU_ID} with {service.free_mem/1024**2:.0f} MiB free.")
        logger.info(f"LangSAM with {MODEL_TYPE} initialized.")

        # Check if Paths exists
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"Could not find input directory: {INPUT_PATH}")

        # Run Prediction with GroundingDINO 
        boxes, logits, phrases, pil_img = service.run_prediction_dino(
            image_path=INPUT_PATH,
            text_prompt=PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            output_dir=DINO_OUTPUT_DIR
        )

        # Run Prediction with SAM
        if boxes is not None and len(boxes) > 0:
            masks = service.run_prediction_sam(
                pil_img=pil_img,
                boxes=boxes,
                output_dir=SAM_OUTPUT_DIR,
                image_path=INPUT_PATH
            )
        
        logger.info(f"DINO found {len(boxes)} objects.")

    except Exception as e:
        logger.error(f"Abbruch durch Fehler: {e}")


if __name__ == "__main__":
    main()        