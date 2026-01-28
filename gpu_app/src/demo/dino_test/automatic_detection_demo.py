from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import torch

# --- KONFIGURATION ---
config_file = "/storage/soltau/models/groundingdino/GroundingDINO_SwinB_cfg.py"
checkpoint = "/storage/soltau/models/groundingdino/weights/groundingdino_swinb_cogcoor.pth"
OUTPUT_DIR = "/storage/soltau/data/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(config_file, checkpoint)

# Inputs
image_paths = {
    "1024": "/storage/soltau/data/tiles_test_1024/tile_0_0.png",
    "512": "/storage/soltau/data/tiles_test_512/top_left.png",
    "256": "/storage/soltau/data/tiles_test_256/upper_left_256.png"
}

TEXT_PROMPTS = [
    "tree"
]

BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.0

for size_name, path in image_paths.items():

    if not os.path.exists(path):
        print(f"Datei nicht gefunden: {path}")
        continue

    print(f"\nTeste Bildgröße: {size_name}px")

    image_source, image = load_image(path)

    all_boxes = []
    all_logits = []
    all_phrases = []

    # --- MULTI-PROMPT ---
    for prompt in TEXT_PROMPTS:
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        if boxes.shape[0] > 0:
            all_boxes.append(boxes)
            all_logits.append(logits)
            all_phrases.extend(phrases)

    if len(all_boxes) == 0:
        print("Keine Boxen gefunden")
        continue

    boxes = torch.cat(all_boxes, dim=0)
    logits = torch.cat(all_logits, dim=0)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mask = areas < 0.10   

    boxes = boxes[mask]
    logits = logits[mask]

    print(f"Boxen nach Filter: {boxes.shape[0]}")

    # --- ANNOTATION ---
    annotated_frame = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=all_phrases
    )

    output_name = f"result_{size_name}px_small_objects.png"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    cv2.imwrite(output_path, annotated_frame)
    print(f"Gespeichert: {output_path}")

print("\nAlle Tests abgeschlossen.")
