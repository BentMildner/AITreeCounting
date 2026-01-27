from samgeo import SamGeo2
from samgeo.text_sam import LangSAM
import os


def run_sam2_tree_detection(image_path, output_path="/storage/soltau/data/results/tree_sam2_result.png"):

    print("Initialisiere LangSAM mit SAM 2 (sam2-hiera-large)...")
    
 
    model = LangSAM(model_type="sam2-hiera-large")


    text_prompt = "tree . tree canopy"


    box_threshold = 0.18
    text_threshold = 0.24

    print(f"Lade Bild: {image_path}")

    print(f"Starte Vorhersage mit Prompt '{text_prompt}'...")
    
    # 3. Vorhersage (Predict)
    masks, boxes, phrases, logits = model.predict(
        image=image_path,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        return_results=True  
    )

    # 4. Ergebnisse prüfen und speichern
    if boxes.nelement() == 0:
        print("Keine Bäume gefunden.")
    else:
        print(f"Erfolg! {len(boxes)} Bäume erkannt.")
        print(f"Verwendetes SAM-Modell: {model.sam.model_id if hasattr(model.sam, 'model_id') else 'SAM 2'}")

        # Visualisierung erstellen
        model.show_anns(
            output=output_path,
            title=f"Erkannte Bäume (sam2-hiera-large)",
            box_color="red",
            alpha=0.6,
            add_boxes=True
        )
        print(f"Visualisierung gespeichert unter: {output_path}")



if __name__ == "__main__":
    my_png_path = "/storage/soltau/data/tiles/tile_0_0.png"
    
    if os.path.exists(my_png_path):
        run_sam2_tree_detection(my_png_path)
    else:
        print(f"Fehler: Die Datei '{my_png_path}' wurde nicht gefunden.")