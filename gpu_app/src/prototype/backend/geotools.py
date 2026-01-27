import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape, box
import cv2
import numpy as np
from config import logger


# Utility Functions for the class 

def save_dino_geotiff(image_path, boxes, output_path):
        """
        Zeichnet Bounding Boxes direkt in die Bilddaten und speichert sie als 
        georeferenziertes GeoTIFF.
        """
        with rasterio.open(image_path) as src:
            # Rasterio liest (Channels, Height, Width) -> OpenCV braucht (H, W, C)
            img = src.read().transpose(1, 2, 0) 
            meta = src.meta.copy()

            # Bild von RGB zu BGR für OpenCV konvertieren (falls nötig)
            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Zeichne jede Box direkt in das Pixel-Array
            for box in boxes:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                # Rote Box zeichnen (Farbe in BGR: 0, 0, 255)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Zurück zu RGB für den Export
            img_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(img_final)
        
        logger.info(f"DINO GeoTIFF with boxes saved in: {output_path}")


def save_sam_geotiff(data, reference_path, output_path):
        """
        Speichert ein Numpy-Array als GeoTIFF unter Verwendung der 
        Georeferenzierung eines Referenzbildes.
        """
        with rasterio.open(reference_path) as src:
            out_meta = src.meta.copy()
            # Update Metadaten für die Maske (1 Kanal, Byte-Format)
            out_meta.update({
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "nodata": 0
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(data.astype(np.uint8), 1)
        
        logger.info(f"GeoTIFF erfolgreich gespeichert: {output_path}")  



def save_dino_as_geojson(boxes, reference_tif_path, output_path):
    """
    Konvertiert DINO Bounding Boxes in ein georeferenziertes GeoJSON.
    """
    if boxes is None or len(boxes) == 0:
        return None

    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        crs = src.crs

    # Boxen von Tensors/Numpy in Shapely-Geometrien umwandeln
    geometries = []
    for b in boxes:
        # b ist [x1, y1, x2, y2]
        coords = b.cpu().numpy() if hasattr(b, "cpu") else b
        # Erstelle ein Polygon aus der Box und transformiere es in Weltkoordinaten
        geom = box(coords[0], coords[1], coords[2], coords[3])
        
        # Manuelle Transformation der Box-Ecken in Geo-Koordinaten
        # (Rasterio transform * Pixel-Koordinate = Geo-Koordinate)
        left, top = transform * (coords[0], coords[1])
        right, bottom = transform * (coords[2], coords[3])
        geometries.append(box(left, bottom, right, top))

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    # Wichtig für Leafmap: Immer nach EPSG:4326 (WGS84) konvertieren
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(output_path, driver='GeoJSON')
    return output_path

def save_sam_as_geojson(mask_array, reference_tif_path, output_path):
    """
    Vektorisiert die SAM-Rastermaske und speichert sie als GeoJSON.
    """
    if mask_array is None:
        return None

    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        crs = src.crs

    # rasterio.features.shapes extrahiert Polygone aus dem Raster
    # mask=(mask_array > 0) sorgt dafür, dass nur die "weißen" Flächen extrahiert werden
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            rasterio.features.shapes(mask_array, mask=(mask_array > 0), transform=transform)
        )
    )

    geoms = list(results)
    if not geoms:
        return None

    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
    # Konvertierung für Web-Anzeige
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(output_path, driver='GeoJSON')
    return output_path           