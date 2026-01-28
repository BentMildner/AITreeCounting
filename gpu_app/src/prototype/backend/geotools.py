# File for utility functions to handle cpu tasks related to geospatial data

import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape, box
import cv2
import numpy as np
from config import logger


# Utility Functions for the class 

# draw boxes on the image and save as GeoTIFF
def save_dino_geotiff(image_path, boxes, output_path):
        with rasterio.open(image_path) as src:
            # Rasterio opens images 
            img = src.read().transpose(1, 2, 0) 
            meta = src.meta.copy()

            # Convert to BGR for OpenCV
            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # draws boxes into pixel arrays
            for box in boxes:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # back to RGB for export
            img_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(img_final)
        
        logger.info(f"DINO GeoTIFF with boxes saved in: {output_path}")


 # Saves a binary mask as a GeoTIFF using the reference raster for georeferencing
def save_sam_geotiff(data, reference_path, output_path):
        with rasterio.open(reference_path) as src:
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "nodata": 0
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(data.astype(np.uint8), 1)

        logger.info(f"GeoTIFF successfully saved: {output_path}")  



# transforms DINO boxes into pixel coordinates to GeoJSON
def save_dino_as_geojson(boxes, reference_tif_path, output_path):
    if boxes is None or len(boxes) == 0:
        return None

    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        crs = src.crs

    # convert boxes to polygons in world coordinates
    geometries = []
    for b in boxes:
        # b is [x1, y1, x2, y2]
        coords = b.cpu().numpy() if hasattr(b, "cpu") else b
        geom = box(coords[0], coords[1], coords[2], coords[3])
        
        # Transformation in world coordinates
        left, top = transform * (coords[0], coords[1])
        right, bottom = transform * (coords[2], coords[3])
        geometries.append(box(left, bottom, right, top))

    # create GeoDataFrame and save as GeoJSON
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(output_path, driver='GeoJSON')
    return output_path

# turns SAM mask array into GeoJSON polygons
def save_sam_as_geojson(mask_array, reference_tif_path, output_path):
    if mask_array is None:
        return None

    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        crs = src.crs

    # converts mask to shapes, each polygon with raster value
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            rasterio.features.shapes(mask_array, mask=(mask_array > 0), transform=transform)
        )
    )

    geoms = list(results)
    if not geoms:
        return None

    # create GeoDataFrame and save as GeoJSON
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(output_path, driver='GeoJSON')
    return output_path           