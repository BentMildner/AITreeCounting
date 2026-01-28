import cv2
import os

img = cv2.imread("/storage/soltau/data/tiles_test_1024/tile_0_0.png")

tiles = {
    "top_left": img[0:512, 0:512],
    "top_right": img[0:512, 512:1024],
    "bottom_left": img[512:1024, 0:512],
    "bottom_right": img[512:1024, 512:1024]
}

output_path = "/storage/soltau/data/tiles_test_512/"
os.makedirs(output_path, exist_ok=True)

for name, tile in tiles.items():
    cv2.imwrite(f"{output_path}{name}.png", tile)

print("4 Test-Kacheln erstellt.")