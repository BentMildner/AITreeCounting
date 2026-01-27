import cv2
import os

# Das Bild ist 512x512 Pixel groß
img = cv2.imread("/storage/soltau/data/tiles_test_512/top_left.png")

# Quadranten in 256x256 schneiden
tiles = {
    "upper_left_256": img[0:256, 0:256],
    "upper_right_256": img[0:256, 256:512],
    "lower_left_256": img[256:512, 0:256],
    "lower_right_256": img[256:512, 256:512]
}

# Speichern zum Testen
output_path = "/storage/soltau/data/tiles_test_256/"
os.makedirs(output_path, exist_ok=True)

for name, tile in tiles.items():
    cv2.imwrite(f"{output_path}{name}.png", tile)

print("4 Test-Kacheln (256x256) erstellt.")
