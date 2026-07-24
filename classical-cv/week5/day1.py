# ============================================================
# Week 5, Day 1 — Per-Channel RGB Histograms
# What I learned:
#   - cv.calcHist requires the image wrapped in a list: [img]
#   - OpenCV loads images in BGR order, so channel index 0 = Blue,
#     1 = Green, 2 = Red (NOT the intuitive RGB order)
#   - calcHist returns a (256, 1) shaped array — matplotlib
#     needs it flattened to (256,) via .flatten()
#   - histSize=[256] means 256 bins, one per intensity value
#   - ranges=[0, 256] is exclusive on the right — covers 0..255
# ============================================================

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("sfight.jpg")

if img is None:
    raise FileNotFoundError("Could not load 'sfight.jpg'. Check the file path.")

b, g, r = cv.split(img)
hist_b = cv.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv.calcHist([r], [0], None, [256], [0, 256])


print("Blue pixel with highest count:", np.argmax(hist_b))
print("Green pixel with highest count:", np.argmax(hist_g))
print("Red pixel with highest count:", np.argmax(hist_r))

# Close on Escape key
def on_key(event):
    if event.key == 'escape':
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()