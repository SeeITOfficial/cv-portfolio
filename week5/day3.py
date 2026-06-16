# ============================================================
# Week 5, Day 3 — Histogram Back-Projection
# What I learned:
#   - Back-projection outputs a probability map, not a normal image
#   - We use HSV colorspace because Hue is lighting-invariant
#   - The model histogram comes from a SAMPLE CROP, not the full image
#   - cv.normalize scales the histogram to 0-255 range so the
#     back-projected values stay within uint8 bounds
#   - cv.calcBackProject takes [full_image_hsv], channel, model_hist,
#     ranges, scale — and returns the probability map
#   - A Gaussian blur on the result smooths out noise before
#     thresholding — standard post-processing step
# ============================================================

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# --- Load image ---
img_bgr = cv.imread('nexx.jpeg')

if img_bgr is None:
    raise FileNotFoundError("Image not found. Check path.")

# Convert full image to HSV — this is what we'll back-project onto
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

# ── STEP 1: Define a sample ROI (your color model source) ────
x, y, w, h = cv.selectROI("Select ROI for Color Model", img_bgr, fromCenter=False, showCrosshair=True)
cv.destroyWindow("Select ROI for Color Model")

if w == 0 or h == 0:
    raise ValueError("No ROI selected. Please select a valid region.")
roi_bgr = img_bgr[y:y+h, x:x+w]
roi_hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)

# ── STEP 2: Build the model histogram from the ROI ───────────
# Using only the Hue channel ([0]) — ignore S and V
# Hue range in OpenCV is 0-180 (not 0-360), so ranges=[0,180]
# 180 bins = one bin per hue degree
model_hist = cv.calcHist(
    [roi_hsv],   # source: the sample crop
    [0],         # channel 0 = Hue
    None,        # no mask
    [180],       # 180 bins for full hue range
    [0, 180]     # hue goes 0-180 in OpenCV
)

# Normalize to 0-255 so back-projected values are valid uint8
cv.normalize(model_hist, model_hist, 0, 255, cv.NORM_MINMAX)

# ── STEP 3: Back-project onto the full image ─────────────────
# cv.calcBackProject(images, channels, hist, ranges, scale)
# scale=1 means no additional scaling after lookup
prob_map = cv.calcBackProject(
    [img_hsv],   # full image in HSV
    [0],         # use Hue channel
    model_hist,  # our color model
    [0, 180],    # hue range
    1            # scale factor
)

# ── POST-PROCESSING ──────────────────────────────────────────
# Blur smooths out noise — standard step before thresholding
prob_map_blur = cv.GaussianBlur(prob_map, (11, 11), 0)

# Threshold: pixels above 50 probability = detected, rest = black
_, prob_map_thresh = cv.threshold(
    prob_map_blur, 50, 255, cv.THRESH_BINARY
)

# Create a masked version of the original to visualize detection
# The mask highlights only the detected regions in the original color
mask_3ch = cv.merge([prob_map_thresh, prob_map_thresh, prob_map_thresh])
detected = cv.bitwise_and(img_bgr, mask_3ch)

# ── PLOT ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Histogram Back-Projection — Pink/Magenta Detection', fontsize=14)

display = [
    # Row 1: the inputs and raw output
    (cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB),  'Original Image',       None),
    (cv.cvtColor(roi_bgr, cv.COLOR_BGR2RGB),  'Sample ROI (Model)',   None),
    (prob_map,                                 'Raw Probability Map',  'gray'),
    # Row 2: processed outputs
    (prob_map_blur,                            'Blurred Prob Map',     'gray'),
    (prob_map_thresh,                          'Thresholded Mask',     'gray'),
    (cv.cvtColor(detected, cv.COLOR_BGR2RGB),  'Detected Region',      None),
]

for ax, (img, title, cmap) in zip(axes.flatten(), display):
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()

def on_key(event):
    if event.key == 'escape':
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()