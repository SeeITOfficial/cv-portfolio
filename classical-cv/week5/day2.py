# ============================================================
# Week 5, Day 2 — CLAHE vs Global Histogram Equalization
# What I learned:
#   - equalizeHist only accepts single-channel uint8 images
#   - CLAHE is applied per-tile (tileGridSize), then stitched
#   - clipLimit controls how aggressively contrast is boosted
#     — higher = more contrast but more noise amplification
#   - For COLOR images, convert to LAB, equalize only the L
#     channel (luminance), convert back — this preserves hue
#   - Global equalization flattens the histogram across the
#     whole image — good for uniform scenes, bad for mixed ones
# ============================================================

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# --- Load image ---
img_bgr = cv.imread('sfight.jpg')

if img_bgr is None:
    raise FileNotFoundError("Image not found. Check path.")

# --- Convert to grayscale for equalization ---
# equalizeHist and CLAHE both require single-channel input
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# ── METHOD 1: Global Histogram Equalization ──────────────────
# Redistributes ALL pixel intensities across the full 0-255 range
# in one shot. Fast, simple, blunt.
eq_global = cv.equalizeHist(gray)

# ── METHOD 2: CLAHE ──────────────────────────────────────────
# clipLimit: histogram bins above this value get clipped and
#            redistributed. Higher = stronger contrast boost.
#            2.0 is the standard starting point.
# tileGridSize: the image is divided into these many tiles.
#               (8,8) = 64 tiles. Each tile equalized independently.
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eq_clahe = clahe.apply(gray)

# ── METHOD 3 (BONUS): CLAHE on color image via LAB ───────────
# LAB separates Luminance (L) from color (A=green-red, B=blue-yellow)
# We equalize only L so color/hue is unchanged
lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)
l_clahe = clahe.apply(l)                    # apply CLAHE to L only
lab_clahe = cv.merge([l_clahe, a, b])       # put it back together
color_result = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)

# --- Compute histograms for the grayscale versions ---
def get_hist(image):
    # Returns flattened (256,) histogram array
    return cv.calcHist([image], [0], None, [256], [0, 256]).flatten()

hist_orig  = get_hist(gray)
hist_global = get_hist(eq_global)
hist_clahe  = get_hist(eq_clahe)

# --- Plot: 2 rows × 3 cols ---
# Row 1: images (gray original, global eq, CLAHE)
# Row 2: their histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Histogram Equalization: Global vs CLAHE', fontsize=14)

images_row = [
    (gray,       'Original (Grayscale)'),
    (eq_global,  'Global equalizeHist'),
    (eq_clahe,   'CLAHE (clip=2.0, 8x8)'),
]

hists_row = [
    (hist_orig,   'Original Histogram',       'gray'),
    (hist_global, 'After Global Equalization', 'purple'),
    (hist_clahe,  'After CLAHE',               'teal'),
]

for ax, (img, title) in zip(axes[0], images_row):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

for ax, (hist, title, color) in zip(axes[1], hists_row):
    ax.plot(hist, color=color)
    ax.set_title(title)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Pixel Count')
    ax.set_xlim([0, 256])

plt.tight_layout()

# --- Also show the color LAB result separately ---
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('CLAHE on Color Image (via LAB)', fontsize=14)

# Convert BGR→RGB for matplotlib display
axes2[0].imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
axes2[0].set_title('Original Color')
axes2[0].axis('off')

axes2[1].imshow(cv.cvtColor(color_result, cv.COLOR_BGR2RGB))
axes2[1].set_title('CLAHE on L channel (LAB)')
axes2[1].axis('off')

plt.tight_layout()

def on_key(event):
    if event.key == 'escape':
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)
fig2.canvas.mpl_connect('key_press_event', on_key)
plt.show()