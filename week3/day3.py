# Week 3 - Day 3
# Binary — manual global threshold, simple but inaccurate if lighting varies
# Otsu — auto finds best global threshold, better than guessing
# Adaptive — different threshold per region, best for uneven lighting
# ret value from cv.threshold = the threshold value used
# adaptiveThreshold returns only one value, no ret

import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('color_wheel.webp')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#thresholding
_, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)  
_, otsu = cv.threshold(gray, 127, 255, cv.THRESH_OTSU)
adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('Grayscale')
axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title('Binary Thresholding')
axes[1, 0].imshow(otsu, cmap='gray')
axes[1, 0].set_title("Otsu's Thresholding")
axes[1, 1].imshow(adaptive, cmap='gray')
axes[1, 1].set_title('Adaptive Thresholding')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()