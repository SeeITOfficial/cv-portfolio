# Week 3 - Day 4
# Erosion — white shrinks, black grows, removes thin white features
# Dilation — white grows, black shrinks, thickens white features  
# Opening = erode then dilate — removes small white noise
# Closing = dilate then erode — fills small black holes
# kernel size controls how aggressive the operation is

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("color_wheel.webp")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, otsu_binary = cv.threshold(gray, 127, 255, cv.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)

eroded = cv.erode(otsu_binary, kernel, iterations=1)    
dilated = cv.dilate(otsu_binary, kernel, iterations=1)
morph_open = cv.morphologyEx(otsu_binary, cv.MORPH_OPEN, kernel)
morph_close = cv.morphologyEx(otsu_binary, cv.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes[0, 0].imshow(gray, cmap='gray')    
axes[0, 0].set_title('Grayscale')
axes[0, 1].imshow(otsu_binary, cmap='gray') 
axes[0, 1].set_title("Otsu's Binary")
axes[1, 0].imshow(eroded, cmap='gray')
axes[1, 0].set_title('Erosion')
axes[1, 1].imshow(dilated, cmap='gray')
axes[1, 1].set_title('Dilation')
axes[2, 0].imshow(morph_open, cmap='gray')
axes[2, 0].set_title('Morphological Opening')
axes[2, 1].imshow(morph_close, cmap='gray')
axes[2, 1].set_title('Morphological Closing')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()