# Week 1 - Day 3
# Learned: shape uses [] not (), height//2 for integer division
# Cropping is numpy slicing: image[row_start:row_end, col_start:col_end]
# cv.flip(img, 1) = horizontal, 0 = vertical, -1 = both
# ax vs plt — ax is one subplot, plt is the whole figure

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("masala.png")

height = image.shape[0]
width = image.shape[1]

cropped = image[(height//2):height, :]

flipped = cv.flip(image, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(cv.cvtColor(cropped, cv.COLOR_BGR2RGB))
axes[0].set_title("Cropped image")
axes[1].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
axes[1].set_title("Original image")
axes[2].imshow(cv.cvtColor(flipped, cv.COLOR_BGR2RGB))
axes[2].set_title("Flipped image")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()