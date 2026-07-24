# Week 2 - Day 3
# Learned: images have 3 channels — split with [:,:,0] [:,:,1] [:,:,2] or cv.split()
# Each channel shows how much of that color exists at each pixel
# Red channel bright = high red content, dark = no red
# subplots(2,3) gives 2D axes grid — index with axes[row,col]
# Loop with axes.flatten() to iterate individual subplots
# Always cmap='gray' for single channel display

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('color_wheel.webp')

#blue, green, red = cv.split(image)

blue_img = image[:, :, 0]
green_img = image[:, :, 1]
red_img = image[:, :, 2]

fig, axes = plt.subplots(2, 3, figsize=(12,4))
axes[0,0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
axes[0,0].set_title("Original RGB")
axes[0,1].imshow(red_img, cmap='gray')
axes[0,1].set_title("Red")
axes[0,2].imshow(green_img, cmap='gray')
axes[0,2].set_title("Green")
axes[1,0].imshow(blue_img, cmap='gray')
axes[1,0].set_title("Blue")
axes[1,1].imshow(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cmap='gray')
axes[1,1].set_title("Original Gray")

for ax in axes.flatten():
    ax.axis('off')
plt.tight_layout()
plt.show()