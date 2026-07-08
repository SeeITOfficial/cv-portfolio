# Week 1 - Day 2
# Learned: grayscale removes the 3rd dimension — shape goes from (H, W, 3) to (H, W)
# Each pixel in grayscale is one number 0-255, not three
# cmap='gray' required in matplotlib to display grayscale correctly

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("masala.png")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

print("Original shape:", image.shape)
print("Gray shape:", gray.shape)
print("Max pixel: ", np.max(image))
print("Min pixel: ", np.min(image))
print("Data type: ", image.dtype)
print("top left pixel: ", gray[0, 0])

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))   
axes[0].set_title("Original")
axes[1].imshow(gray, cmap='gray')
axes[1].set_title("Grayscale")  
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()