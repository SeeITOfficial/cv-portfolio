# Week 1 - Day 1
# Learned: images are numpy arrays, shape=(height, width, channels)
# uint8 means pixel values 0-255, saturated arithmetic
# cv.add and cv.subtract clamp at 0 and 255

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv.imread('masala.png')

# Print its shape
print("Image shape:", img.shape)

print("Data type: ", img.dtype)
print("Max pixel value: ", img.max())
print("Min pixel value: ", img.min())
print("First pixel (top left corner): ", img[0, 0])

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("My First Image")
plt.axis('off')
plt.show()


bright = cv.add(img, np.ones(img.shape, dtype='uint8') * 80)
print("Bright top-left pixel: ", bright[0, 0])
dark = cv.subtract(img, np.ones(img.shape, dtype='uint8') * 80 )
print("Dark top-left pixel: ", dark[0, 0])

fig, axes = plt.subplots(1, 3, figsize=(12,4))
axes[0].imshow(cv.cvtColor(dark, cv.COLOR_BGR2RGB))
axes[0].set_title("Darker")
axes[1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[1].set_title("Original")
axes[2].imshow(cv.cvtColor(bright, cv.COLOR_BGR2RGB))
axes[2].set_title("Brighter")

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()