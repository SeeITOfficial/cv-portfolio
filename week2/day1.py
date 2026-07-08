# Week 2 - Day 1
# Learned: uint8 = 0-255, float32 = 0.0-1.0
# Normalization = dividing by 255.0, required for neural networks later
# BGR to RGB just reverses channel order — B and R swap, G stays
# Equal channels mean swap makes no visible difference

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('masala.png')

print(image.dtype)

img_float = image.astype(np.float32) / 255.0
print(img_float.dtype)
print("Max pixel value: ", img_float.max())
print("Min pixel value: ", img_float.min())
print("First pixel (top left corner): ", img_float[0, 0])

# OpenCV loads BGR, let's prove it
print("BGR first pixel:", image[0, 0])

img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
print("RGB first pixel:", img_rgb[0, 0])