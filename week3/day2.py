# Week 3 - Day 2
# Gaussian — blurs everything uniformly, fast, destroys edges
# Median — kills noise, preserves hard edges, cartoon effect
# Bilateral — blurs smooth areas, preserves edges, slower
# Bilateral used when edges matter — medical imaging, segmentation prep

import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("color_wheel.webp")

gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

gaussian_blur = cv.GaussianBlur(gray_img , (15, 15), 0)
median_blur = cv.medianBlur(gray_img, 15)
bilateral_blur = cv.bilateralFilter(gray_img, 9, 75, 75)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(gray_img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(gaussian_blur, cmap='gray')
axes[0, 1].set_title('Gaussian Blur')
axes[1, 0].imshow(median_blur, cmap='gray')
axes[1, 0].set_title('Median Blur')
axes[1, 1].imshow(bilateral_blur, cmap='gray')
axes[1, 1].set_title('Bilateral Blur')

for ax in axes.flatten():
    ax.axis('off')
plt.tight_layout()
plt.show()