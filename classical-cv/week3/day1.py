import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("color_wheel.webp")

hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
print("HSV image shape: ", hsv_img.shape)
print("HSV image data type: ", hsv_img.dtype)

H_img = hsv_img[:, :, 0]
S_img = hsv_img[:, :, 1]
V_img = hsv_img[:, :, 2]

fig, axes = plt.subplots(2, 2, figsize= (12,6))
axes[0, 0].imshow(H_img, cmap='gray')
axes[0, 0].set_title("H Image")
axes[0, 1].imshow(S_img, cmap='gray')
axes[0, 1].set_title("S Image")
axes[1, 0].imshow(V_img, cmap='gray')
axes[1, 0].set_title("V Image")
axes[1, 1].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
axes[1, 1].set_title("HSV Image")

for ax in axes.flatten():
    ax.axis('off')
plt.tight_layout()
plt.show()