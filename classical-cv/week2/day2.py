# Week 2 - Day 2
# Learned: OpenCV loads BGR, matplotlib expects RGB
# Forgetting to convert = red and blue swap visually, green unaffected
# cv.imshow works correctly with BGR (it's an OpenCV tool)
# plt.imshow needs RGB — always convert before displaying with matplotlib
# Rule: OpenCV=BGR, Matplotlib=RGB, convert when crossing between them

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

bgr_wheel = cv.imread("color_wheel.webp")
cv.imshow("BGR Color Wheel correct for cv", bgr_wheel)
cv.waitKey(0)   

rgb_wheel = cv.cvtColor(bgr_wheel, cv.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb_wheel)
axes[0].set_title("RGB Color Wheel correct for matplotlib")
axes[1].imshow(bgr_wheel)   
axes[1].set_title("BGR Color Wheel wrong for matplotlib")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()