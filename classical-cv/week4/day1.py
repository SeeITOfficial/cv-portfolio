# Week 4 - Day 1
# Canny takes two thresholds — lower and upper
# High thresholds = fewer edges (only strong ones)
# Low thresholds = more edges (weak ones too, plus noise)
# Tune thresholds based on image — no universal perfect value

import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("color_wheel.webp")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

canny1 = cv.Canny(gray, 50 , 150)
canny2 = cv.Canny(gray, 100 , 200)
canny3 = cv.Canny(gray, 30 , 100)

fig, axis = plt.subplots(3, 2, figsize=(10, 10))
axis[0, 0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
axis[0, 0].set_title("Original Image")
axis[0, 1].imshow(gray, cmap="gray")
axis[0, 1].set_title("Grayscale Image")
axis[1, 0].imshow(canny1, cmap="gray")
axis[1, 0].set_title("Canny Edges (50, 150)")
axis[1, 1].imshow(canny2, cmap="gray")  
axis[1, 1].set_title("Canny Edges (100, 200)")
axis[2, 0].imshow(canny3, cmap="gray")
axis[2, 0].set_title("Canny Edges (30, 100)")

for ax in axis.flatten():
    ax.axis("off")

plt.tight_layout()  
plt.show()