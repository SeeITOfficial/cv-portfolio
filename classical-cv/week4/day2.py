# Week 4 - Day 2
# findContours returns all boundaries in a binary image
# Raw count is noisy — always filter by area
# cv.contourArea(c) gives pixel area of a contour
# Filter with list comprehension: [c for c in contours if cv.contourArea(c) > threshold]
# -1 in drawContours = draw all, specific index = draw one
# Always work on image.copy() to preserve original

import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("takes_two.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, otsu = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

contours, _ = cv.findContours(otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print(f"Number of contours: {len(contours)}")
areas = [cv.contourArea(c) for c in contours]
print(f"Largest contour area: {max(areas)}")
print(f"Smallest contour area: {min(areas)}")

# Keep only contours larger than 500 pixels
filtered = [c for c in contours if cv.contourArea(c) > 500]
cv.drawContours(image, filtered, -1, (0, 255, 0), 2)
for i, c in enumerate(filtered):
    M = cv.moments(c)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv.putText(
            image,
            str(i + 1),
            (cx, cy),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

print(f"Filtered contours: {len(filtered)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
axes[0].set_title("Contours")
axes[1].imshow(gray, cmap="gray")
axes[1].set_title("Grayscale")
axes[2].imshow(otsu, cmap="gray")
axes[2].set_title("Otsu's Thresholding")

for ax in axes.flatten():
    ax.axis("off")

plt.tight_layout()
plt.show()