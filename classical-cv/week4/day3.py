# Week 4 - Day 3
# boundingRect returns x, y, w, h — top-left corner + width + height
# rectangle needs top-left (x,y) and bottom-right (x+w, y+h)
# Combine contours + bounding boxes for object detection pipeline
# This is the manual version of what YOLO does automatically

import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("takes_two.jpg")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, otsu_thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

contours, _ = cv.findContours(otsu_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

filtered = [c for c in contours if cv.contourArea(c) > 500]
cv.drawContours(image, filtered, -1, (0, 255, 0), 2)

for each in filtered:
    x, y, w, h = cv.boundingRect(each)
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Filtered Contours with Bounding Boxes")
plt.axis("off")
plt.show()