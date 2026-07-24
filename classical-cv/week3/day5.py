# Week 3 - Day 5
# cv.rectangle, cv.circle, cv.putText draw directly onto the image array
# Colors passed as BGR tuples — (0,0,255) = red, (255,0,0) = blue
# Drawings are baked into the array before display
# Always convert to RGB before matplotlib display

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("color_wheel.webp")

cv.rectangle(image, (150, 100), (400, 300), (0, 255, 0), 2)
cv.circle(image, (300, 200), 50, (255, 0, 0), 2)
cv.putText(image, 'Color Wheel', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
print("pixel at circle center:", image[200, 300])

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()