import cv2 as cv
import numpy as np

bgr_img = cv.imread("../images/nfs.jpg")

if bgr_img is None:
    raise FileNotFoundError("Image not found")

gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

def update(val):
    blur_val = cv.getTrackbarPos("Blur", "Canny Edge Detection")
    low_threshold = cv.getTrackbarPos("Low Threshold", "Canny Edge Detection")
    high_threshold = cv.getTrackbarPos("High Threshold", "Canny Edge Detection")

    if blur_val % 2 == 0 or blur_val < 1:
        blur_val += 1

    gray_blur = cv.GaussianBlur(gray_img, (blur_val, blur_val), 0)

    sx = cv.Sobel(gray_blur, cv.CV_64F, 1, 0, ksize=3)
    sy = cv.Sobel(gray_blur, cv.CV_64F, 0, 1, ksize=3)
    sobel = cv.convertScaleAbs(cv.magnitude(sx, sy))

    canny = cv.Canny(gray_blur, low_threshold, high_threshold)

    overlay = bgr_img.copy()
    overlay[canny == 255] = [0, 255, 0]   # green edges where canny fired

    h, w = gray_img.shape
    small = (w // 2, h // 2)

    top    = np.hstack([cv.resize(cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR), small),
                        cv.resize(cv.cvtColor(sobel,     cv.COLOR_GRAY2BGR), small)])
    bottom = np.hstack([cv.resize(cv.cvtColor(canny,     cv.COLOR_GRAY2BGR), small),
                        cv.resize(overlay,                                    small)])
    grid   = np.vstack([top, bottom])

    cv.imshow('Canny Edge Detection', grid)

cv.namedWindow('Canny Edge Detection', cv.WINDOW_NORMAL)
cv.createTrackbar('Blur',        'Canny Edge Detection', 5,   20,  update)
cv.createTrackbar('Low Threshold', 'Canny Edge Detection', 50,  500, update)
cv.createTrackbar('High Threshold', 'Canny Edge Detection', 150, 500, update)
update(0)

print("Top-left: blurred gray | Top-right: Sobel magnitude")
print("Bottom-left: Canny edges | Bottom-right: edges overlaid on color")
print("ESC to close")

while True:
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
