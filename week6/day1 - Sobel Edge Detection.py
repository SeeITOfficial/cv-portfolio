import cv2 as cv
import numpy as np

brg_img = cv.imread("../images/nfs.jpg")

if brg_img is None:
    raise FileNotFoundError("Image not found, check the path")

gray_img = cv.cvtColor(brg_img, cv.COLOR_BGR2GRAY)

def update(val):
    ksize = cv.getTrackbarPos('Kernel Size', 'Sobel Edge Detection')
    if ksize % 2 == 0 or ksize < 1:
        ksize += 1

    threshold = cv.getTrackbarPos('Threshold', 'Sobel Edge Detection')
    if threshold % 2 == 0 or threshold < 1:
        threshold += 1

    gray_blur = cv.GaussianBlur(gray_img, (threshold, threshold), 0)
    sobelx_raw = cv.Sobel(gray_blur, cv.CV_64F, 1, 0, ksize=ksize)
    sobely_raw = cv.Sobel(gray_blur, cv.CV_64F, 0, 1, ksize=ksize)
    sobelx     = cv.convertScaleAbs(sobelx_raw)   # uint8 for display
    sobely     = cv.convertScaleAbs(sobely_raw)   # uint8 for display
    magnitude  = cv.convertScaleAbs(cv.magnitude(sobelx_raw, sobely_raw))  # float → magnitude → uint8

    h, w = gray_blur.shape
    small = (w // 2, h // 2)

    top    = np.hstack([cv.resize(gray_blur,  small), cv.resize(sobelx,    small)])
    bottom = np.hstack([cv.resize(sobely,     small), cv.resize(magnitude, small)])
    grid   = np.vstack([top, bottom])

    cv.imshow('Sobel Edge Detection', grid)

cv.namedWindow('Sobel Edge Detection', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('Kernel Size', 'Sobel Edge Detection', 1, 21, update)
cv.createTrackbar('Threshold', 'Sobel Edge Detection', 1, 15, update)
update(3)

while True:
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()