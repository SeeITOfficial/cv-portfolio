import cv2 as cv
import numpy as np

bgr_img = cv.imread("../images/nexx.jpeg")

if bgr_img is None:
    raise FileNotFoundError("Image not found")

h, w = bgr_img.shape[:2]

points = []
bgr_copy = bgr_img.copy()

def click_handler(event, x, y, flags, param):
    global points, bgr_copy

    if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        cv.circle(bgr_copy, (x, y), 6, (0, 255, 0), -1)

        cv.imshow("Perspective Transform", bgr_copy)

        if len(points) == 4:
            run_transform()


def run_transform():
    global points

    src = np.float32(points)

    tl, tr, br, bl = src

    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    max_width = int(max(top_width, bottom_width))

    left_height = np.linalg.norm(bl - tl) 
    right_height = np.linalg.norm(br - tr)
    max_height = int(max(left_height, right_height))

    dest = np.float32([[0, 0],
                       [max_width - 1, 0],
                       [max_width - 1, max_height - 1],
                       [0, max_height - 1]]) #forms a perfect rectangle
    
    M = cv.getPerspectiveTransform(src, dest)
    warped = cv.warpPerspective(bgr_img, M, (max_width, max_height))

    pts_draw = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv.polylines(bgr_copy, [pts_draw], isClosed=True, color=(0, 0, 255), thickness=2)
    cv.imshow("Perspective Transform", bgr_copy)

    cv.imshow("Corrected Perspective", warped)
    print(f"Output size : {max_width} x {max_height}")
    print("press R to reset the points and select again. ESC to exit.")



def reset_points():
    global points, bgr_copy
    points = []
    bgr_copy = bgr_img.copy()
    cv.imshow("Perspective Transform", bgr_copy)

    try:
        cv.destroyWindow("Corrected Perspective")
    except:
        pass

cv.namedWindow("Perspective Transform", cv.WINDOW_NORMAL)
cv.setMouseCallback("Perspective Transform", click_handler)
cv.imshow("Perspective Transform", bgr_copy)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        reset_points()

cv.destroyAllWindows()
