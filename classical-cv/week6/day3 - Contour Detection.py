import cv2 as cv
import numpy as np

bgr_img = cv.imread("../images/shapes.jpg")

if bgr_img is None:
    raise FileNotFoundError("Image not found")

gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

def update(val):
    blur_val = cv.getTrackbarPos("Blur", "Contour Detection")
    low_threshold = cv.getTrackbarPos("Low Threshold", "Contour Detection")
    high_threshold = cv.getTrackbarPos("High Threshold", "Contour Detection")
    min_area = cv.getTrackbarPos("Min Area", "Contour Detection") * 10
    accuracy_percentage = cv.getTrackbarPos("Shape Accuracy (%)", "Contour Detection") / 100.0

    if blur_val % 2 == 0 or blur_val < 1:
        blur_val += 1

    gray_blur = cv.GaussianBlur(gray_img, (blur_val, blur_val), 0)
    canny = cv.Canny(gray_blur, low_threshold, high_threshold)

    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

    overlay_all = bgr_img.copy()
    cv.drawContours(overlay_all, final_contours, -1, (128, 255, 128), 5) # all contours onn this overlay inn green

    overlay_bbox = bgr_img.copy()
    for cnt in final_contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(overlay_bbox, (x, y), (x + w, y + h), (255, 0, 0), 5) # bounding box for each contour in blue

    overlay_shape = bgr_img.copy()
    for cnt in final_contours:
        M = cv.moments(cnt)
        if M["m00"] != 0: #avoids division by zero for small indistinguishable contours whose shape is close to lines
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(overlay_shape, (cx, cy), 3, (0, 0, 255), -1) # centroid in red

        
        shape_accuracy = accuracy_percentage * cv.arcLength(cnt, True)
        shape_approx_predict = cv.approxPolyDP(cnt, shape_accuracy, True)

        vertices = len(shape_approx_predict)
        if vertices == 3:
            shape_name = "Triangle"
            color = (0, 0, 255)
        elif vertices == 4:
            shape_name = "Rectangle"
            color = (255, 0, 0)
        else:
            shape_name = f'Polygon ({vertices} sides)'
            color = (0, 255, 0)
        
        cv.drawContours(overlay_shape, [shape_approx_predict], -1, color, 5)

    h, w = gray_img.shape
    small = (w // 2, h // 2)

    top    = np.hstack([cv.resize(cv.cvtColor(canny, cv.COLOR_GRAY2BGR), small),
                        cv.resize(overlay_all,small)])
    bottom = np.hstack([cv.resize(overlay_bbox, small),
                        cv.resize(overlay_shape, small)])
    grid   = np.vstack([top, bottom])

    title = f"Contours Detected: {len(final_contours)} | Min Area: {min_area}"
    cv.setWindowTitle("Contour Detection", title)
    cv.imshow('Contour Detection', grid)

cv.namedWindow('Contour Detection', cv.WINDOW_NORMAL)
cv.createTrackbar('Blur', 'Contour Detection', 5, 20, update)
cv.createTrackbar('Low Threshold', 'Contour Detection', 50, 500, update)
cv.createTrackbar('High Threshold', 'Contour Detection', 150, 500, update)
cv.createTrackbar('Min Area', 'Contour Detection', 10, 5000, update)
cv.createTrackbar('Shape Accuracy (%)', 'Contour Detection', 5, 100, update)
update(0)

print("Top-left: Canny edges | Top-right: all contours in green")
print("Bottom-left: bounding boxes in blue | Bottom-right: shape approximation with centroids")

while True:
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()