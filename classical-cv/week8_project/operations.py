import cv2 as cv
import numpy as np

def to_grayscale(bgr_img):
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    grayscale_3channel = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
    
    return grayscale_3channel


def apply_blur(bgr_img, strength):
    strength = max(1, min(strength, 25))
    if strength % 2 == 0:
        strength += 1

    blurred_image = cv.GaussianBlur(bgr_img, (strength, strength), sigmaX=0)

    return blurred_image




def apply_canny(bgr_img, thresh1, thresh2):
    gray_image = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    thresh1 = max(0, thresh1)
    thresh2 = max(0, thresh2)
    thresh2 = max(thresh1, thresh2)
    canny_image = cv.Canny(gray_image, threshold1=thresh1, threshold2=thresh2)
    final_bgr_image = cv.cvtColor(canny_image, cv.COLOR_GRAY2BGR)

    return final_bgr_image


def apply_clahe(bgr_img, gridsize):
    lab_image = cv.cvtColor(bgr_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab_image)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    l_clahe = clahe.apply(l)
    merged_lab = cv.merge([l_clahe, a, b])
    final_bgr = cv.cvtColor(merged_lab, cv.COLOR_LAB2BGR)

    return final_bgr


def show_colorspaces(bgr_img):
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    lab_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2LAB)

    hsv_as_bgr = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
    lab_as_bgr = cv.cvtColor(lab_img, cv.COLOR_LAB2BGR)

    return hsv_as_bgr, lab_as_bgr


def plot_histogram(bgr_img):
    import matplotlib.pyplot as plt

    b, g, r = cv.split(bgr_img)
    b_hist = cv.calcHist([b], [0], None, [256], [0, 256])
    g_hist = cv.calcHist([g], [0], None, [256], [0, 256])
    r_hist = cv.calcHist([r], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channel_data = [
        (b_hist, 'Blue Channel', 'blue'),
        (g_hist, 'Green Channel', 'green'),
        (r_hist, 'Red Channel',   'red'),
    ]

    for ax, (hist, title, color) in zip(axes, channel_data):
        ax.plot(hist.flatten(), color=color) 
        ax.set_title(title)
        ax.set_xlabel('Pixel Intensity (0–255)')
        ax.set_ylabel('Pixel Count')
        ax.set_xlim([0, 256])

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())
    buf = cv.cvtColor(buf, cv.COLOR_RGBA2RGB)

    plt.close(fig)

    return buf



def apply_sharpening(bgr_img, amount):
    amount = float(max(0.5, min(amount, 5.0)))

    blurred = cv.GaussianBlur(bgr_img, (5, 5), sigmaX=0)

    high_freq = bgr_img.astype(np.float32) - blurred.astype(np.float32)
    unsharp = bgr_img.astype(np.float32) + amount * high_freq
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

    return unsharp


def correct_perspective(bgr_img, points): 
    src = np.float32(points)

    TL, TR, BR, BL = src

    top_width = np.linalg.norm(TR - TL)
    bottom_width = np.linalg.norm(BR - BL)
    max_width = int(max(top_width, bottom_width))

    left_height = np.linalg.norm(BL - TL)
    right_height = np.linalg.norm(BR - TR)
    max_height = int(max(left_height, right_height))

    dest = np.float32([[0, 0],
                       [max_width - 1, 0],
                       [max_width - 1, max_height - 1],
                       [0, max_height - 1]])
    
    Matrix = cv.getPerspectiveTransform(src, dest)
    warped = cv.warpPerspective(bgr_img, Matrix, (max_width, max_height))


    return warped


def detect_contours(bgr_img, thresh1, thresh2, kernel, cont_area):
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    kernel = max(1, min(kernel, 25))
    if kernel % 2 == 0:
        kernel += 1

    thresh1 = max(0, thresh1)
    thresh2 = max(0, thresh2)
    thresh2 = max(thresh1, thresh2)

    blurred_img = cv.GaussianBlur(gray_img, (kernel, kernel), sigmaX=0)
    canny_img = cv.Canny(blurred_img, threshold1=thresh1, threshold2=thresh2)

    contours, _ = cv.findContours(canny_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv.contourArea(contour) > cont_area]

    bgr_copy = bgr_img.copy()
    cv.drawContours(bgr_copy, filtered_contours, -1, (0, 255, 0), 2)

    return bgr_copy, len(filtered_contours)


def apply_backprojection(bgr_img, roi_bgr):
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    roi_hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)

    roi_h_hist = cv.calcHist([roi_hsv], [0], None, [180], [0, 180])
    cv.normalize(roi_h_hist, roi_h_hist, 0, 255, cv.NORM_MINMAX)

    backproj = cv.calcBackProject([hsv_img], [0], roi_h_hist, [0, 180], scale=1)
    backproj = cv.GaussianBlur(backproj, (11, 11), sigmaX=0)
    _, thresh_mask = cv.threshold(backproj, 50, 255, cv.THRESH_BINARY)
    masked_result = cv.bitwise_and(bgr_img, bgr_img, mask=thresh_mask)

    return masked_result