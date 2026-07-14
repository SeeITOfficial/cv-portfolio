import cv2 as cv
import numpy as np

img = cv.imread("../images/nfs.jpg")
if img is None:
    raise FileNotFoundError("Image not found. Please check the path.")

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

identity = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.float32)

sharpen_4 = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]], dtype=np.float32)

sharpen_8 = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]], dtype=np.float32)

edge_detect = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]], dtype=np.float32)

def update(val):
    amount = cv.getTrackbarPos('Amount', 'Sharpening Kernels') / 10.0
    blur_kernel = cv.getTrackbarPos('Blur Kernel', 'Sharpening Kernels')  

    if blur_kernel % 2 == 0 or blur_kernel < 1:
        blur_kernel += 1

    identity_result = cv.filter2D(gray_img, -1, identity)
    sharpen_4_result = cv.filter2D(gray_img, -1, sharpen_4)
    sharpen_8_result = cv.filter2D(gray_img, -1, sharpen_8)
    edge_detect_result = cv.filter2D(gray_img, -1, edge_detect)

    #unsharp mask
    blurred = cv.GaussianBlur(gray_img, (blur_kernel, blur_kernel), 0)
    high_freq = gray_img.astype(np.float32) - blurred.astype(np.float32)
    unsharp = gray_img.astype(np.float32) + amount * high_freq
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

    h, w = gray_img.shape
    small = (w//2, h//2)

    tl = cv.resize(identity_result, small)
    tm = cv.resize(sharpen_4_result, small)
    tr = cv.resize(sharpen_8_result, small)
    bl = cv.resize(edge_detect_result, small)
    bm = cv.resize(unsharp, small)
    br = cv.resize(blurred, small)

    cv.putText(tl, "Identity", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
    cv.putText(tm, "Sharpen 4-neighbor", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
    cv.putText(tr, "Sharpen 8-neighbor", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
    cv.putText(bl, "Edge-only", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
    cv.putText(bm, "Unsharp Mask", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
    cv.putText(br, "Blurred", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)

    top = np.hstack([tl, tm, tr])
    bottom = np.hstack([bl, bm, br])
    grid = np.vstack([top, bottom])

    title = f"Unsharp Amount: {amount:.1f}, Blur Kernel: {blur_kernel}"
    cv.setWindowTitle("Sharpening Kernels", title)
    cv.imshow("Sharpening Kernels", grid)



cv.namedWindow('Sharpening Kernels', cv.WINDOW_NORMAL)
cv.createTrackbar('Amount', 'Sharpening Kernels', 10, 50, update)
cv.createTrackbar('Blur Kernel', 'Sharpening Kernels', 5,  21, update)
update(0)


while True:
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
