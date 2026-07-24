import cv2 as cv
import numpy as np

bgr_img = cv.imread("../images/nexx.jpeg")

if bgr_img is None:
    raise FileNotFoundError("Image not found")

h, w = bgr_img.shape[:2]
center = (w // 2, h // 2)

def update(val):
    angle = cv.getTrackbarPos("Rotation", "Affine Transformation") - 180
    scale = cv.getTrackbarPos("Scale", "Affine Transformation") / 100.0
    tx = cv.getTrackbarPos("Translate X", "Affine Transformation") - 300
    ty = cv.getTrackbarPos("Translate Y", "Affine Transformation") - 300
    shear_val = cv.getTrackbarPos("Shear", "Affine Transformation") / 100.0 - 0.5

    #guard
    if scale <= 0:
        scale = 0.01

    Rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv.warpAffine(bgr_img, Rotation_matrix, (w, h), borderMode=cv.BORDER_REFLECT)

    Translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv.warpAffine(bgr_img, Translation_matrix, (w, h), borderMode=cv.BORDER_REFLECT)

    Shear_matrix = np.float32([[1, shear_val, 0], [shear_val, 1, 0]])
    sheared_img = cv.warpAffine(bgr_img, Shear_matrix, (w, h), borderMode=cv.BORDER_REFLECT)

    M_rot_3x3 = np.vstack([Rotation_matrix, [0, 0, 1]])

    M_shear_3x3 = np.float32([[1, shear_val, 0],
                            [shear_val, 1, 0],
                            [0, 0, 1]])

    M_trans_3x3 = np.float32([[1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]])

    M_combined = (M_trans_3x3 @ M_rot_3x3 @ M_shear_3x3)[:2] # The @ operator is matrix multiplication in numpy. The order matters — A @ B means "apply B first, then A." So here: shear happens first, then rotation is applied on top of that.
    combined_img = cv.warpAffine(bgr_img, M_combined, (w, h), borderMode=cv.BORDER_REFLECT)

    small = (w // 2, h // 2)

    top    = np.hstack([cv.resize(rotated_img,     small),
                        cv.resize(translated_img,  small)])
    bottom = np.hstack([cv.resize(sheared_img,     small),
                        cv.resize(combined_img,    small)])
    grid   = np.vstack([top, bottom])

    title = f'Rotation:{angle}° Scale:{scale:.2f} TX:{tx} TY:{ty} Shear:{shear_val:.2f}'
    cv.setWindowTitle('Affine Transformation', title)
    cv.imshow('Affine Transformation', grid)


cv.namedWindow('Affine Transformation', cv.WINDOW_NORMAL)
# Sliders centered at neutral position
cv.createTrackbar('Rotation',    'Affine Transformation', 180, 360, update)
cv.createTrackbar('Scale',       'Affine Transformation', 100, 200, update)
cv.createTrackbar('Translate X', 'Affine Transformation', 300, 600, update)
cv.createTrackbar('Translate Y', 'Affine Transformation', 300, 600, update)
cv.createTrackbar('Shear',       'Affine Transformation', 50,  100, update)
update(0)

print("Top-left: rotation+scale | Top-right: translation")
print("Bottom-left: shear | Bottom-right: rotation+translation combined")

while True:
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()