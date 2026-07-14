import cv2 as cv
import numpy as np

bgr_img = cv.imread("../images/nfs.jpg")
if bgr_img is None:
    raise FileNotFoundError("Image not found. Please check the path.")

'''Convert to float32 so that subsequent frequency-domain or filter operations
(e.g. Fourier transforms, convolution with kernels, scaling, or subtraction)
don't suffer from integer overflow/underflow and retain precision for
negative values and fractional results.'''

gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY).astype(np.float32)
h, w = gray_img.shape

dft = np.fft.fft2(gray_img)
dft_shift = np.fft.fftshift(dft)

#center coordinates for mask
cx, cy = w//2, h//2

def make_masks(radius):
    #build coordinate grid; distance from center for every pixel
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

    #hard mask
    low_pass_hard = (dist_from_center <= radius).astype(np.float32)
    high_pass_hard = (dist_from_center > radius).astype(np.float32)

    #soft mask
    low_pass_soft = np.exp(-(dist_from_center**2) / (2 * (radius/2)**2 + 1e-6))
    high_pass_soft = 1 - low_pass_soft

    return low_pass_hard, high_pass_hard, low_pass_soft, high_pass_soft

def apply_mask(mask):
    #apply mask to DFT
    dft_masked = dft_shift * mask

    #inverse DFT
    dft_ishift = np.fft.ifftshift(dft_masked)
    img_back = np.abs(np.fft.ifft2(dft_ishift))

    #Normalize the result to the range [0, 255] and convert to uint8
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    return img_back


def update(val):
    radius = max(1, cv.getTrackbarPos('Radius', 'Frequency Domain Filters'))

    low_pass_hard, high_pass_hard, low_pass_soft, high_pass_soft = make_masks(radius)

    low_pass_result_hard = apply_mask(low_pass_hard)
    high_pass_result_hard = apply_mask(high_pass_hard)
    low_pass_result_soft = apply_mask(low_pass_soft)
    high_pass_result_soft = apply_mask(high_pass_soft)

   
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    small = (w//2, h//2)

    # resize panels
    tl = cv.resize(low_pass_result_hard, small)
    tr = cv.resize(high_pass_result_hard, small)
    bl = cv.resize(low_pass_result_soft, small)
    br = cv.resize(high_pass_result_soft, small)

    # add labels to each panel
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(tl, 'Hard Low-pass', (10, 20), font, 0.6, (255), 1, cv.LINE_AA)
    cv.putText(tr, 'Hard High-pass', (10, 20), font, 0.6, (255), 1, cv.LINE_AA)
    cv.putText(bl, 'Soft Low-pass', (10, 20), font, 0.6, (255), 1, cv.LINE_AA)
    cv.putText(br, 'Soft High-pass', (10, 20), font, 0.6, (255), 1, cv.LINE_AA)

    top = np.hstack([tl, tr])
    bottom = np.hstack([bl, br])
    grid = np.vstack([top, bottom])

    title = f'Radius: {radius}px'
    cv.setWindowTitle('Frequency Domain Filters', title)
    cv.imshow('Frequency Domain Filters', grid)


cv.namedWindow('Frequency Domain Filters', cv.WINDOW_NORMAL)
cv.createTrackbar('Radius', 'Frequency Domain Filters', 30, 200, update)
update(0)

print("Top-left: original | Top-right: spectrum + mask overlay")
print("Bottom-left: soft low-pass | Bottom-right: soft high-pass")
print("ESC to close")

while True:
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()