import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def compute_spectrum(gray_img):
    dft = np.fft.fft2(gray_img.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    return dft_shift, magnitude

images = {
    "NEXX Card": cv.imread("../images/nexx.jpeg"),
    "NFS Car": cv.imread("../images/nfs.jpg"),
}

images = {k: v for k, v in images.items() if v is not None}
if not images:
    raise FileNotFoundError("No images loaded")

fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
fig.suptitle("Fourier Transform Frequency Spectrum", fontsize=12)

if len(images) == 1:
    axes = [axes]

for row, (name, img_bgr) in enumerate(images.items()):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    dft_shift, magnitude = compute_spectrum(gray)

    # Column 0: original grayscale
    axes[row][0].imshow(gray, cmap='gray')
    axes[row][0].set_title(f'{name} — Original')
    axes[row][0].axis('off')

    # Column 1: frequency spectrum
    axes[row][1].imshow(magnitude, cmap='gray')
    axes[row][1].set_title(f'{name} — Frequency Spectrum')
    axes[row][1].axis('off')

    # Column 2: spectrum with colormap for easier reading
    axes[row][2].imshow(magnitude, cmap='hot')
    axes[row][2].set_title(f'{name} — Spectrum (hot colormap)')
    axes[row][2].axis('off')

    print(f"{name}: image shape {gray.shape}, "
          f"spectrum min={magnitude.min()}, max={magnitude.max()}")

plt.tight_layout()

def on_key(event):
    if event.key == 'escape':
        plt.close('all')
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()