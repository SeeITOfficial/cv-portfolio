# Computer Vision Portfolio
Hello Reader, 
I'm a final-year Computer Science student from India building a production-ready computer vision portfolio. 
This repo documents everything from learning, building, debugging, and eventually shipping. 
Over a course of Six phases i will be building six deployed projects with one ultimate goal: become a CV engineer with valuable knowledge and skills to apply in various aspects of our real world.

I learnt from a series of tutorials, documentations and personal research. Every project here was built from scratch, deployed publicly, and represents something whose roots I actually understand.

## Stack
Python 3.11, OpenCV, NumPy, Matplotlib, Gradio, PyTorch (Phase 3+)


## Live Demo

`Phase 1 Project is a CV Image Processing Toolkit`
[https://cv-toolkit-cabi.onrender.com](https://cv-toolkit-cabi.onrender.com)
Upload any image. Select an operation. See what classical computer vision actually does to your pixels.


## Roadmap I followed

### Phase 1 -> Classical Computer Vision
*The foundation; math, pixels, and OpenCV.*


Week 1 Image basics -> loading, channels, pixel manipulation
Week 2 Color spaces -> BGR, HSV, LAB, grayscale
Week 3 Filtering -> Gaussian, median, bilateral blur 
Week 4 Morphology -> erosion, dilation, opening, closing
Week 5 Histograms and color analysis -> calcHist, CLAHE, back-projection 
Week 6 Geometric transforms -> affine, perspective, homography 
Week 7 Frequency domain -> Fourier transform, low/high-pass filters, sharpening 
Week 8 `Project 1 -> Image Processing Toolkit` 

### Phase 2 -> Feature Extraction
*Coming next. Keypoints, descriptors, matching.*

### Phase 3 -> Deep Learning for CV
*CNNs, transfer learning, custom training.*

### Phase 4 -> Object Detection
*YOLO, deployment pipelines, real-time detection.*

### Phase 5 -> Anomaly Detection
*Industrial inspection, unsupervised methods.*

### Phase 6 -> Capstone
*Full-stack CV system. End to end.*

---

## Project 1 -> CV Image Processing Toolkit

`Live:` [https://cv-toolkit-cabi.onrender.com](https://cv-toolkit-cabi.onrender.com)

A Gradio web app where you upload any image and apply classical CV operations interactively. Built entirely with OpenCV and deployed on Render.

### What it does


 Grayscale -> single-channel conversion 
 Gaussian Blur -> Low-pass spatial filtering with adjustable kernel 
 Canny Edge Detection -> Two-threshold edge detection 
 CLAHE -> Contrast-limited adaptive histogram equalization using LAB colorspace 
 Colorspaces -> HSV and LAB channel visualization 
 Histogram -> Per-channel BGR intensity distribution plot 
 Sharpening -> Unsharp masking with adjustable strength 
 Contour Detection -> Canny -> findContours -> filtered by area with live count 

### Why each operation matters

`CLAHE over global equalization` -> Global equalization flattens the entire histogram at once, destroying already-bright regions. CLAHE works on small tiles independently and clips the histogram before equalizing, so it boosts local contrast without blowing out highlights. For color images it operates on the L channel in LAB colorspace so hue and saturation are untouched.

`HSV over BGR for color analysis` -> In BGR, the same orange looks completely different under different lighting conditions. In HSV, the Hue channel captures color identity independently of brightness. That's why histogram back-projection and skin detection both work in HSV.

`Canny over raw Sobel` -> Sobel gives you a grayscale gradient map where every pixel has some edge score. Canny adds non-maximum suppression (thins edges to 1 pixel) and hysteresis thresholding (keeps weak edges only if connected to strong ones). The result is binary, thin, and connected -> exactly what contour detection needs downstream.

### Architecture

```
cv-toolkit/
├── app.py           <- Gradio interface, input/output wiring
├── operations.py    <- All CV logic, one function per operation
└── requirements.txt
```

CV logic is completely separated from the UI. Every function in `operations.py` takes a BGR numpy array and returns a BGR numpy array. Gradio handles the RGB conversion at the boundary.

### Screenshots

![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/CLAHE.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/Canny Edge Detection.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/Contour Detection.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/Gaussian Blur.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/Grayscale.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/Histogram Equalization.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/Histogram.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/HSV and LAB colorspaces.png>)
![alt text](<classical-cv/week8_project/assets/classical-cv screenshots/Sharpening.png>)

## Repository Structure

```
cv-portfolio/
├── README.md
├── assets/
│   └── screenshots/
├── classical-cv/
│   ├── week1/
│   ├── week2/
│   ├── week3/
│   ├── week4/
│   ├── week5/
│   ├── week6/
│   ├── week7/
│   └── week8_project/
│       ├── app.py
│       ├── operations.py
│       └── requirements.txt
└── .gitignore
```


## Running Locally

```bash
git clone https://github.com/SeeITOfficial/cv-portfolio.git
cd cv-portfolio/classical-cv/week8_project
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

`Note:` Free Render instances spin down after inactivity. First load after a period of inactivity may take 30–60 seconds. Refresh if it doesn't load immediately.


## About Me

Final-year CS student. Targeting Computer Vision engineering roles and MSc programs.


GitHub: [SeeITOfficial](https://github.com/SeeITOfficial)