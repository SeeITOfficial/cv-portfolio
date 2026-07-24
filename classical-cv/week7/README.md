## Geometric Transformations
CV Systems suffer one fundamental problem; the camera is never perfectly aligned woth the subject. Documents photographed from an angle, cars are photographged sideways and through this, important details are missed or certain views are misinterpreted.

## Day 1 --> Afffine Transforms
An `affine transform` answers: "how do I move every pixel in this image to a new position according to a mathematical rule?" This rule ensures that parallel lines remain parallel.
The rule is a 2×3 matrix multiplied by every pixel's (x, y) coordinate:
        [x']   [a  b  tx] [x]
        [y'] = [c  d  ty] [y]
                        [1]
Where:
- `tx`, `ty` = translation (how far to move in x and y)
- `a`, `d` = scale (how much to stretch/shrink)
- `b`, `c` = rotation and shear components  

`Three main operations`
- Rotation
- Translation; how far to move in x and y
- Shear; Slants the image along one axis

When you rotate or translate, some pixels move outside the image boundary and new empty space appears. `warpAffine` fills that space with black by default. You can change this with `borderMode=cv.BORDER_REFLECT` to mirror edge pixels instead


## Day 2 --> Perspective Transformation and Document Scanner
`Perspective Distortion` is when you photograph a document at an angle and the edges end up appearing to converge in or out of the photo. Thus a rectangle looks like a trapezoid. Affine Tramsforms cant fix that because it cant map a trapezoid back to a rectangle; `Perspective Transform` can. `Why?` - it uses a `3x3` matrix that adds an extra ability to deal with parralel lines that converge at vanishing points.The nine values in the matrix form the `homography`.

We give OpenCV four point pairs:
`4 points in the source image` — the corners of your document as they appear in the photo (a trapezoid)
`4 points in the destination` — where those corners should end up (a perfect rectangle)

- `pythonM = cv.getPerspectiveTransform(src_points, dst_points)`
- `result = cv.warpPerspective(image, M, (output_width, output_height))`
OpenCV solves the `3×3 matrix` from those 8 constraints (4 points × 2 coordinates each). We never touch the matrix math directly but we use it in the cv methods.

## Day 3 --> Homography
Perspective tramsform manually allows us to click through the points, `Homograpy` computes them automatically from thousands of pairs of matched feature points between two images using the same `3x3 matrix`.
- `Why many points instead of 4?` Because real matched points have noise so some matches are wrong (outliers). If we use exactly 4 bad matches, the homography is garbage. 
- `cv.findHomography` uses `RANSAC (Random Sample Consensus)` It tries thousands of random subsets of 4 points, finds the matrix that agrees with the most matches, and discards the outliers automatically. More input points = more robust result.

`RANSAC` asks: `given all these noisy point matches, what's the transformation that most of them agree on?`
It works by:
1. Picking 4 random point pairs → computes a candidate homography
2. Testing all other point pairs against it and counting how many "agree" = `(inliers)`
Repeats thousands of times for different 4 points
Keeps the homography with the most inliers

This is why homography is robust to bad matches. Even if 40% of the feature matches are wrong, RANSAC finds the correct transform from the 60% that are right.