## Edge Detection and Gradients

An Edge is a place where pixel intensity changes rapidly.
Mathematically, edges are calculated using gradients. A gradient measures rate of change of intensity; in this case pixel intensity.
`High intensity` = rapid change = `edge`.
`Zero intensity` = no change = flat region/ uniform region/ surface = no `edge`.


## Day 1 --> Sobel 
computes gradients `x` and `y` separately and then combines them to get the full edge magitude. 
`X` direction = `Vertical edges`
`Y` direction = `Horizontal edges`
`Sobel is directional` - you can detect edges only in the horizontal orientation or only in the vertical orientation.
`Sobel X` subtracts the left neighbors from the right neighbors. If there's a sharp left-to-right intensity change (a vertical edge is detected), the result is large. If the region is flat, the subtraction is near zero.
`Sobel Y` does the same vertically — subtracts top from bottom. 
To get the full edge magnitude regardless of direction you combine them
`Sobel gives you a gradient magitude at every pixel`


## Day 2 --> Canny
Runs in 4 steps 
1. `Gaussian Blur` --> kills noise before gradient computation
2. `Sobel gradients` --> computes `X` and `Y` gradients and finds the magnitude
3. `Non-maximum suppresion` --> for every pixel, if not the strongest response, suppress to zero; this thins the edges to exactly 1 pixel wide thus giving the actual true edges as sharp lines
4. `Hysteresis thresholding` --> Two thresholds: threshold1 (low) and threshold2 (high).

Gradient above `threshold2` → definitely an edge (strong)
Gradient below `threshold1` → definitely not an edge, discarded
Gradient between them → only kept if it's connected to a strong edge
`This is what makes Canny's edges clean and connected instead of Sobel's noisy scattered pixels.`


## Day 3 --> Contours
From day 2, canny gives you a binnary image; white lines on a black background. There is no structure, cant tell hos big, what shape or where its center is. 
`Contours` gives structure.It traces the white lines from cannyin a binary image and groups them into connected curves closed or open.
Each contour is an arrya of `(x,y)` points that form that boundary.
from a single contour we can find
- Area
- Perimeter
- Bounding box
- Approximation
- Moments

`cv.findContours` has two important parameters:
`cv.findContours(binary_image, mode, method)`

* `mode` controls which contours are returned:
1. `cv.RETR_EXTERNAL` -> only the outermost contours. Ignores contours inside other contours. Used when you want the outline of objects only.
2. `cv.RETR_TREE` -> all contours with full hierarchy. A donut shape gives you the outer ring and the inner hole as separate contours with a parent-child relationship.

* `method` controls how contour points are stored:
1. `cv.CHAIN_APPROX_NONE` -> stores every single pixel on the boundary
2. `cv.CHAIN_APPROX_SIMPLE` -> stores only the endpoints of straight segments. A rectangle needs only 4 points instead of hundreds. 