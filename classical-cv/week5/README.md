# Week 5 - Histograms and Color Analysis

## What I built
Three scripts covering histogram computation, contrast enhancement,
and histogram back-projection for color detection.

## Scripts
- `day1_histogram.py` - per-channel BGR histogram plotting with cv.calcHist
- `day2_clahe.py` - global equalizeHist vs CLAHE vs LAB+CLAHE on color images
- `day3_backprojection.py` - interactive ROI selection + probability map generation

## Key concepts
- Histogram x-axis = intensity (0–255), y-axis = pixel count
- CLAHE operates per-tile to avoid blowing out uniform regions
- Back-projection outputs a probability map
- HSV hue channel is lighting-invariant - use it for color detection, not BGR

## Sample output
 Refer to output folder in week5
 