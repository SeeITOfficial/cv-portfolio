import cv2 as cv
import numpy as np

img1 = cv.imread("../images/login1.png")
img2 = cv.imread("../images/nexx.jpeg")

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images not found. Please check the file paths.")

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#detecting ORB keypoints and descriptors
orb = cv.ORB_create(nfeatures=1000)
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

print(f"Number of keypoints in image 1: {len(keypoints1)}")
print(f"Number of keypoints in image 2: {len(keypoints2)}")

#match decriptors using BFMatcher; compares every descriptor from image 1 with every descriptor from image 2
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Total matches found: {len(matches)}")
print(f"Good matches found: {len(good_matches)}")

#computing homography
if len(good_matches) < 4:
    raise ValueError("Not enough good matches to compute homography. At least 4 are required.")

# Extract location of good matches
src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography using RANSAC
homography_matrix, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 3.0)

det = homography_matrix[0,0]*homography_matrix[1,1] - homography_matrix[0,1]*homography_matrix[1,0]
if det < 0:
    print("Homography rejected: invalid matrix.")

inliners = int(mask.sum())
print(f"Number of inliers found after RANSAC: {inliners} / {len(good_matches)}")

#warp img1 into img2's perspective
#Warping img1 into img2's perspective means: take every pixel from img1 and move it to where it would appear if the camera was in img2's position.
h2, w2 = img2.shape[:2]
warped = cv.warpPerspective(img1, homography_matrix, (w2, h2))

# Display the results
mask_flat = mask.ravel().tolist()
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask_flat[i]]

match_img = cv.drawMatches(
    img1, keypoints1,
    img2, keypoints2,
    inlier_matches[:50],   # draw max 50 matches
    None,
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

h_m, w_m = match_img.shape[:2]
small_match = cv.resize(match_img, (w_m // 2, h_m // 2))

h1, w1 = img1.shape[:2]
small = (w1 // 2, h1 // 2)

top    = np.hstack([cv.resize(img1,    small),
                    cv.resize(img2,    small)])
bottom = np.hstack([cv.resize(warped,  small),
                    cv.resize(img2,    small)])

grid = np.vstack([top, bottom])

cv.namedWindow('Homography', cv.WINDOW_NORMAL)
cv.imshow('Homography', grid)

cv.namedWindow('Inlier Matches', cv.WINDOW_NORMAL)
cv.imshow('Inlier Matches', small_match)

print("Top-left: img1 | Top-right: img2")
print("Bottom-left: img1 warped into img2 perspective | Bottom-right: img2")
print("ESC to close")

while True:
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()