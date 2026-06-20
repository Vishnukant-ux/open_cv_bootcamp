# Important Points

## Homography

A homography relates two planar images.

At least 4 corresponding points are required.

## ORB

ORB = Oriented FAST and Rotated BRIEF

Used for:

- Feature Detection
- Feature Description

## Feature Matching

DescriptorMatcher with Hamming distance is used.

Only top 10% matches are retained.

## Image Alignment Steps

1. Read images
2. Convert to grayscale
3. Detect ORB keypoints
4. Compute descriptors
5. Match features
6. Estimate homography
7. Warp image using perspective transform

## OpenCV Functions

cv2.ORB_create()

cv2.detectAndCompute()

cv2.DescriptorMatcher_create()

cv2.findHomography()

cv2.warpPerspective()
