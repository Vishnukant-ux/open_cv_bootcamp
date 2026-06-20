# Panorama using OpenCV

## Theory

Panorama stitching combines multiple overlapping images into one wide image.

Steps:

1. Detect keypoints
2. Match keypoints
3. Estimate homography
4. Warp images
5. Blend images

## OpenCV Function 

stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)
