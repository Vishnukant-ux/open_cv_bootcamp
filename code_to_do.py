import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images
im1 = cv2.imread("boat.jpg")      # Template
im2 = cv2.imread("coast.jpg")     # Image to align

# Convert BGR to RGB
im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# Display original images
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.imshow(im1_rgb)
plt.title("Template")
plt.axis("off")

plt.subplot(122)
plt.imshow(im2_rgb)
plt.title("Scanned Image")
plt.axis("off")

plt.show()

# Convert to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# ORB detector
MAX_NUM_FEATURES = 500

orb = cv2.ORB_create(MAX_NUM_FEATURES)

keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

# Draw keypoints
im1_display = cv2.drawKeypoints(
    im1_rgb,
    keypoints1,
    None,
    color=(255,0,0)
)

im2_display = cv2.drawKeypoints(
    im2_rgb,
    keypoints2,
    None,
    color=(255,0,0)
)

plt.figure(figsize=(12,6))

plt.subplot(121)
plt.imshow(im1_display)
plt.title("Template Keypoints")
plt.axis("off")

plt.subplot(122)
plt.imshow(im2_display)
plt.title("Scanned Keypoints")
plt.axis("off")

plt.show()

# Match descriptors
matcher = cv2.DescriptorMatcher_create(
    cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
)

matches = list(
    matcher.match(
        descriptors1,
        descriptors2,
        None
    )
)

matches.sort(key=lambda x: x.distance)

numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Draw matches
im_matches = cv2.drawMatches(
    im1_rgb,
    keypoints1,
    im2_rgb,
    keypoints2,
    matches,
    None
)

plt.figure(figsize=(18,8))
plt.imshow(im_matches)
plt.axis("off")
plt.title("Feature Matches")
plt.show()

# Extract locations
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(
    points2,
    points1,
    cv2.RANSAC
)

# Warp scanned image
height, width, channels = im1.shape

im2_reg = cv2.warpPerspective(
    im2,
    h,
    (width, height)
)

im2_reg_rgb = cv2.cvtColor(
    im2_reg,
    cv2.COLOR_BGR2RGB
)

# Display final alignment
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.imshow(im1_rgb)
plt.title("Template")

plt.axis("off")

plt.subplot(122)
plt.imshow(im2_reg_rgb)
plt.title("Aligned Image")

plt.axis("off")

plt.show()

# Save result
cv2.imwrite(
    "aligned_result.jpg",
    im2_reg
)

print("Alignment completed.")

