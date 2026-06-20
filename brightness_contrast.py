import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("coast.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Brightness
matrix = np.ones(img_rgb.shape, dtype="uint8") * 50

brighter = cv2.add(img_rgb, matrix)
darker = cv2.subtract(img_rgb, matrix)

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(darker)
plt.title("Darker")

plt.subplot(132)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(133)
plt.imshow(brighter)
plt.title("Brighter")

plt.show()

# Contrast

matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

lower_contrast = np.uint8(
    cv2.multiply(np.float64(img_rgb), matrix1)
)

higher_contrast = np.uint8(
    np.clip(
        cv2.multiply(np.float64(img_rgb), matrix2),
        0,
        255
    )
)

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(lower_contrast)
plt.title("Lower Contrast")

plt.subplot(132)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(133)
plt.imshow(higher_contrast)
plt.title("Higher Contrast")

plt.show()
