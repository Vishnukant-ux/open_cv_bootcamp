import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images.jpeg", cv2.IMREAD_GRAYSCALE)

_, thresh1 = cv2.threshold(
    img,
    50,
    255,
    cv2.THRESH_BINARY
)

_, thresh2 = cv2.threshold(
    img,
    130,
    255,
    cv2.THRESH_BINARY
)

adaptive = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    7
)

plt.figure(figsize=(12,10))

plt.subplot(221)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(222)
plt.imshow(thresh1, cmap="gray")
plt.title("Global 50")

plt.subplot(223)
plt.imshow(thresh2, cmap="gray")
plt.title("Global 130")

plt.subplot(224)
plt.imshow(adaptive, cmap="gray")
plt.title("Adaptive")

plt.show()
