import cv2
import matplotlib.pyplot as plt

img = cv2.imread("coast.jpg", cv2.IMREAD_GRAYSCALE)

retval, thresh = cv2.threshold(
    img,
    100,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(122)
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded")

plt.show()
