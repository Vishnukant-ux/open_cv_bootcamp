import cv2
import numpy as np
import matplotlib.pyplot as plt

rect = np.zeros((200,500), dtype="uint8")
rect[:, :250] = 255

circle = np.zeros((200,500), dtype="uint8")
cv2.circle(circle, (250,100), 60, 255, -1)

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(rect, cmap="gray")
plt.title("Rectangle")

plt.subplot(122)
plt.imshow(circle, cmap="gray")
plt.title("Circle")

plt.show()

# AND

result = cv2.bitwise_and(rect, circle)

plt.imshow(result, cmap="gray")
plt.title("Bitwise AND")
plt.show()

# OR

result = cv2.bitwise_or(rect, circle)

plt.imshow(result, cmap="gray")
plt.title("Bitwise OR")
plt.show()

# XOR

result = cv2.bitwise_xor(rect, circle)

plt.imshow(result, cmap="gray")
plt.title("Bitwise XOR")
plt.show()

# NOT

result = cv2.bitwise_not(circle)

plt.imshow(result, cmap="gray")
plt.title("Bitwise NOT")
plt.show()
