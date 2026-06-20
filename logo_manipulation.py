import cv2
import matplotlib.pyplot as plt

# Logo
logo_bgr = cv2.imread("coke.jpg")
logo_rgb = cv2.cvtColor(logo_bgr, cv2.COLOR_BGR2RGB)

# Background
bg_bgr = cv2.imread("checker_board.jpg")
bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

bg_rgb = cv2.resize(
    bg_rgb,
    (logo_rgb.shape[1], logo_rgb.shape[0])
)

# Create Mask
gray = cv2.cvtColor(
    logo_rgb,
    cv2.COLOR_RGB2GRAY
)

_, mask = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

mask_inv = cv2.bitwise_not(mask)

background = cv2.bitwise_and(
    bg_rgb,
    bg_rgb,
    mask=mask
)

foreground = cv2.bitwise_and(
    logo_rgb,
    logo_rgb,
    mask=mask_inv
)

result = cv2.add(
    background,
    foreground
)

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(logo_rgb)
plt.title("Logo")

plt.subplot(132)
plt.imshow(bg_rgb)
plt.title("Background")

plt.subplot(133)
plt.imshow(result)
plt.title("Final Result")

plt.show()
