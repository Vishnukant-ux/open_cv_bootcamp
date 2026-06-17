import cv2
import matplotlib.pyplot as plt

img = cv2.imread("coke.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb)
plt.title("RGB Image")
plt.show()

img that i used=(https://companieslogo.com/coca-cola/logo/)
