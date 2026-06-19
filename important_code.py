# Here I will be giving all the codes that were covered in lecture 

#to_access_pixels
import cv2

img = cv2.imread("boat.jpg", 0)
print("Black Pixel:", img[0,0])
print("White Pixel:", img[0,6])

#to_modify_pixels
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("boat.jpg",0)

img_copy = img.copy()

img_copy[2,2] = 200
img_copy[2,3] = 200
img_copy[3,2] = 200
img_copy[3,3] = 200

plt.imshow(img_copy,cmap="gray")
plt.title("Modified Pixels")
plt.show()

#to_crop
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("checker_board.jpg")
img = img[:,:,::-1]

cropped = img[200:400,300:600]

plt.imshow(cropped)
plt.title("Cropped Image")
plt.show()

#to resize scale factor 
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images.jpeg")
img = img[:,:,::-1]

cropped = img[200:400,300:600]

resized = cv2.resize(cropped,None,fx=2,fy=2)

plt.imshow(resized)
plt.title("2x Resized")
plt.show()

#to resize customly 
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("boat.jpg")
img = img[:,:,::-1]

cropped = img[200:400,300:600]

dim = (100,200)

resized = cv2.resize(cropped,dim)

plt.imshow(resized)
plt.title("Custom Size")
plt.show()

#to change aspect ratio
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("boat.jpg")
img = img[:,:,::-1]

cropped = img[200:400,300:600]

desired_width = 100

aspect_ratio = desired_width / cropped.shape[1]

desired_height = int(cropped.shape[0] * aspect_ratio)

dim = (desired_width, desired_height)

resized = cv2.resize(cropped, dim)

plt.imshow(resized)
plt.title("Aspect Ratio Maintained")
plt.show()

