import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("Apollo_11.jpg")

# Display Original Image
plt.figure(figsize=(8,6))
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis("off")
plt.show()


# Draw Line

imageLine = image.copy()

cv2.line(
    imageLine,
    (200, 100),
    (400, 100),
    (0, 255, 255),
    thickness=5,
    lineType=cv2.LINE_AA
)

cv2.imwrite("outputs/line.jpg", imageLine)

plt.figure(figsize=(8,6))
plt.imshow(imageLine[:, :, ::-1])
plt.title("Line Annotation")
plt.axis("off")
plt.show()


# Draw Circle

imageCircle = image.copy()

cv2.circle(
    imageCircle,
    (900, 500),
    100,
    (0, 0, 255),
    thickness=5,
    lineType=cv2.LINE_AA
)

cv2.imwrite("outputs/circle.jpg", imageCircle)

plt.figure(figsize=(8,6))
plt.imshow(imageCircle[:, :, ::-1])
plt.title("Circle Annotation")
plt.axis("off")
plt.show()


# Draw Rectangle

imageRectangle = image.copy()

cv2.rectangle(
    imageRectangle,
    (500, 100),
    (700, 600),
    (255, 0, 255),
    thickness=5,
    lineType=cv2.LINE_8
)

cv2.imwrite("outputs/rectangle.jpg", imageRectangle)

plt.figure(figsize=(8,6))
plt.imshow(imageRectangle[:, :, ::-1])
plt.title("Rectangle Annotation")
plt.axis("off")
plt.show()


# Add Text

imageText = image.copy()

cv2.putText(
    imageText,
    "Apollo 11 Saturn V Launch, July 16, 1969",
    (200, 700),
    cv2.FONT_HERSHEY_PLAIN,
    2.3,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)

cv2.imwrite("outputs/text.jpg", imageText)

plt.figure(figsize=(8,6))
plt.imshow(imageText[:, :, ::-1])
plt.title("Text Annotation")
plt.axis("off")
plt.show()

print("All images saved and displayed successfully!")
