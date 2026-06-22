import cv2
import numpy as np

# Load COCO Labels

with open("coco_class_labels.txt", "r") as f:
    labels = f.read().strip().split("\n")

# Load TensorFlow Model

modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


# Read Image

image = cv2.imread("images/street.jpg")

if image is None:
    print("Image not found!")
    exit()

h, w = image.shape[:2]


# Create Blob

blob = cv2.dnn.blobFromImage(
    image,
    size=(300, 300),
    swapRB=True,
    crop=False
)

net.setInput(blob)


# Perform Detection

detections = net.forward()


# Draw Results

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.25:

        class_id = int(detections[0, 0, i, 1])

        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        label = labels[class_id]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

# Display Output

cv2.imshow("TF Object Detection", image)

# Save output image
cv2.imwrite("detected_output.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

