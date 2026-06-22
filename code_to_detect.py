import cv2
import time

# Load DNN model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Read image
image = cv2.imread("test_image.jpg")

if image is None:
    print("Could not load image")
    exit()

(h, w) = image.shape[:2]

# Create blob
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0,
    (300, 300),
    (104, 117, 123)
)

net.setInput(blob)

start = time.time()
detections = net.forward()
end = time.time()

# Detect faces
for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.7:

        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (x1, y1, x2, y2) = box.astype("int")

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = f"{confidence*100:.2f}%"

        cv2.putText(
            image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# Inference time
cv2.putText(
    image,
    f"Time: {(end-start)*1000:.2f} ms",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 0, 0),
    2
)

# Save output
cv2.imwrite("output.jpg", image)

# Display output
cv2.imshow("Face Detection Output", image)

print("Output saved as output.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

