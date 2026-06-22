import cv2
import numpy as np


# OpenPose MPI Model


nPoints = 15

POSE_PAIRS = [
    [0,1],[1,2],[2,3],[3,4],
    [1,5],[5,6],[6,7],
    [1,14],[14,8],[8,9],[9,10],
    [14,11],[11,12],[12,13]
]

protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "model/pose_iter_160000.caffemodel"


# Load Model


print("Loading OpenPose Model...")

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

print("Model Loaded Successfully!")


# Read Image


image = cv2.imread("Tiger_Woods.png")

if image is None:
    print("Image not found!")
    exit()

frameWidth = image.shape[1]
frameHeight = image.shape[0]


# Convert Image to Blob


inpBlob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0 / 255,
    size=(368, 368),
    mean=(0, 0, 0),
    swapRB=False,
    crop=False
)

net.setInput(inpBlob)


# Forward Pass


output = net.forward()

H = output.shape[2]
W = output.shape[3]

threshold = 0.1

points = []


# Detect Keypoints


for i in range(nPoints):

    probMap = output[0, i, :, :]

    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold:
        points.append((int(x), int(y)))
    else:
        points.append(None)


# Draw Keypoints


imgPoints = image.copy()

for i, p in enumerate(points):

    if p:

        cv2.circle(
            imgPoints,
            p,
            8,
            (0, 255, 255),
            thickness=-1
        )

        cv2.putText(
            imgPoints,
            str(i),
            p,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


# Draw Skeleton


imgSkeleton = image.copy()

for pair in POSE_PAIRS:

    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:

        cv2.line(
            imgSkeleton,
            points[partA],
            points[partB],
            (0, 255, 255),
            2
        )

        cv2.circle(
            imgSkeleton,
            points[partA],
            5,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            imgSkeleton,
            points[partB],
            5,
            (0, 0, 255),
            -1
        )


# Save Outputs


cv2.imwrite("pose_points_output.png", imgPoints)
cv2.imwrite("pose_skeleton_output.png", imgSkeleton)

print("Outputs Saved!")


# Display Results


cv2.imshow("Detected Keypoints", imgPoints)
cv2.imshow("Skeleton", imgSkeleton)

cv2.waitKey(0)
cv2.destroyAllWindows()

