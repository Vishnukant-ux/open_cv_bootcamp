import cv2
import sys
import numpy as np

PREVIEW = 0      # Preview Mode
BLUR = 1         # Blurring Filter
FEATURES = 2     # Corner Feature Detector
CANNY = 3        # Canny Edge Detector

feature_params = dict(
    maxCorners=500,
    qualityLevel=0.2,
    minDistance=15,
    blockSize=9
)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()

    if not has_frame:
        break

    if image_filter == PREVIEW:
        result = frame

    elif image_filter == BLUR:
        result = cv2.GaussianBlur(frame, (13, 13), 0)

    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)

    elif image_filter == FEATURES:
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            **feature_params
        )

        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 5, (0, 255, 0), 1)

    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)

    if key == 27:      # ESC
        alive = False
    elif key == ord('1'):
        image_filter = PREVIEW
    elif key == ord('2'):
        image_filter = BLUR
    elif key == ord('3'):
        image_filter = FEATURES
    elif key == ord('4'):
        image_filter = CANNY

source.release()
cv2.destroyAllWindows()
