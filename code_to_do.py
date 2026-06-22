import cv2

# SELECT TRACKER


tracker_type = "KCF"  # KCF, CSRT, MOSSE

if tracker_type == "KCF":
    tracker = cv2.legacy.TrackerKCF_create()

elif tracker_type == "CSRT":
    tracker = cv2.legacy.TrackerCSRT_create()

elif tracker_type == "MOSSE":
    tracker = cv2.legacy.TrackerMOSSE_create()

else:
    print("Invalid tracker type")
    exit()


# LOAD VIDEO


video = cv2.VideoCapture("race_car.mp4")

if not video.isOpened():
    print("Could not open video")
    exit()

success, frame = video.read()

if not success:
    print("Could not read video")
    exit()


# OUTPUT VIDEO


width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(video.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "tracked_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps_video,
    (width, height)
)


# SELECT ROI


bbox = cv2.selectROI(
    "Select Object",
    frame,
    fromCenter=False,
    showCrosshair=True
)

tracker.init(frame, bbox)


# TRACK OBJECT


while True:

    success, frame = video.read()

    if not success:
        break

    timer = cv2.getTickCount()

    success, bbox = tracker.update(frame)

    fps = cv2.getTickFrequency() / (
        cv2.getTickCount() - timer
    )

    if success:

        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Tracking Failure",
            (100, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        tracker_type + " Tracker",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (50, 170, 50),
        2
    )

    cv2.putText(
        frame,
        "FPS : " + str(int(fps)),
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (50, 170, 50),
        2
    )

    # Save frame
    out.write(frame)

    # Display live tracking
    cv2.imshow("Object Tracking", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

video.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as tracked_output.mp4")


# PLAY FINAL OUTPUT VIDEO


final_video = cv2.VideoCapture("tracked_output.mp4")

while True:

    ret, frame = final_video.read()

    if not ret:
        break

    cv2.imshow("Final Output Video", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

final_video.release()
cv2.destroyAllWindows()

