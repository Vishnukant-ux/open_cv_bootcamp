import cv2

# Open video file
cap = cv2.VideoCapture("race.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("Width :", frame_width)
print("Height:", frame_height)
print("FPS   :", fps)

# Create output video
out = cv2.VideoWriter(
    "race_car_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Write frame to output file
    out.write(frame)

    # Display frame
    cv2.imshow("Video Writing Demo", frame)

    # Press ESC to stop
    if cv2.waitKey(25) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as race_car_output.mp4")

video used =(https://youtube.com/shorts/qEnRdIA-5mM?si=J0kUAY4GvS9UWwpV)
