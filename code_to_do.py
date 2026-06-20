import cv2
import matplotlib.pyplot as plt

# Read images
images = []

for i in range(1, 7):
    img = cv2.imread(f"s{i}.png")

    if img is None:
        print(f"Could not read s{i}.png")
    else:
        images.append(img)

print("Images loaded:", len(images))

# Create panorama
stitcher = cv2.Stitcher_create()

status, panorama = stitcher.stitch(images)

if status == cv2.Stitcher_OK:

    panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(18,8))
    plt.imshow(panorama)
    plt.axis("off")
    plt.title("Panorama Result")
    plt.show()

else:
    print("Stitching failed.")
    print("Error code:", status)

