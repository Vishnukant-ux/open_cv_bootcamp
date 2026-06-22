import cv2
import numpy as np
import matplotlib.pyplot as plt

# Exposure images
filenames = [
    "img_0.033.jpg",
    "img_0.25.jpg",
    "img_2.5.jpg",
    "img_15.jpg"
]

# Exposure times
times = np.array(
    [1/30.0, 0.25, 2.5, 15.0],
    dtype=np.float32
)

# Read images
images = []

for filename in filenames:
    img = cv2.imread(filename)

    if img is None:
        print("Could not load:", filename)
        exit()

    images.append(img)

# Align images
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# Camera response function
calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, times)

# Merge HDR
mergeDebevec = cv2.createMergeDebevec()
hdr = mergeDebevec.process(images, times, response)

# Drago Tonemapping

tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdr)

cv2.imwrite(
    "results/ldr-drago.jpg",
    np.clip(ldrDrago * 255, 0, 255).astype("uint8")
)


# Reinhard Tonemapping

tonemapReinhard = cv2.createTonemapReinhard(
    1.5, 0, 0, 0
)

ldrReinhard = tonemapReinhard.process(hdr)

cv2.imwrite(
    "results/ldr-reinhard.jpg",
    np.clip(ldrReinhard * 255, 0, 255).astype("uint8")
)


# Mantiuk Tonemapping

tonemapMantiuk = cv2.createTonemapMantiuk(
    2.2, 0.85, 1.2
)

ldrMantiuk = tonemapMantiuk.process(hdr)

cv2.imwrite(
    "results/ldr-mantiuk.jpg",
    np.clip(ldrMantiuk * 255, 0, 255).astype("uint8")
)

# Convert images for display

drago_display = cv2.cvtColor(
    np.clip(ldrDrago * 255, 0, 255).astype(np.uint8),
    cv2.COLOR_BGR2RGB
)

reinhard_display = cv2.cvtColor(
    np.clip(ldrReinhard * 255, 0, 255).astype(np.uint8),
    cv2.COLOR_BGR2RGB
)

mantiuk_display = cv2.cvtColor(
    np.clip(ldrMantiuk * 255, 0, 255).astype(np.uint8),
    cv2.COLOR_BGR2RGB
)

# Display all results

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(drago_display)
plt.title("Drago Tonemapping")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(reinhard_display)
plt.title("Reinhard Tonemapping")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mantiuk_display)
plt.title("Mantiuk Tonemapping")
plt.axis("off")

plt.tight_layout()
plt.show()

