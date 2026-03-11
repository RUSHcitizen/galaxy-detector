from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# When you want to run type ~/run_galaxy.sh in terminal
# Load the real Webb Deep Field image
print("Loading Webb Deep Field image...")
img = Image.open("/home/kumariaaatharv/deepfield.jpg")
img_array = np.array(img)

print(f"Image size: {img.size}")
print(f"Image shape: {img_array.shape}")
print(f"Max brightness: {img_array.max()}")
print(f"Min brightness: {img_array.min()}")

# Display it
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.title("Webb First Deep Field - Loaded in Python!")
plt.axis('off')
plt.show()
print("Done!")

# Find the brightest regions
brightness = img_array.mean(axis=2)
print(f"Average brightness: {brightness.mean():.2f}")
print(f"Number of very bright pixels: {(brightness > 200).sum()}")
print(f"Number of dark pixels: {(brightness < 20).sum()}")

from scipy import ndimage

# Find bright objects and circle them
threshold = 200
bright_mask = brightness > threshold

# Label connected bright regions (each galaxy/star)
labeled, num_objects = ndimage.label(bright_mask)
print(f"Number of bright objects found: {num_objects}")

# Draw circles around each object
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)
ax.set_title(f"Webb Deep Field - {num_objects} objects detected!")
ax.axis('off')

for i in range(1, num_objects + 1):
    # Find center of each object
    center = ndimage.center_of_mass(bright_mask, labeled, i)
    y, x = center
    circle = plt.Circle((x, y), 8, color='red', fill=False, linewidth=1)
    ax.add_patch(circle)

plt.tight_layout()
plt.show()
print("Detection complete!")
