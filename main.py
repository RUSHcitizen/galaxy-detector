from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage
import sys
import os

def analyze_image(image_path):
    """
    Analyzes a space image and detects bright astronomical objects.
    Returns detection results and displays annotated image.
    """

    # ── Load image ──────────────────────────────────────────────
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        print("Make sure the image is in your home folder: /home/kumariaaatharv/")
        return

    print(f"\n{'='*50}")
    print(f"GALAXY DETECTOR - Analyzing: {os.path.basename(image_path)}")
    print(f"{'='*50}")

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    total_pixels = img_array.shape[0] * img_array.shape[1]

    print(f"Image size:     {img.size[0]} x {img.size[1]} pixels")
    print(f"Total pixels:   {total_pixels:,}")

    # ── Brightness analysis ──────────────────────────────────────
    brightness = img_array.mean(axis=2)
    avg_brightness = brightness.mean()
    max_brightness = brightness.max()

    print(f"Avg brightness: {avg_brightness:.2f} / 255")
    print(f"Max brightness: {max_brightness:.2f} / 255")

    # ── Smart threshold ──────────────────────────────────────────
    # Instead of hardcoding 200, we adapt to the image's brightness
    # This makes it work on any image, not just our specific one
    threshold = avg_brightness + (max_brightness - avg_brightness) * 0.75
    threshold = max(threshold, 150)  # Never go below 150
    print(f"Detection threshold: {threshold:.1f}")

    # ── Object detection ─────────────────────────────────────────
    bright_mask = brightness > threshold
    bright_pixel_count = bright_mask.sum()

    # Label connected bright regions
    labeled, num_objects = ndimage.label(bright_mask)

    # Filter out tiny noise (objects smaller than 3 pixels)
    real_objects = []
    for i in range(1, num_objects + 1):
        size = (labeled == i).sum()
        if size >= 3:
            real_objects.append(i)

    num_real = len(real_objects)

    print(f"\nBright pixels:  {bright_pixel_count:,} ({bright_pixel_count/total_pixels*100:.2f}% of image)")
    print(f"Raw detections: {num_objects}")
    print(f"Real objects:   {num_real} (filtered noise)")
    print(f"\n>>> DETECTED {num_real} BRIGHT ASTRONOMICAL OBJECTS <<<")

    # ── Size breakdown ───────────────────────────────────────────
    sizes = [(labeled == i).sum() for i in real_objects]
    if sizes:
        large = sum(1 for s in sizes if s > 50)
        medium = sum(1 for s in sizes if 10 < s <= 50)
        small = sum(1 for s in sizes if 3 <= s <= 10)
        print(f"    Large objects  (>50px):  {large}  — likely nearby galaxies")
        print(f"    Medium objects (10-50px): {medium}  — likely distant galaxies")
        print(f"    Small objects  (3-10px):  {small}  — likely distant stars/galaxies")

    # ── Visualization ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('black')

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", color='white', fontsize=13, pad=10)
    axes[0].axis('off')

    # Annotated image
    axes[1].imshow(img)
    axes[1].set_title(f"{num_real} Astronomical Objects Detected",
                      color='white', fontsize=13, pad=10)
    axes[1].axis('off')

    # Draw circles with size-based colors
    for i in real_objects:
        center = ndimage.center_of_mass(bright_mask, labeled, i)
        y, x = center
        size = (labeled == i).sum()

        if size > 50:
            color, radius = 'red', 12
        elif size > 10:
            color, radius = 'yellow', 7
        else:
            color, radius = 'cyan', 4

        circle = plt.Circle((x, y), radius, color=color, fill=False, linewidth=0.8)
        axes[1].add_patch(circle)

    # Legend
    legend_elements = [
        mpatches.Patch(color='red', label=f'Large objects ({large})'),
        mpatches.Patch(color='yellow', label=f'Medium objects ({medium})'),
        mpatches.Patch(color='cyan', label=f'Small objects ({small})')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right',
                   facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)

    plt.suptitle(f"Galaxy Detector — {os.path.basename(image_path)}",
                 color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    print(f"\nAnalysis complete!")
    print(f"{'='*50}\n")

    return num_real


def main():
    print("\n" + "="*50)
    print("   WEBB/HUBBLE GALAXY DETECTOR")
    print("   by Atharv Kumaria")
    print("="*50)

    # If filename given as argument, use that
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        analyze_image(image_path)
        return

    # Otherwise ask user for filename
    print("\nImages should be in your home folder: /home/kumariaaatharv/")
    print("Example filenames: deepfield.jpg, hubble.jpg, webb2.png")
    print("\nType 'quit' to exit\n")

    while True:
        filename = input("Enter image filename (or 'quit'): ").strip()

        if filename.lower() == 'quit':
            print("Goodbye! Keep exploring the universe. 🌌")
            break

        # Check if they typed full path or just filename
        if filename.startswith('/'):
            image_path = filename
        else:
            image_path = f"/home/kumariaaatharv/{filename}"

        analyze_image(image_path)

        another = input("\nAnalyze another image? (yes/no): ").strip().lower()
        if another != 'yes' and another != 'y':
            print("Goodbye! Keep exploring the universe.")
            break


if __name__ == "__main__":
    main()