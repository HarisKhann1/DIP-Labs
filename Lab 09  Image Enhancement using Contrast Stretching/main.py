import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('Haris.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was properly loaded
if img is None:
    print("Error: Could not load input_image.jpg")
    exit()

# Apply contrast stretching
min_intensity = np.min(img)
max_intensity = np.max(img)
stretched_img = (img - min_intensity) * (255.0 / (max_intensity - min_intensity))

# Convert to uint8 (0-255 range)
stretched_img = np.uint8(stretched_img)

# Display both original and stretched images in one frame
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Stretched Image
plt.subplot(1, 2, 2)
plt.imshow(stretched_img, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')

plt.tight_layout()
plt.show()
