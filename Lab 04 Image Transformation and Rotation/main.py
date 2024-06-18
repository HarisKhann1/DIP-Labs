import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('Haris.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError("The image file was not found.")

# Get the image dimensions (height and width)
height, width = image.shape

# Translation: Moving the image
# Define the translation matrix
tx, ty = 50, 30  # translation in x and y directions
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation to the image
translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

# Rotation: Rotating the image
# Define the rotation matrix
center = (width // 2, height // 2)  # center of the image
angle = 45  # angle of rotation in degrees
scale = 1.0  # scale factor
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation to the image
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Scaling: Resizing the image
# Define the scaling factors
scale_x, scale_y = 1.5, 1.5  # scaling factors in x and y directions

# Apply the scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Plot the original and transformed images
plt.figure(figsize=(12, 8))

plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.axis('off')

plt.subplot(222), plt.imshow(translated_image, cmap='gray'), plt.title('Translated')
plt.axis('off')

plt.subplot(223), plt.imshow(rotated_image, cmap='gray'), plt.title('Rotated')
plt.axis('off')

plt.subplot(224), plt.imshow(scaled_image, cmap='gray'), plt.title('Scaled')
plt.axis('off')

plt.tight_layout()
plt.show()
