import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('Haris.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError("The image file was not found.")

# Apply median filter
median_filter = cv2.medianBlur(image, 5)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.axis('off')
plt.subplot(122), plt.imshow(median_filter, cmap='gray'), plt.title('Median Filter')
plt.axis('off')
plt.tight_layout()
plt.show()
