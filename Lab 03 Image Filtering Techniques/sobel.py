import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('Haris.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError("The image file was not found.")

# Apply Sobel filter
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobelx, sobely)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.axis('off')
plt.subplot(132), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.axis('off')
plt.subplot(133), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.axis('off')
plt.tight_layout()
plt.show()
