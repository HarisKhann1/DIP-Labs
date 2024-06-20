import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the noisy image
image = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError("The image file was not found.")

# Display the original noisy image
plt.figure(figsize=(10, 8))
plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.axis('off')

# Apply mean filter for denoising
mean_filtered = cv2.blur(image, (5, 5))

# Display the mean filtered image
plt.subplot(222), plt.imshow(mean_filtered, cmap='gray'), plt.title('Mean Filtered')
plt.axis('off')

# Apply median filter for denoising
median_filtered = cv2.medianBlur(image, 5)

# Display the median filtered image
plt.subplot(223), plt.imshow(median_filtered, cmap='gray'), plt.title('Median Filtered')
plt.axis('off')

# Apply Gaussian filter for denoising
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

# Display the Gaussian filtered image
plt.subplot(224), plt.imshow(gaussian_filtered, cmap='gray'), plt.title('Gaussian Filtered')
plt.axis('off')

plt.tight_layout()
plt.show()
