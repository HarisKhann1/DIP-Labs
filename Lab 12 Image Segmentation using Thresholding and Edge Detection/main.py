import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('Haris.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was properly loaded
if img is None:
    print("Error: Could not load input_image.jpg")
    exit()


# Edge detection using Sobel operator
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edge_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
edge_sobel = cv2.normalize(edge_sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Edge detection using Canny edge detector
canny_edges = cv2.Canny(img, 100, 200)

# Create a single figure for all plots
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Sobel Edge Detection
plt.subplot(2, 2, 3)
plt.imshow(edge_sobel, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

# Canny Edge Detection
plt.subplot(2, 2, 4)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
