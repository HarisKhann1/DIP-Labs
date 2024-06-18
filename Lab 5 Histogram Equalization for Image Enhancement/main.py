import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Load the image
image = cv2.imread('Haris.jpg', 0)  # Load the image in grayscale

# Step 2: Calculate the histogram of the image
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# Step 3: Calculate the cumulative distribution function (CDF) of the histogram
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize CDF for better visualization

# Step 4: Perform histogram equalization
cdf_m = np.ma.masked_equal(cdf, 0)  # Mask zeros to avoid division errors
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Perform equalization

# Step 5: Replace the original pixel intensities with their equalized values
image_equalized = cdf[image]

# Step 6: Display original image and equalized image for comparison
plt.subplot(121), plt.imshow(image, 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image_equalized, 'gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
plt.show()
