import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('Haris.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was properly loaded
if img is None:
    print("Error: Could not load input_image.jpg")
    exit()

# Display the original image
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate and display the histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Plot the histogram using Matplotlib
plt.figure(figsize=(8, 6))
plt.plot(hist, color='black')
plt.title('Histogram of the Original Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(True)
plt.show()

# Calculate statistical measurements
mean_value = np.mean(img)
variance_value = np.var(img)
std_deviation = np.sqrt(variance_value)

# Display statistical measurements
print(f"Mean: {mean_value}")
print(f"Variance: {variance_value}")
print(f"Standard Deviation: {std_deviation}")
