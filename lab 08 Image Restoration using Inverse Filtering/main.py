import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import matplotlib.pyplot as plt  # Matplotlib library for displaying images

# Read the input image in grayscale
image = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)  # Make sure to provide the correct path to your image

# Add Gaussian blur to the image
kernel_size = 21  # Size of the Gaussian kernel
sigma = 5  # Standard deviation for the Gaussian kernel

# Create a Gaussian kernel
kernel = cv2.getGaussianKernel(kernel_size, sigma)  # 1D Gaussian kernel
kernel = kernel * kernel.T  # Convert the 1D kernel to a 2D kernel

# Apply the Gaussian kernel to the image
blurred_image = cv2.filter2D(image, -1, kernel)

# Perform inverse filtering to deblur the image

# Perform Fourier Transform on the blurred image and the kernel
blurred_fft = np.fft.fft2(blurred_image)  # Fourier transform of the blurred image
kernel_fft = np.fft.fft2(kernel, s=blurred_image.shape)  # Fourier transform of the kernel

# Add a small constant to the kernel's Fourier transform to avoid division by zero
kernel_fft = kernel_fft + 1e-8

# Perform inverse filtering by dividing the Fourier transform of the blurred image by the Fourier transform of the kernel
restored_fft = blurred_fft / kernel_fft

# Apply the inverse Fourier Transform to get the restored image back in the spatial domain
restored_image = np.fft.ifft2(restored_fft)

# Take the real part of the restored image
restored_image = np.real(restored_image)

# Display the original, blurred, and restored images using Matplotlib
plt.figure(figsize=(15, 5))  # Create a figure with a specified size

plt.subplot(1, 3, 1)  # Create the first subplot
plt.imshow(image, cmap='gray')  # Display the original image
plt.title('Original Image')  # Set the title of the subplot
plt.axis('off')  # Turn off the axis

plt.subplot(1, 3, 2)  # Create the second subplot
plt.imshow(blurred_image, cmap='gray')  # Display the blurred image
plt.title('Blurred Image')  # Set the title of the subplot
plt.axis('off')  # Turn off the axis

plt.subplot(1, 3, 3)  # Create the third subplot
plt.imshow(restored_image, cmap='gray')  # Display the restored image
plt.title('Restored Image')  # Set the title of the subplot
plt.axis('off')  # Turn off the axis

plt.show()  # Show the figure
