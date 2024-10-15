import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    return image

# Function to enhance the image using sharpening and contrast adjustment
def enhance_image(image):
    # Sharpening kernel
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)

    # Adjust contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(sharpened)

    return enhanced_image

# Function to postprocess and save the image
def postprocess_and_save(image, save_path):
    cv2.imwrite(save_path, image)

# Load and enhance an image
#input_image_path = 'test.png'
input_image_path = '3a.png'
input_image = preprocess_image(input_image_path)

#real_image_path = 'real.png'
real_image_path = '3b.png'
real_image = preprocess_image(real_image_path)

enhanced_image = enhance_image(input_image)

# Save the enhanced image
#output_image_path = 'enhanced_image.png'
output_image_path = '3c.png'
postprocess_and_save(enhanced_image, output_image_path)

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('50% MRI RUN')
plt.axis('off')

# Sharpened Image
plt.subplot(1, 3, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('50% but Processed')
plt.axis('off')

# Enhanced Image
plt.subplot(1, 3, 3)
plt.imshow(real_image, cmap='gray')
plt.title('100% Run')
plt.axis('off')

plt.show()

plt.savefig('1_done.png')