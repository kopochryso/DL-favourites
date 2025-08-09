import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img_path = r'C:\Users\chryk\Downloads\InnoPP\weed-mapping\OLA-test\ntomates.jpg'
image = cv2.imread(img_path, 0)

if image is None:
    raise ValueError("Image not found or path incorrect")

# Apply GaussianBlur if you want (optional)
image_blur = cv2.GaussianBlur(image, (5,5), 0)

# Calculate histogram with 256 bins (0-255)
hist, bins = np.histogram(image_blur, bins=256, range=(0,256))

# Calculate Otsu threshold using OpenCV (auto threshold)
otsu_thresh, _ = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu threshold value: {otsu_thresh}")


# Apply Otsu threshold to create binary image
_, binary_img = cv2.threshold(image_blur, otsu_thresh, 255, cv2.THRESH_BINARY)
# Resize for better display if needed
resize_factor = 0.5
height, width = binary_img.shape
binary_img_resized = cv2.resize(binary_img, (int(width*resize_factor), int(height*resize_factor)))
cv2.imshow("Binary Image (Otsu Threshold)", binary_img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Plot histogram
plt.figure(figsize=(10,5))
plt.title('Image Histogram with Otsu Threshold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.plot(bins[:-1], hist, color='black')

# Mark threshold with a red vertical line
plt.axvline(x=otsu_thresh, color='red', linestyle='--', label=f'Threshold = {otsu_thresh:.2f}')
plt.legend()
plt.show()


