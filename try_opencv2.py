# filter pics without downscaling image
import cv2
import numpy as np

# Load the image
image = cv2.imread('tennis_ball10.jpg')

# Define a size threshold (e.g., max width or height of 1000 pixels)
max_dimension = 1000

# Check if the image is too large
if image.shape[0] > max_dimension or image.shape[1] > max_dimension:
    # Calculate the scale factor
    scale_factor = max_dimension / float(max(image.shape[0], image.shape[1]))
    
    # Calculate the new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    dim = (new_width, new_height)
    
    # Resize the image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
else:
    # Use the original image if it is not too large
    resized_image = image

output = resized_image.copy()

# Convert the image to the HSV color space
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Define the flexible range for the tennis ball color (morning and afternoon)
lower_yellow_green = np.array([20, 50, 60])
upper_yellow_green = np.array([60, 255, 255])

# Create a mask for the tennis ball color
mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

# Apply the mask to get the yellow-green parts of the image
masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

# Convert the masked image to grayscale
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Use the HoughCircles function to detect circles
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, 
    param1=100, param2=30, minRadius=15, maxRadius=50
)

# Ensure at least some circles were found
if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    
    # Loop over the circles and draw them on the image
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Display the result
cv2.imshow("Tennis Ball Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
