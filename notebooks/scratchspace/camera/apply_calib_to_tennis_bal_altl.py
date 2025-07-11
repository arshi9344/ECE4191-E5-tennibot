
"""
This file takes an image of a tennis ball and applies the camera calibration matrix to reproject the tennis ball's
coordinates in 3D space. The steps are as follows:
1. Set IMAGE_PATH to the correct path of the tennis ball image.
2. Set CALIB_MATRIX to the path of the camera calibration matrix.
3. Define the color ranges for the tennis ball in the HSV color space. (You don't have to do this)

Then, the file:
- uses our OpenCV code (generated by Claude AI with background subtraction, etc.) to detect the tennis ball
- finds the pixel coordinates of the image of the tennis ball
- recenters the image coordinates so that the x-coordinates are centered at the image center (because our robot will be at the center of the image)
- reprojects the centered image coordinates to 3D space using the camera calibration matrix
- visualizes the original image, the 2D image points, and the reprojected 3D points

"""



import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Constants
IMAGE_PATH = 'tennis_ball/image_22.jpg'
CALIB_MATRIX = 'camera_calibration-2_3.npz'
Y_FLIP = True
X_FLIP = True
APERTURE_WIDTH_MM = 3.6
APERTURE_HEIGHT_MM = 2.7
IMAGE_SIZE = (640, 480)
TENNIS_BALL_DIAMETER_MM = 65  # Diameter of a tennis ball in mm

# Camera calibration parameters
FOCAL_LENGTH_MM = 4.15

# Define multiple color ranges for different lighting conditions
color_ranges = [
    (np.array([20, 40, 80]), np.array([80, 255, 255])),  # Yellow-green
    (np.array([15, 30, 70]), np.array([85, 255, 255]))  # Broader range
]


def calculate_pixels_per_mm(fx, image_width, sensor_width_mm):
    return fx / sensor_width_mm


def estimate_distance_with_only_ratio(real_radius_cm, pixel_radius, focal_length_pixels):
    """
    Estimate the distance to the tennis ball.

    :param real_radius_cm: The real-world radius of the tennis ball (in cm).
    :param pixel_radius: The detected radius of the tennis ball in pixels.
    :param focal_length_pixels: The camera's focal length in pixels (from camera calibration).
    :return: Estimated distance to the tennis ball in cm.
    """
    distance_cm = (focal_length_pixels * real_radius_cm) / pixel_radius

    return distance_cm

def estimate_distance_with_fx(matrix, r_m, r_px, u):
    dist = matrix[0, 0] * r_m / r_px
    horizontal_offset = (u - matrix[0, 2]) * (r_m / r_px)
    return dist, horizontal_offset



def distance_with_ratio_and_focal_length(object_size_px, pixels_per_mm, object_real_size_mm, focal_length_mm):
    object_size_mm = object_size_px / pixels_per_mm
    distance_mm = object_real_size_mm * focal_length_mm / object_size_mm
    return distance_mm


# Load calibration data
with np.load(CALIB_MATRIX) as calibration_data:
    mtx = calibration_data['camera_matrix']
    print("[fx 0 cx\n0 fy cy\n0 0 1]")
    print(mtx)

    fx = mtx[0, 0]  # Focal length in the x-axis
    cx = mtx[0, 2]  # Principal point in the x-axis
    fy = mtx[1, 1]  # Focal length in the y-axis
    cy = mtx[1, 2]  # Principal point in the y-axis
    dist = calibration_data['dist_coeffs']
    rvecs = calibration_data['rvecs']
    tvecs = calibration_data['tvecs']


fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, IMAGE_SIZE, 3.6, 2.7)
# Load the image
frame = cv2.imread(IMAGE_PATH)

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Apply a slight Gaussian blur to reduce noise
blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

# Convert the frame to the HSV color space
hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

# Create a mask for the tennis ball color (combine multiple ranges)
mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
for lower, upper in color_ranges:
    mask |= cv2.inRange(hsv, lower, upper)

# Apply background subtraction
fg_mask = bg_subtractor.apply(frame)

# Combine color mask with foreground mask
combined_mask = cv2.bitwise_and(mask, fg_mask)

# Perform morphological operations to remove noise and improve detection
kernel = np.ones((5, 5), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tennis Ball Centers and Sizes
tennis_ball_data = []

# Track the tennis balls
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Adjust this threshold as needed
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity > 0.7:  # Enforce circularity check
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            diameter = radius * 2
            tennis_ball_data.append((center, diameter))

            # Draw the circle and centroid on the frame
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 128, 255), -1)

# Calculate pixels per mm
# Assuming the image width in pixels is the native resolution width
image_width_px = 3264
sensor_width_mm = image_width_px / calculate_pixels_per_mm(fx, image_width_px, FOCAL_LENGTH_MM)
pixels_per_mm = calculate_pixels_per_mm(fx, frame.shape[1], sensor_width_mm)

### Calculate distances ###
for center, diameter in tennis_ball_data:
    # This is estimating distance using focal length. The focal length here is just a placeholder, not a real value
    distance = distance_with_ratio_and_focal_length(diameter, pixels_per_mm, TENNIS_BALL_DIAMETER_MM, FOCAL_LENGTH_MM)
    print(f"Tennis ball at {center}: Distance = {distance:.2f} mm")

    # Estimating distance using pin hole camera approach
    distance = estimate_distance_with_only_ratio(TENNIS_BALL_DIAMETER_MM, diameter, (fx + fy) / 2)
    print(f"Tennis ball at {center}: Distance = {distance:.2f} mm")

    dist, x_offset = estimate_distance_with_fx(mtx, TENNIS_BALL_DIAMETER_MM/1000, diameter/2, center[0])
    print(f"Tennis ball at {center}: Distance = {distance:.2f} mm, horizontal_offset{x_offset:.2f} mm")


# Display the resulting frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(frame_rgb)
plt.title("Detected Tennis Balls")
plt.axis('off')
plt.show()

# Existing 3D reprojection code can remain unchanged
# ...