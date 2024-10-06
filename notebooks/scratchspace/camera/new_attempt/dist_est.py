import cv2
import numpy as np
import os
import torch
import pathlib
import math

# Define real tennis ball radius in meters (3.25 cm radius)
TENNIS_BALL_RADIUS_M = 0.0325

# Function to undistort the image
def undistort(frame, camera_matrix, dist_coeffs):
    return cv2.undistort(frame, camera_matrix, dist_coeffs)

# Function to calculate the distance to the object (tennis ball)
def calculate_distance(camera_matrix, u, v, r_px, r_m=TENNIS_BALL_RADIUS_M):
    """
    Calculate the distance to the tennis ball.
    :param camera_matrix: The camera matrix (intrinsics)
    :param u: The x-coordinate of the detected object (in pixels)
    :param v: The y-coordinate of the detected object (in pixels)
    :param r_px: The detected radius of the object in pixels
    :param r_m: The real-world radius of the object in meters (default is 0.0325m for a tennis ball)
    :return: (distance, angle) where distance is the distance to the object in meters, and angle is the angle in radians
    """
    fx = camera_matrix[0, 0]  # Focal length in x direction
    cx = camera_matrix[0, 2]  # Principal point x-coordinate
    
    # Distance to the object in meters
    distance = fx * r_m / r_px
    
    # Calculate the offset in the x-direction
    y = (u - cx) * (r_m / r_px)
    
    # Calculate the angle
    angle = math.asin(y / distance)
    
    return distance, angle

# Pathlib management (this part from your previous code)
emp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Define the model path (assuming it's in 'new_attempt/best.pt')
model_path = pathlib.Path('new_attempt/best.pt')

# Ensure the path is converted to a string
model_path_str = str(model_path)

# Assuming 'folder_path' is the path to the folder containing the files in 'new_attempt'
folder_path = 'new_attempt'

# List of .npy file names
file_names = ['camera_matrix (1).npy', 'distortion (1).npy', 'square_size (1).npy']

# Initialize an empty list to store the loaded arrays
new_arrays = []

# Loop through the file names and load each .npy file into the list
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)  # Proper way to join paths
    new_array = np.load(file_path)  # Load the .npy file
    new_arrays.append(new_array)  # Append the loaded array to the list

# Now 'new_arrays' contains the loaded arrays from the .npy files
camera_matrix, distortion_coeffs, square_size = new_arrays

# Print the loaded arrays to verify
print("Camera Matrix:", camera_matrix)
print("Distortion Coefficients:", distortion_coeffs)
print("Square Size:", square_size)

# Read the distorted image
distorted_image = cv2.imread('tennis_ball/image_22.jpg')

# Undistort the image using the calibration data
undistorted_image = undistort(distorted_image, camera_matrix, distortion_coeffs)

# Load YOLO model for tennis ball detection
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path_str, trust_repo=True)

# Perform inference on the undistorted image
results = model(undistorted_image)

# Extract bounding box and center of detected tennis ball
# YOLO results should give bounding boxes in the form (x1, y1, x2, y2) and confidence scores
# Assume we're detecting a single tennis ball with the highest confidence
detections = results.xyxy[0].cpu().numpy()  # Get detections in numpy format
if len(detections) > 0:
    # Get the most confident detection (assuming it's the tennis ball)
    x1, y1, x2, y2, confidence, class_id = detections[0]

    # Calculate the center (u, v) and radius in pixels
    u = (x1 + x2) / 2  # x-coordinate of the center
    v = (y1 + y2) / 2  # y-coordinate of the center
    r_px = (x2 - x1) / 2  # Radius of the bounding box in pixels

    # Calculate distance and angle to the tennis ball
    distance, angle = calculate_distance(camera_matrix, u, v, r_px)
    
    # Print the results
    print(f"Distance to the tennis ball: {distance:.2f} meters")
    print(f"Angle to the tennis ball: {math.degrees(angle):.2f} degrees")
else:
    print("No tennis ball detected.")

# Display the undistorted image with detection results
results.show()

# Optionally display the original distorted and undistorted images
cv2.imshow('Distorted Image', distorted_image)
cv2.imshow('Undistorted Image', undistorted_image)

# Wait for a key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
