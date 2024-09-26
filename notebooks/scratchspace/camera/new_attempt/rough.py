import cv2
import numpy as np

# Load the calibration data
calib_data = np.load('new_attempt/camera_calibration-2_7.npz')

# Extract the camera matrix and distortion coefficients
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

print("Camera Matrix:", camera_matrix)
print("Distortion Coefficients:", dist_coeffs)

# Read the distorted image
distorted_image = cv2.imread('tennis_ball/image_0.jpg')

# Undistort the image
undistorted_image = cv2.undistort(distorted_image, camera_matrix, dist_coeffs)

# Display the original distorted image
cv2.imshow('Distorted Image', distorted_image)

# Display the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)

# Wait for a key press to close the windows
cv2.waitKey(0)  # Press any key to close the image windows
cv2.destroyAllWindows()  # Close all windows
