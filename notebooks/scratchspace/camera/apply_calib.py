"""
This is a file to test our camera calibration matrix. First, you need to:
- Fill out the image path to test it on
- Then, fill out the path to the calibration matrix that you want to test
Then, the file will:
- Un-distort the image using the calibration matrix
- Generate a bunch of random 3D points, and project these onto a 2D image plane
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the calibration data
with np.load('camera_calibration-2_7.npz') as calibration_data:
    mtx = calibration_data['camera_matrix']  # Camera matrix
    dist = calibration_data['dist_coeffs']  # Distortion coefficients
    rvecs = calibration_data['rvecs']  # Rotation vectors
    tvecs = calibration_data['tvecs']  # Translation vectors

def undistort_image(image):
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted = cv2.undistort(image, mtx, dist, None, newcameramtx)
    return undistorted

def project_points(object_points):
    image_points, _ = cv2.projectPoints(object_points, rvecs[0], tvecs[0], mtx, dist)
    return image_points.reshape(-1, 2)

# Load an image
img = cv2.imread('images_calib3/images_calib3_image_28.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Undistort the image
undistorted_img = undistort_image(img)
undistorted_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

# Visualize original and undistorted images
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(undistorted_rgb)
plt.title('Undistorted Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Create some 3D points
x, y, z = np.meshgrid(np.arange(-1, 2, 1), np.arange(-1, 2, 1), np.arange(-1, 2, 1))
object_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
object_points = object_points.astype(np.float32)

# Project 3D points to 2D image plane
image_points = project_points(object_points)

# Visualize 3D points and their 2D projections
fig = plt.figure(figsize=(12, 5))

# 3D plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Points')

# 2D projection plot
ax2 = fig.add_subplot(122)
ax2.scatter(image_points[:, 0], image_points[:, 1])
ax2.set_xlim(0, img.shape[1])
ax2.set_ylim(img.shape[0], 0)  # Invert y-axis to match image coordinates
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('2D Projected Points')

plt.tight_layout()
plt.show()