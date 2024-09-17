
"""
This is a file to test our camera calibration matrix. What this does is:
- Generate some 2D points (these are stand-ins for pixel coordinates for a tennis ball)
- Reproject these to 3D coordinates, assuming they lie on a plane at Z=0
- Visualize the 2D points and the reprojected 3D points
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


def reproject_to_3d(image_points, Z=0):
    """
    Reproject 2D points to 3D assuming they lie on a plane at Z=0

    :param image_points: Nx2 array of 2D image points
    :param Z: Z-coordinate of the plane (default is 0)
    :return: Nx3 array of 3D points
    """
    # Ensure image_points is a numpy array
    image_points = np.array(image_points, dtype=np.float32)

    # Get rotation and translation vectors
    rvec, tvec = rvecs[0], tvecs[0]

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Invert [R|t]
    Rt = np.column_stack((R, tvec))
    P = mtx.dot(Rt)
    P_inv = np.linalg.pinv(P)

    # Add homogeneous coordinate to image points
    image_points_homogeneous = np.column_stack((image_points, np.ones(len(image_points))))

    # Reproject to 3D
    points_3d_homogeneous = P_inv.dot(image_points_homogeneous.T).T

    # Convert from homogeneous coordinates to 3D coordinates
    points_3d = points_3d_homogeneous[:, :3] / points_3d_homogeneous[:, 3:]

    # Adjust Z coordinate
    points_3d[:, 2] = Z

    return points_3d


# Create some 2D points (let's say we detected these in an image)
image_points = np.array([
    [100, 100], [200, 100], [200, 200], [100, 200],
    [150, 150], [250, 150], [250, 250], [150, 250]
], dtype=np.float32)

# Reproject to 3D
points_3d = reproject_to_3d(image_points)

# Visualize the results
fig = plt.figure(figsize=(12, 5))

# 2D plot
ax1 = fig.add_subplot(121)
ax1.scatter(image_points[:, 0], image_points[:, 1])
ax1.set_xlim(0, 300)
ax1.set_ylim(300, 0)  # Invert y-axis to match image coordinates
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('2D Image Points')

# 3D plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Reprojected 3D Points')

plt.tight_layout()
plt.show()

# Print the 3D coordinates
print("Reprojected 3D coordinates:")
print(points_3d)