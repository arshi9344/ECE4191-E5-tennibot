import numpy as np
import cv2 as cv
import glob

"""
This file is used to calibrate the camera using a set of images of a checkerboard pattern.
It saves a calibration matrix to the same folder as this script.

NOTE that I've renamed the camera calibration matrices with a number like 2_3 or 2_7 to indicate the reprojection error, e.g. 2_3 for 2.3 and 2_7 for 2.7
"""

# Define the dimensions of the checkerboard (inner corners)
CHECKERBOARD = (6, 9)  # Adjust based on your checkerboard
#square_size = 0.022  # Square size in meters (25mm)
square_size = 0.0234 
# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 0.001)

# Prepare object points based on the size of the checkerboard squares
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane
valid_images=[]


# Load calibration images (update path to your images)
images = glob.glob('images_calib/*.jpg')
print(f"Found {len(images)} images.")


for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Image {fname} could not be loaded. Skipping...")
        continue  # Skip if the image cannot be loaded

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        print(f"Corners detected in {fname}")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        valid_images.append(fname)  # Add image to valid list

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(5)
    else:
        print(f"No corners found in {fname}. Skipping...")

cv.destroyAllWindows()

print("Calibrating camera...")
# Perform camera calibration to get the camera matrix and distortion coefficients
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera calibration completed. Saving parameters...  ")

# Save calibration data
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

print("Calibration data saved.")
# Display camera matrix and distortion coefficients
print("Camera Matrix: \n", camera_matrix)
print("Distortion Coefficients: \n", dist_coeffs)

# Threshold for acceptable reprojection error (in pixels)
REPROJ_ERROR_THRESHOLD = 3.0

# After the camera calibration step, calculate reprojection error and filter bad images
mean_error = 0
valid_objpoints = []
valid_imgpoints = []
valid_img_names = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)

    if error < REPROJ_ERROR_THRESHOLD:
        print(f"Image {valid_images[i]} - Reprojection error: {error}")
        valid_img_names.append(valid_images[i])
        valid_objpoints.append(objpoints[i])
        valid_imgpoints.append(imgpoints[i])
        mean_error += error
    else:
        print(f"Image {valid_images[i]} rejected due to high reprojection error: {error}")

# Recalculate the reprojection error with only the valid images
if valid_objpoints and valid_imgpoints:
    mean_error /= len(valid_objpoints)
    print(f"{len(valid_objpoints)} out of {len(images)} passed the reprojection error threshold.")
    print(valid_img_names)
    print(f"Total reprojection error (valid images): {mean_error}")
else:
    print("No valid images passed the reprojection error threshold.")