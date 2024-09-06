import numpy as np
import cv2 as cv
import glob

# Define the dimensions of the checkerboard (inner corners)
CHECKERBOARD = (7, 10)  # Adjust based on your checkerboard
square_size = 0.022  # Square size in meters (25mm)

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.00001)

# Prepare object points based on the size of the checkerboard squares
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images (update path to your images)
images = glob.glob('images/*.jpg')
print(f"Found {len(images)} images.")
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    print("here's the result: ", ret, corners)
    if ret:
        print(f"Corners detected in {fname}")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(5000)

cv.destroyAllWindows()

# Perform camera calibration to get the camera matrix and distortion coefficients
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration data
np.savez('camera_calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# Display camera matrix and distortion coefficients
print("Camera Matrix: \n", camera_matrix)
print("Distortion Coefficients: \n", dist_coeffs)


# After the camera calibration step
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total reprojection error: {mean_error / len(objpoints)}")
