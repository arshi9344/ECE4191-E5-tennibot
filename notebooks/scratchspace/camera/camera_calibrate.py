import numpy as np
import cv2 as cv
import glob

# Define the dimensions of the checkerboard (number of inner corners per a chessboard row and column)
CHECKERBOARD = (9, 6)  # Update this to match your checkerboard
square_size = 0.023  # Update this to the actual square size in meters

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.00001)

# Prepare object points based on the size of the checkerboard squares
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane
valid_images = []

# Load calibration images (update path to your images)
images = glob.glob('images/images_calib7/*.jpg')
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
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print("Camera calibration completed.")

# Threshold for acceptable reprojection error (in pixels)
REPROJ_ERROR_THRESHOLD = 1.0  # Adjust this value as needed

# Calculate reprojection error and filter bad images
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

# Recalibrate with valid images
if valid_objpoints and valid_imgpoints:
    mean_error /= len(valid_objpoints)
    print(f"{len(valid_objpoints)} out of {len(images)} passed the reprojection error threshold.")
    print("Recalibrating camera with valid images...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        valid_objpoints, valid_imgpoints, gray.shape[::-1], None, None
    )
    print("Recalibration completed. Saving parameters...")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs}")
    # Save calibration data
    np.savez('camera_calib7.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
    print("Calibration data saved.")


    # Calculate total reprojection error after recalibration
    mean_error = 0
    for i in range(len(valid_objpoints)):
        imgpoints2, _ = cv.projectPoints(
            valid_objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv.norm(valid_imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(valid_objpoints)
    print(f"Total reprojection error after recalibration: {mean_error}")
else:
    print("No valid images passed the reprojection error threshold.")


"""
    
# Load an image to undistort
test_img = cv.imread('path_to_test_image.jpg')
h, w = test_img.shape[:2]
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

# Undistort the image
undistorted_img = cv.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Display the original and undistorted images
cv.imshow('Original Image', test_img)
cv.imshow('Undistorted Image', undistorted_img)
cv.waitKey(0)
cv.destroyAllWindows()

"""