import cv2
import numpy as np
import torch
import math
import pathlib
from pathlib import Path
import os

from IPython.display import display as ipy_display, clear_output
import matplotlib.pyplot as plt
import time
import sys
import logging

import sys
sys.path.insert(0, '../vision_model')

from robot_core.perception.detection_results import BallDetection, BoxDetection, DetectionResult

# Define real tennis ball radius in meters (3.25 cm radius)
MODEL_PATH = 'best.pt'
CALIB_MATRIX_PATH = 'calib6_matrix.npz'


class TennisBallDetectorHeight:
    """Detect tennis balls and calculate distances and angles relative to the camera.

    This class uses a YOLO model for detecting tennis balls and calculates the horizontal
    distance, total distance, and angle relative to the camera for robot steering purposes.
    Optionally, frames can be shown, and verbose output can be enabled.
    """

    def __init__(self,
                 model_path=MODEL_PATH,
                 calibration_data_path=CALIB_MATRIX_PATH,
                 collection_zone=(200, 150, 400, 350),
                 camera_height=0,
                 cache=True,
                 windows=False,
                 verbose=False,
                 TENNIS_BALL_RADIUS_M = 0.0325

):
        """Initializes the TennisBallDetector.

        Args:
            model_path (str): Path to the trained YOLO model.
            camera_matrix (np.ndarray, optional): Camera matrix from calibration. Default is None.
            distortion_coeffs (np.ndarray, optional): Distortion coefficients from calibration. Default is None.
            collection_zone (tuple, optional): A tuple defining the region for collection (x_min, y_min, x_max, y_max). Default is None.
            camera_height (float, optional): Height of the camera from the ground (in meters). Default is 0.
            cache (bool, optional): If True, cache the YOLO model. Default is True.
            windows (bool, optional): If True, adapts for Windows OS path handling. Default is False.
            verbose (bool, optional): If True, print additional details during detection (e.g., vertical distance). Default is False.
        """
        if windows:
            pathlib.PosixPath = pathlib.WindowsPath

        # Set the path to the model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        calibration_data_path = os.path.join(current_dir, calibration_data_path)
        model_path = os.path.join(current_dir, model_path)

        # # Load YOLO model based on caching preference
        # # Load the YOLO model and calibration data
        # current_dir = Path(__file__).parent  # Get the directory of the current script
        # # Construct the path to the .npz file
        # custom_cache_dir = str(current_dir / 'cache')
        # # Set the custom cache directory
        # os.environ['TORCH_HOME'] = custom_cache_dir



        if cache:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

        calibration_data = np.load(calibration_data_path)
        self.camera_matrix = calibration_data['camera_matrix']
        self.distortion_coeffs = calibration_data['dist_coeffs']
        self.collection_zone = collection_zone
        self.camera_height = camera_height
        self.verbose = verbose
        self.TENNIS_BALL_RADIUS_M = TENNIS_BALL_RADIUS_M

    def undistort(self, frame):
        """Undistorts the image using the camera matrix and distortion coefficients.

        Args:
            frame (np.ndarray): The distorted input frame.

        Returns:
            np.ndarray: The undistorted frame.
        """
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs)
        return frame

    def calculate_distance(self, u, v, r_px):
        """Calculates the total and horizontal distance to the tennis ball, as well as the angle.

        Args:
            u (float): The x-coordinate of the detected object (in pixels).
            v (float): The y-coordinate of the detected object (in pixels).
            r_px (float): The detected radius of the object (in pixels).

        Returns:
            tuple: A tuple containing:
                - horizontal_distance (float): The straight-line distance along the x-y plane (ignoring z-height) to the ball in meters.
                - total_distance (float): The total straight-line distance along the x-y-z plane to the ball in meters.
                - angle (float): The angle to the ball along the x-y plane (ignoring z-height) in radians.
        """
        if r_px == 0:
            print("Error: Detected radius is zero, cannot calculate distance.")
            return None, None, None

        fx = self.camera_matrix[0, 0]  # Focal length in x direction
        cx = self.camera_matrix[0, 2]  # Principal point x-coordinate

        total_distance = fx * self.TENNIS_BALL_RADIUS_M / r_px

        if self.camera_height > 0:
            horizontal_distance = math.sqrt(max(total_distance ** 2 - self.camera_height ** 2, 0))
        else:
            horizontal_distance = total_distance

        y = (u - cx) * (self.TENNIS_BALL_RADIUS_M / r_px)

        angle = math.asin(y / horizontal_distance) if horizontal_distance != 0 else 0

        # Calculate the x and y position of the ball relative to the camera in meters, with 0,0 being the camera position

        return horizontal_distance, total_distance, angle

    def polar_to_cartesian(self, horizontal_distance, angle):
        """Converts polar coordinates (horizontal distance, angle) to Cartesian coordinates (x, y).

        Args:
            horizontal_distance (float): The horizontal distance from the camera to the object in meters.
            angle (float): The angle to the object in radians.

        Returns:
            tuple: A tuple containing the x and y coordinates in meters.
        """
        x = horizontal_distance * math.cos(angle)
        y = horizontal_distance * math.sin(angle)
        return x, y

    def is_in_collection_zone(self, u, v):
        """Checks if the given coordinates are within the collection zone.

        Args:
            u (float): The x-coordinate in pixels.
            v (float): The y-coordinate in pixels.

        Returns:
            bool: True if within the collection zone, False otherwise.
        """
        if self.collection_zone:
            x_min, y_min, x_max, y_max = self.collection_zone
            return x_min <= u <= x_max and y_min <= v <= y_max
        return False

    def detect(self, frame, draw_collection_zone=True) -> DetectionResult:
        """Detects the tennis ball and returns relevant distance and angle information using a single frame.
        Args:
            frame: The frame to detect the tennis ball in. Needs to be RGB colour space (not BGR).
            draw_collection_zone (bool, optional): If True, draw the collection zone on the frame. Default is True.
        Returns:
            dict: A dictionary containing:
                - 'horizontal_distance' (float): The horizontal distance to the ball in meters.
                - 'total_distance' (float, optional): The total distance to the ball in meters (only when verbose=True).
                - 'angle' (float, optional): The horizontal angle to the ball in radians (only when verbose=True).
                - 'cartesian_coords' (tuple): Cartesian coordinates (x, y) based on the horizontal distance and angle.
                - 'in_collection_zone' (bool): True if the ball is within the collection zone, False otherwise.
        """
        # Base detection result with values needed for both verbose and non-verbose modes
        detection_result = DetectionResult(
            box_detection=None,
            ball_detection=None
        )
        if frame is None:
            print(f"Error: Frame is None")
            return detection_result

        undistorted_frame = self.undistort(frame)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*")
            results = self.model(undistorted_frame)  # Run inference
        detections = results.xyxy[0].cpu().numpy()  # Get detections in numpy format

        if len(detections) > 0:
            # TODO: Return results for all the balls that are detected, not just the first (biggest) one in the list
            x1, y1, x2, y2, confidence, class_id = detections[0]

            # Calculate center (u, v) and radius in pixels
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2
            r_px = (x2 - x1) / 2

            horizontal_distance, total_distance, angle = self.calculate_distance(u, v, r_px)

            # Convert to Cartesian coordinates (x, y) based on horizontal distance and angle
            x, y = self.polar_to_cartesian(horizontal_distance, angle)

            # Add tennis ball bounding box and labels to frame
            for index, obj in results.pandas().xyxy[0].iterrows():
                x_min, y_min = int(obj['xmin']), int(obj['ymin'])
                x_max, y_max = int(obj['xmax']), int(obj['ymax'])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"{obj['name']} {obj['confidence']:.2f}"
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if draw_collection_zone:
                x_min, y_min, x_max, y_max = self.collection_zone
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

            ball_result = BallDetection(
                x=x,
                y=y,
                angle=angle,
                total_distance=total_distance,
                confidence=confidence,
                frame=frame,
                in_collection_zone=self.is_in_collection_zone(u, v)

            )
            detection_result.ball_detection = ball_result
            if self.verbose:
                # Verbose output
                print(f"Horizontal Distance: {horizontal_distance:.2f} meters")
                print(f"Total Distance: {total_distance:.2f} meters")
                print(f"Angle: {math.degrees(angle):.2f} degrees")
                print(f"Cartesian Coordinates: (x: {x:.2f} meters, y: {y:.2f} meters)")
                print(f"In Collection Zone: {detection_result['in_collection_zone']}")
                if self.camera_height > 0:
                    print(f"Camera Height: {self.camera_height} meters")
                    print(f"Vertical Distance: {self.camera_height} meters (from floor)")


        else:
            print("No detections found.")

        return detection_result
