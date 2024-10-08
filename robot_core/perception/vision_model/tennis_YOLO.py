import cv2
import numpy as np
import torch
import math
import pathlib
import os

from IPython.display import display as ipy_display, clear_output
import matplotlib.pyplot as plt
import time
import sys
import logging
from ultralytics import YOLO


from robot_core.perception.detection_results import BallDetection, BoxDetection, DetectionResult

# Define real tennis ball radius in meters (3.25 cm radius)
MODEL_PATH = 'box_tennis.pt'
CALIB_MATRIX_PATH = 'calib6_matrix.npz'
current_dir = os.path.dirname(os.path.abspath(__file__))
calibration_data_path = os.path.join(current_dir, CALIB_MATRIX_PATH)
model_path = os.path.join(current_dir, MODEL_PATH)

calibration_data = np.load(calibration_data_path)
model = YOLO(model_path)


class ObjectDetection:
    def __init__(self,
                 model=model,
                 calibration_data=calibration_data,
                 camera_height=0,
                 verbose=False,
        ):
        self.camera_matrix = calibration_data['camera_matrix']
        self.distortion_coeffs = calibration_data['dist_coeffs']
        self.model = model
        self.camera_height = camera_height
        self.verbose = verbose

        self.class_id_to_name = self.model.names  # Mapping class IDs to names
        self.logger = logging.getLogger(__name__)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Undistorts the image using the camera matrix and distortion coefficients."""
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs)
        return frame

    def calculate_distance(self, u: float, v: float, r_px: float, object_radius: float):
        """Calculates horizontal and total distance to the object."""
        if r_px == 0:
            self.logger.error("Detected radius is zero, cannot calculate distance.")
            return None, None, None

        fx = self.camera_matrix[0, 0]  # Focal length in x direction
        cx = self.camera_matrix[0, 2]  # Principal point x-coordinate

        total_distance = fx * object_radius / r_px

        if self.camera_height > 0:
            horizontal_distance = math.sqrt(max(total_distance ** 2 - self.camera_height ** 2, 0))
        else:
            horizontal_distance = total_distance

        y = (u - cx) * (object_radius / r_px)

        angle = math.atan2(y, horizontal_distance) if horizontal_distance != 0 else 0
        return horizontal_distance, total_distance, angle

    def polar_to_cartesian(self, horizontal_distance: float, angle: float) -> tuple:
        """Converts polar coordinates (horizontal distance, angle) to Cartesian coordinates (x, y)."""
        x = horizontal_distance * math.cos(angle)
        y = horizontal_distance * math.sin(angle)
        return x, y

    def is_in_zone(self, u: float, v: float, zone: tuple) -> bool:
        """Check if a detection is within the defined zone (e.g., collection zone or deposition zone)."""
        x_min, y_min, x_max, y_max = zone
        return x_min <= u <= x_max and y_min <= v <= y_max

    def get_object_radius(self, class_name: str):
        """To be implemented by subclasses to provide object-specific radii."""
        raise NotImplementedError("Subclasses must implement get_object_radius method.")

    def create_detection_instance(self, class_name: str, x: float, y: float, angle: float, total_distance: float,
                                  confidence: float, in_zone: bool):
        """To be implemented by subclasses to create appropriate detection instances."""
        raise NotImplementedError("Subclasses must implement create_detection_instance method.")

    def get_zone(self, class_name: str):
        """To be implemented by subclasses to provide object-specific zones."""
        raise NotImplementedError("Subclasses must implement get_zone method.")

    def detect(self, frame: np.ndarray, draw_zones=True):
        if frame is None:
            self.logger.error("Frame is None")
            return [], frame

        undistorted_frame = self.undistort(frame)
        results = self.model(undistorted_frame)  # Run inference

        detections = []

        # Process each result (batch of images)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            boxes_data = boxes.data.cpu().numpy()  # Each box: [x1, y1, x2, y2, confidence, class_id]
            for box in boxes_data:
                x1, y1, x2, y2, confidence, class_id = box
                class_id = int(class_id)
                class_name = self.class_id_to_name.get(class_id, 'unknown')

                # Calculate center (u, v) and radius in pixels
                u = (x1 + x2) / 2
                v = (y1 + y2) / 2
                r_px = (x2 - x1) / 2  # Approximate radius

                object_radius = self.get_object_radius(class_name)
                if object_radius is None:
                    # Skip objects that we don't have radius information for
                    continue

                horizontal_distance, total_distance, angle = self.calculate_distance(u, v, r_px, object_radius)
                x, y = self.polar_to_cartesian(horizontal_distance, angle)

                in_zone = self.is_in_zone(u, v, self.get_zone(class_name))

                # Draw bounding box and labels on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detection_instance = self.create_detection_instance(class_name, x, y, angle, total_distance, confidence, in_zone)
                if detection_instance:
                    detections.append(detection_instance)

        if draw_zones:
            self.draw_zones(frame)

        if self.verbose:
            self.print_detections(detections)

        return detections, frame

    def draw_zones(self, frame: np.ndarray):
        """To be implemented by subclasses if they have zones to draw."""
        pass

    def print_detections(self, detections: list):
        """Prints detailed information about detections."""
        for detection in detections:
            print(f"{detection}")


class TennisBallDetector(ObjectDetection):
    TENNIS_BALL_RADIUS_M = 0.0325

    def __init__(self, collection_zone, TENNIS_BALL_RADIUS_M=0.0325):
        self.collection_zone = collection_zone
        self.TENNIS_BALL_RADIUS_M = TENNIS_BALL_RADIUS_M
        super().__init__()

    def get_object_radius(self, class_name: str):
        return self.TENNIS_BALL_RADIUS_M if class_name == 'tennis-ball' else None

    def get_zone(self, class_name: str):
        return self.collection_zone if class_name == 'tennis-ball' else None

    def create_detection_instance(self, class_name: str, x: float, y: float, angle: float, total_distance: float,
                                  confidence: float, in_zone: bool):
        if class_name != 'tennis-ball':
            return None
        return BallDetection(
            x=x, y=y, angle=angle, total_distance=total_distance, confidence=confidence, in_collection_zone=in_zone
        )

    def draw_zones(self, frame: np.ndarray):
        x_min, y_min, x_max, y_max = self.collection_zone
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)


class BoxDetector(ObjectDetection):

    def __init__(self, deposition_zone, BOX_SIZE_M=0.16):
        self.BOX_SIZE_M = BOX_SIZE_M  # assume 16 cm height for the box
        self.deposition_zone = deposition_zone

        super().__init__()

    def get_object_radius(self, class_name: str):
        return self.BOX_SIZE_M / 2 if class_name == 'box' else None

    def get_zone(self, class_name: str):
        return self.deposition_zone if class_name == 'box' else None

    def create_detection_instance(self, class_name: str, x: float, y: float, angle: float, total_distance: float,
                                  confidence: float, in_zone: bool):
        if class_name != 'box':
            return None
        return BoxDetection(
            x=x, y=y, angle=angle, total_distance=total_distance, confidence=confidence, in_deposition_zone=in_zone
        )

    def draw_zones(self, frame: np.ndarray):
        x_min, y_min, x_max, y_max = self.deposition_zone
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)


