import time
import numpy as np
import math
from scipy.spatial import KDTree
from typing import List, Optional, Any, Tuple
from robot_core.utils.position import Position, PositionTypes
from robot_core.orchestration.scan_point_utils import ScanPoint, ScanPointGenerator
from robot_core.perception.detection_results import BallDetection, BoxDetection
from robot_core.hardware.dimensions import COURT_XLIM, COURT_YLIM
from robot_core.coordinator.commands import RobotCommands, StateWrapper, VisionCommands
from robot_core.utils.command_utils import StatefulCommandQueue, Command, CommandStatus


""""
-  add_ball_coords(ball_detection_results: List[BallDetection]): Take in x y coords of a ball, and then update the occupancy map in whatever way (add, or refine existing, or do nothing if outside quadrant bounds)
    - ball_detection_results is a list of BallDetection objects (from perception/detection_results.py)
    
- remove_ball(): remove a ball from the occupancy map. Potentially you could give each ball a unique id so you only have to supply the id, and not the x,y coords to remove it (so that you can remove a ball that has moved)
- get_nearest_ball(): returns the nearest ball to the robot. x, y
- is_empty(): returns True if there are no balls in the occupancy map


"""

class OccupancyMap:
    def __init__(
            self,
            detection_results_q,
            quadrant_bounds=(0, COURT_XLIM, 0, COURT_YLIM),
            ASSOCIATION_THRESHOLD = 0.4
    ):

        # Shared multiprocessing.manager variables
        self.detection_results_q = detection_results_q  # Shared queue with the latest detection results

        # Constants
        self.quadrant_bounds = quadrant_bounds  # (xmin, xmax, ymin, ymax)
        self.ASSOCIATION_THRESHOLD = ASSOCIATION_THRESHOLD  # Maximum distance to associate a detection with a known ball

        # Our 'occupancy grid' for the balls and other state variables
        self.occupancy_map = []  # List of known balls positions [(x, y), ...]
        self.start_time = time.time()


    def update_ball_detections(self, detections: List[BallDetection]):
        # Update the list of known balls
        for detection in detections:
            # Estimate the global position of the ball
            # Add the ball if it's within bounds and not already in the list
            if self._is_within_bounds(detection):
                self._associate_ball(detection)

    def _associate_ball(self, detection: BallDetection):
        detected_x, detected_y = detection.x, detection.y
        for ball in self.occupancy_map:
            distance = math.hypot(ball['x'] - detected_x, ball['y'] - detected_y)
            if distance < self.ASSOCIATION_THRESHOLD:
                # Update the existing ball position
                ball['x'] = detected_x
                ball['y'] = detected_y
                return
        # If no existing ball is close enough, add as a new ball
        self.occupancy_map.append({'x': detected_x, 'y': detected_y})

    def remove_collected_ball(self, collected_x, collected_y):
        for ball in self.occupancy_map:
            distance = math.hypot(ball['x'] - collected_x, ball['y'] - collected_y)
            if distance < self.ASSOCIATION_THRESHOLD:
                self.occupancy_map.remove(ball)
                break
    #
    # def _estimate_ball_global_position(self, detection: BallDetection):
    #
    #     # Estimate the global position using the robot's current pose and the detection's relative position
    #     robot_x = self.robot_pose['x']
    #     robot_y = self.robot_pose['y']
    #     robot_th = self.robot_pose['th']
    #
    #     dx = detection.x
    #     dy = detection.y
    #     cos_th = np.cos(robot_th)
    #     sin_th = np.sin(robot_th)
    #     gx = robot_x + dx * cos_th - dy * sin_th
    #     gy = robot_y + dx * sin_th + dy * cos_th
    #     return gx, gy

    def _is_within_bounds(self, detection: BallDetection):
        x, y = detection.x, detection.y
        xmin, xmax, ymin, ymax = self.quadrant_bounds
        return xmin <= x <= xmax and ymin <= y <= ymax


    #
    # def _angle_between_points(self, p1, p2):
    #     dx = p2[0] - p1[0]
    #     dy = p2[1] - p1[1]
    #     return np.arctan2(dy, dx)

    def find_nearest_ball(self, robot_x, robot_y) -> Tuple[float, float]:
        # Find the nearest ball position to the robot
        distances = [np.hypot(ball[0] - robot_x, ball[1] - robot_y) for ball in self.occupancy_map]
        nearest_ball = self.occupancy_map[np.argmin(distances)]
        return nearest_ball # x, y

