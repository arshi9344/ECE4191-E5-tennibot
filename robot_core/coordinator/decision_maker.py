import time
import numpy as np
import math
from scipy.spatial import KDTree
from typing import List, Optional, Any, Tuple
from robot_core.utils.position import Position, PositionTypes
from robot_core.orchestration.scan_point_utils import ScanPoint, ScanPointGenerator
from robot_core.perception.detection_results import BallDetection, BoxDetection, DetectionResult
from robot_core.hardware.dimensions import COURT_XLIM, COURT_YLIM
from robot_core.coordinator.robot_states import RobotStates, StateWrapper, VisionStates
from robot_core.utils.command_utils import StatefulCommandQueue, Command, CommandStatus

class DecisionMaker:
    def __init__(
            self,
            robot_pose,
            goal_position,
            detection_results_q,
            shared_data,
            collection_box_location,
            quadrant_bounds=(0, COURT_XLIM, 0, COURT_YLIM),
            max_capacity=5,
            deposit_time=8*60,
            ASSOCIATION_THRESHOLD = 0.4
    ):

        # Shared multiprocessing.manager variables
        self.robot_pose = robot_pose  # Shared dict with robot's current pose {'x': x, 'y': y, 'th': th}
        self.goal_position = goal_position  # Shared dict for the goal position
        self.detection_results_q = detection_results_q  # Shared queue with the latest detection results
        self.shared_data = shared_data # Shared dict with the latest detection results
        self.command_queue = shared_data['command_queue']  # Shared command queue

        # Constants
        self.max_capacity = max_capacity  # Maximum number of balls the robot can carry
        self.quadrant_bounds = quadrant_bounds  # (xmin, xmax, ymin, ymax)
        self.collection_box_location = collection_box_location  # (x, y) # not used for now
        self.ASSOCIATION_THRESHOLD = ASSOCIATION_THRESHOLD  # Maximum distance to associate a detection with a known ball

        # Our 'occupancy grid' for the balls and other state variables
        self.occupancy_map = []  # List of known balls positions [(x, y), ...]
        self.collected_balls = 0  # Number of balls currently collected
        self.last_issued_command_id = None
        # F
        # self.state = 'SEARCH'  # Can be 'SEARCH', 'COLLECT', 'DEPOSIT', 'ALIGN'
        # self.current_goal = None  # The current goal position (x, y)
        self.search_points = []  # List of points to search
        self.current_search_index = 0
        self.start_time = time.time()
        self.deposit_time = deposit_time  # Time in seconds after which to deposit
        self.last_deposit_time = self.start_time
        self.initialise_search_points()

    def initialise_search_points(self):
        # These are an example of the points we get from ScanPointGenerator
        # Point: (2.0, -2.0).Limited?: False, Rotation: (0, 0),
        # Point: (4.0, -2.0).Limited?: False, Rotation: (0, 0),
        # Point: (4.0, -4.0).Limited?: False, Rotation: (0, 0),
        # Point: (2.0, -4.0).Limited?: False, Rotation: (0, 0)]
        max_scan_distance = 2
        flip_x = False
        flip_y = True
        # Generate scan points and lines
        scan_gen = ScanPointGenerator(scan_radius=max_scan_distance, flip_x=flip_x, flip_y=flip_y)
        self.search_points = scan_gen.points


    def update_ball_detections(self, detections: List[BallDetection]):
        # Update the list of known balls
        for detection in detections:
            # Estimate the global position of the ball
            ball_global_position = self._estimate_ball_global_position(detection)
            # Add the ball if it's within bounds and not already in the list
            if self._is_within_bounds(*ball_global_position):
                self._associate_ball(*ball_global_position)

    def _associate_ball(self, detected_x, detected_y):
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

    def _estimate_ball_global_position(self, detection: BallDetection):

        # Estimate the global position using the robot's current pose and the detection's relative position
        robot_x = self.robot_pose['x']
        robot_y = self.robot_pose['y']
        robot_th = self.robot_pose['th']

        dx = detection.x
        dy = detection.y
        cos_th = np.cos(robot_th)
        sin_th = np.sin(robot_th)
        gx = robot_x + dx * cos_th - dy * sin_th
        gy = robot_y + dx * sin_th + dy * cos_th
        return gx, gy

    def _is_within_bounds(self, x, y):
        xmin, xmax, ymin, ymax = self.quadrant_bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def decide_next_action(self):
        # Decide the robot's next action and update the goal position
        current_time = time.time()
        time_since_last_deposit = current_time - self.last_deposit_time

        # Get the current command state

        # STOP = 0
        # SEARCH = 1  # essentially the same as drive
        # COLLECT = 2  #
        # DEPOSIT = 3
        # ALIGN = 5

        ### RETURN
        if self.collected_balls >= self.max_capacity or time_since_last_deposit >= self.deposit_time:
            # Need to return home to deposit balls
            self._issue_command(RobotStates.DEPOSIT)
            self._update_goal_position(
                Position(self.collection_box_location[0], self.collection_box_location[1], 0, PositionTypes.BOX)
            )
            # Check if within threshold of box, if so, deposit balls

            # Check computer vision data, see if box is detected, and navigate towards it
            # need to insert additional logic here to detect box, navigate towards it, line up, and open teh door

        ### DEPOSIT
        # Same if condition as above, but check if the robot is aligned against the box
        # if robot is at the deposit location:
        #     deposit balls
        #     self.collected_balls = 0
        #     self.last_deposit_time = time.time()
        #     self.shared_data['robot_state'].set(RobotStates.SEARCH)
        #     self.current_goal = self.next_search_position()
        #     self.update_goal_position(self.current_goal)
        # else:
        #     self.shared_data['robot_state'].set(RobotStates.RETURN)
        #     self.current_goal = self.collection_box_location
        #     self.update_goal_position(self.current_goal)


        ### COLLECT, Position.type = PositionTypes.BALL
        # KEEP BEING INSIDE COLLECT, UNTIL VISION NO LONGER SEES A BALL IN COLLECTION ZONE / OR BALL IS COLLECTED
        # Orchestrator goes to the goal position, and then stamps once close enough
        elif len(self.occupancy_map) > 0:
            # There are known balls to collect
            nearest_ball = self.find_nearest_ball()
            angle = self._angle_between_points((self.robot_pose['x'], self.robot_pose['y']), nearest_ball)

            self._issue_command(RobotStates.COLLECT)
            self._update_goal_position(
                Position(*nearest_ball, angle, PositionTypes.BALL)
            )

        ### SEARCH, Position.type = PositionTypes.SCAN_POINT
        else:
            # No known balls, continue searching
            # TODO: need to check if previous command is finished or not

            next_scan_point = self.next_search_position()
            angle = self._angle_between_points((self.robot_pose['x'], self.robot_pose['y']), next_scan_point)

            self._issue_command(RobotStates.SEARCH)
            self._update_goal_position(Position(
                *next_scan_point, angle, PositionTypes.SCAN_POINT
            ))

            # rotate once at the scan point location




    def _update_goal_position(self, goal: Position):
        # Update the shared goal_position dict
        self.goal_position.update({
            'goal': goal,
            'time': time.time()
        })

    def _issue_command(self, command_data: RobotStates):
        # Issue a command to the robot
        self.last_issued_command_id = self.command_queue.put(command_data)

    def _angle_between_points(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.arctan2(dy, dx)

    def find_nearest_ball(self) -> Tuple[float, float]:
        # Find the nearest ball position to the robot
        robot_x = self.robot_pose['x']
        robot_y = self.robot_pose['y']
        distances = [np.hypot(ball[0] - robot_x, ball[1] - robot_y) for ball in self.occupancy_map]
        nearest_ball = self.occupancy_map[np.argmin(distances)]
        return nearest_ball # x, y

    def next_search_position(self) -> Tuple[float, float]:
        # Return the next point in the search_points list
        if self.current_search_index >= len(self.search_points):
            self.current_search_index = 0  # Loop back to the beginning
        next_point = self.search_points[self.current_search_index]
        self.current_search_index += 1
        return next_point

    def on_reach_goal(self):
        # Called when the robot reaches its goal
        if self.state == 'COLLECT':
            # Assume the robot has collected the ball
            self.collected_balls += 1
            if self.current_goal in self.ball_positions:
                self.ball_positions.remove(self.current_goal)
        elif self.state == 'DEPOSIT':
            # Assume the robot has deposited the balls
            self.collected_balls = 0
            self.last_deposit_time = time.time()
        elif self.state == 'SEARCH':
            # Reached a search point, do nothing
            pass


    def process(self):
        # Called in each iteration
        detections = self.detection_results_q.get_nowait()
        if detections:
            self.update_ball_detections(detections)
            self.update_box_detection(detections)
        self.decide_next_action()
