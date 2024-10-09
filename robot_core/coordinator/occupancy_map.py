
from robot_core.perception.detection_results import BallDetection, BoxDetection
from robot_core.hardware.dimensions import COURT_XLIM, COURT_YLIM
from robot_core.coordinator.commands import RobotCommands, StateWrapper, VisionCommands
from robot_core.utils.command_utils import StatefulCommandQueue, Command, CommandStatus

from typing import List, Tuple, Optional


# class Ball:
#     def __init__(self, ball_id: int, x: float, y: float, confidence: float):
#         self.ball_id = ball_id
#         self.x = x
#         self.y = y
#         self.confidence = confidence
#
#     def update(self, detection: BallDetection):
#         total_confidence = self.confidence + detection.confidence
#         self.x = (self.x * self.confidence + detection.x * detection.confidence) / total_confidence
#         self.y = (self.y * self.confidence + detection.y * detection.confidence) / total_confidence
#         self.confidence = total_confidence
#

class Ball:
    def __init__(self, ball_id: int, detection: BallDetection):
        self.ball_id = ball_id
        self.x = detection.x
        self.y = detection.y

    def update(self, detection: BallDetection):
        # TODO: Add moving average using existing functions
        self.x = detection.x
        self.y = detection.y

    def __repr__(self):
        return f"Ball {self.ball_id}: ({self.x}, {self.y})"


class OccupancyMap:
    def __init__(self,
                 quadrant_bounds: Tuple[float, float, float, float] = (0, COURT_XLIM, 0, COURT_YLIM),
                 matching_threshold: float = 0.15, # in meters
                 confidence_threshold: float = 0.7  # 0 to 1
        ):
        self.quadrant_bounds = quadrant_bounds  # (xmin, xmax, ymin, ymax)
        self.balls = {}  # ball_id -> Ball
        self.next_ball_id = 1
        self.matching_threshold = matching_threshold
        self.confidence_threshold = confidence_threshold

    def _is_within_bounds(self, detection: BallDetection):
        x, y = detection.x, detection.y
        xmin, xmax, ymin, ymax = self.quadrant_bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def _satisfies_confidence_threshold(self, detection: BallDetection):
        return detection.confidence >= self.confidence_threshold

    def update(self, detections: List[BallDetection]):
        for detection in detections:
            if not self._is_within_bounds(detection) or not self._satisfies_confidence_threshold(detection):
                continue  # Skip detections outside the bounds
            matched = False
            for ball in self.balls.values():
                distance = ((ball.x - detection.x) ** 2 + (ball.y - detection.y) ** 2) ** 0.5
                if distance < self.matching_threshold:
                    ball.update(detection)
                    matched = True
                    break
            if not matched:
                ball = Ball(self.next_ball_id, detection)
                self.balls[self.next_ball_id] = ball
                self.next_ball_id += 1

    def get_closest_ball(self, robot_x: float, robot_y: float, robot_theta: float) -> Optional[Tuple[float, float, int]]:
        if not self.balls:
            return None  # No balls in the map
        closest_ball = None
        min_distance = float('inf')
        for ball in self.balls.values():
            distance = ((ball.x - robot_x) ** 2 + (ball.y - robot_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_ball = ball
        if closest_ball:
            return closest_ball.x, closest_ball.y, closest_ball.ball_id
        else:
            return None

    def remove_ball(self, ball_id: int = None, coords: Tuple[float, float] = None):
        if ball_id is not None:
            if ball_id in self.balls:
                del self.balls[ball_id]
        elif coords is not None:
            x, y = coords
            for id_, ball in list(self.balls.items()):
                distance = ((ball.x - x) ** 2 + (ball.y - y) ** 2) ** 0.5
                if distance < self.matching_threshold:
                    del self.balls[id_]
                    break

    def is_empty(self) -> bool:
        return len(self.balls) == 0

    def __repr__(self):
        if self.is_empty():
            return "OccupancyMap: Empty."
        output_str = "OccupancyMap: "
        for ball in self.balls.values():
            output_str += f"{ball}, "
        return output_str


if __name__ == '__main__':
    # A better example of test usage is in decision_maker.py, where the test code uses an OccupancyMap to track tennis balls.
    # Test OccupancyMap
    detections=[
        BallDetection(x=0.5, y=0.5, angle=0, total_distance=0.5, confidence=0.8, in_collection_zone=False),
        BallDetection(x=0.6, y=0.6, angle=0, total_distance=0.6, confidence=0.9, in_collection_zone=False),
        BallDetection(x=1.5, y=1.5, angle=0, total_distance=1.5, confidence=0.7, in_collection_zone=False),
        BallDetection(x=0.5, y=0.4, angle=0, total_distance=0.5, confidence=0.8, in_collection_zone=False),
        BallDetection(x=0.6, y=0.7, angle=0, total_distance=0.6, confidence=0.9, in_collection_zone=False),
        BallDetection(x=1.5, y=1.5, angle=0, total_distance=1.5, confidence=0.7, in_collection_zone=False),
        BallDetection(x=0.5, y=0.5, angle=0, total_distance=0.5, confidence=0.8, in_collection_zone=False),
        BallDetection(x=0.6, y=0.6, angle=0, total_distance=0.6, confidence=0.9, in_collection_zone=False),
        BallDetection(x=1.4, y=1.5, angle=0, total_distance=1.5, confidence=0.7, in_collection_zone=False),
        BallDetection(x=0.4, y=0.5, angle=0, total_distance=0.5, confidence=0.8, in_collection_zone=False),
        BallDetection(x=0.6, y=0.6, angle=0, total_distance=0.6, confidence=0.9, in_collection_zone=False),
        BallDetection(x=1.5, y=1.5, angle=0, total_distance=1.5, confidence=0.7, in_collection_zone=False)
    ]

    # Define the quadrant bounds (xmin, xmax, ymin, ymax)
    quadrant_bounds = (0.0, 5.0, 0.0, 4.0)  # Example bounds
    occupancy_map = OccupancyMap(quadrant_bounds)   # Create an instance of OccupancyMap


    # Update the occupancy map with detections
    occupancy_map.update(detections)

    # Get the closest ball to the robot at position (1.0, 1.0)
    closest_ball = occupancy_map.get_closest_ball(robot_x=1.0, robot_y=1.0, robot_theta=0.0)
    print(f"Closest ball: {closest_ball}")  # Outputs: Closest ball: (2.05, 1.05, 1)

    # Remove a ball by ID
    occupancy_map.remove_ball(ball_id=1)

    # Check if the occupancy map is empty
    print(f"Is the occupancy map empty? {occupancy_map.is_empty()}")  # Outputs: False
