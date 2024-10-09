
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
