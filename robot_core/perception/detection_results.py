from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class BallDetection:
    x: float # +X is in FRONT
    y: float # +Y is to the RIGHT
    angle: float # Radians, anticlockwise is positive, angle from X axis
    total_distance: float
    confidence: float
    in_collection_zone: bool

    @property
    def coords(self):
        return self.x, self.y, self.angle


@dataclass
class BoxDetection:
    x: float
    y: float
    angle: float
    total_distance: float
    confidence: float
    in_collection_zone: bool

    @property
    def coords(self):
        return self.x, self.y, self.angle

@dataclass
class DetectionResult:
    box_detection: Optional[List[BoxDetection]]  # if no box detected, keep it as None
    ball_detection: Optional[List[BallDetection]]  # If no ball detected, keep it as None
    frame: List[Any]  # The frame that we passed to tennis_YOLO, but with bounding boxes, labels, etc.