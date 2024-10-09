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
    in_deposition_zone: bool  # if needed

    @property
    def coords(self):
        return self.x, self.y, self.angle
    
# @dataclass
# class DetectionResult:
#     box_detection: Optional[List[BoxDetection]]  # None if no box is detected
#     ball_detection: Optional[List[BallDetection]]  # None if no tennis ball is detected
#     frame: Any  # The frame with bounding boxes and annotations