from dataclasses import dataclass

@dataclass
class BallDetection:
    x: float # +X is in FRONT
    y: float # +Y is to the RIGHT
    angle: float # Radians, anticlockwise is positive, angle from X axis
    total_distance: float
    confidence: float
    frame: [] # The frame that we passed to tennis_YOLO, but with bounding boxes, labels, etc.
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
    frame: []
    in_collection_zone: bool

    @property
    def coords(self):
        return self.x, self.y, self.angle

@dataclass
class DetectionResult:
    box_detection: None or [BoxDetection] # if no box detected, keep it as None
    ball_detection: None or [BallDetection] # If no ball detected, keep it as None