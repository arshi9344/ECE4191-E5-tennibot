from dataclasses import dataclass
from enum import Enum

class PositionTypes(Enum):
    ROBOT = 1
    BALL = 2
    BOX = 3
    SCAN_POINT = 4

@dataclass
class Position:
    x: float or None
    y: float or None
    th: float or None
    type: PositionTypes

    @property
    def coords(self):
        return self.x, self.y, self.th

