from dataclasses import dataclass
from enum import Enum

class PositionTypes(Enum):
    ROBOT = 1
    BALL = 2
    BOX = 3
    SCAN_POINT = 4

@dataclass
class Position:
    x: float
    y: float
    th: float
    type: PositionTypes

    @property
    def coords(self):
        return self.x, self.y, self.th

