from dataclasses import dataclass


@dataclass
class PositionData:
    x: float
    y: float
    th: float
    is_ball: bool

    @property
    def coords(self):
        return self.x, self.y, self.th


