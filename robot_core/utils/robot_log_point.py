from dataclasses import dataclass
from typing import Tuple

@dataclass
class RobotLogPoint:
    pose: Tuple[float, float, float]
    current_wheel_w: Tuple[float, float]
    target_wheel_w: Tuple[float, float]
    duty_cycle_commands: Tuple[float, float]
    goal_position: Tuple[float, float, float]
    time: float
