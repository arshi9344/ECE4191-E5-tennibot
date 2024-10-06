from enum import Enum, auto

class RobotStates(Enum):
    SEARCH = 1
    COLLECT = 2
    RETURN = 3
    DEPOSIT = 4
    STOP = 5
    ALIGN = 6

class VisionStates(Enum):
    NONE = 0
    DETECT_BALL = 1
    DETECT_BOX = 2


class StateWrapper:
    def __init__(self, manager, stateEnum, initial_state):
        self._value = manager.Value('i', initial_state.value)
        self._stateEnum = stateEnum

    def set(self, state):
        if not isinstance(state, self._stateEnum):
            raise ValueError(f"State must be a {self._stateEnum} Enum")
        self._value.value = state.value

    def get(self):
        return self._stateEnum(self._value.value)