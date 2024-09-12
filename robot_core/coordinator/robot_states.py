from enum import Enum, auto

class RobotStates(Enum):
    DRIVE = 1
    STOP = 2
    COLLECT = 3
    DEPOSIT = 4

class StateWrapper:
    def __init__(self, manager, initial_state):
        self._value = manager.Value('i', initial_state.value)

    def set(self, state):
        if not isinstance(state, RobotStates):
            raise ValueError("State must be a RobotStates Enum")
        self._value.value = state.value

    def get(self):
        return RobotStates(self._value.value)