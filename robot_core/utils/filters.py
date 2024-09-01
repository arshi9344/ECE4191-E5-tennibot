
# Used to implement velocity smoothing in DiffDriveRobot.
# Lower tau means less smoothing, faster response to changes.
# Higher tau means more smoothing, slower response to changes.
# We use this instead of ExponentialMovingAverage because LowPassFilter also takes into account the time difference between updates.

class LowPassFilter:
    def __init__(self, tau):
        self.tau = tau
        self.previous_value = None
        self._last_smoothed_value = None

    def update(self, new_value, dt):
        if self.previous_value is None:
            self.previous_value = new_value
            self._last_smoothed_value = new_value
        else:
            alpha = dt / (self.tau + dt)
            self._last_smoothed_value = alpha * new_value + (1 - alpha) * self.previous_value
            self.previous_value = self._last_smoothed_value
        return self._last_smoothed_value

    @property
    def value(self):
        if self._last_smoothed_value is None:
            return 0
        return self._last_smoothed_value


# Used to implement smoothing dt.
# Lower alpha means less smoothing, faster response to changes
# Higher alpha means more smoothing, slower response to changes.
class ExponentialMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = (1 - self.alpha) * new_value + self.alpha * self.value
        return self.value
