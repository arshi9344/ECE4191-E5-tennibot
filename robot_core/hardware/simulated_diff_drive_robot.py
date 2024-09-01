"""
This is a simulated DiffDrive robot. We can use this to test our higher level code without having to run it on the real robot.
"""
import numpy as np

from robot_core.hardware.dimensions import WHEEL_RADIUS, WHEEL_SEP

class DiffDriveRobot:

    def __init__(self, inertia=5, dt=0.1, drag=0.2, wheel_radius=WHEEL_RADIUS, wheel_sep=WHEEL_SEP):

        self.x = 0.0  # y-position
        self.y = 0.0  # y-position
        self.th = 0.0  # orientation

        self.wl = 0.0  # rotational velocity left wheel
        self.wr = 0.0  # rotational velocity right wheel

        self.I = inertia
        self.d = drag
        self.dt = dt

        self.r = wheel_radius
        self.l = wheel_sep

    # Should be replaced by motor encoder measurement which measures how fast wheel is turning
    # Here, we simulate the real system and measurement
    def motor_simulator(self, w, duty_cycle):

        torque = self.I * duty_cycle

        if (w > 0):
            w = min(w + self.dt * (torque - self.d * w), 3)
        elif (w < 0):
            w = max(w + self.dt * (torque - self.d * w), -3)
        else:
            w = w + self.dt * (torque)

        return w

    # Veclocity motion model
    def base_velocity(self, wl, wr):

        v = (wl * self.r + wr * self.r) / 2.0

        w = (wl * self.r - wr * self.r) / self.l

        return v, w

    # Kinematic motion model
    def pose_update(self, duty_cycle_l, duty_cycle_r):

        self.wl = self.motor_simulator(self.wl, duty_cycle_l)
        self.wr = self.motor_simulator(self.wr, duty_cycle_r)

        v, w = self.base_velocity(self.wl, self.wr)

        self.x = self.x + self.dt * v * np.cos(self.th)
        self.y = self.y + self.dt * v * np.sin(self.th)
        self.th = self.th + w * self.dt

        return self.x, self.y, self.th

    @property
    def pose(self):
        return self.x, self.y, self.th