import time
import numpy as np

from robot_core.hardware.dimensions import WHEEL_RADIUS, WHEEL_SEP

"""
    This class implements a simple PI controller for a robot. The controller is used to control the speed of the robot's wheels.
    Note that the controller is not used to control the robot's position or orientation - rather, it accepts a desired linear and angular velocity and outputs duty cycles for the left and right wheel motors.
    - The controller uses a proportional-integral (PI) control scheme to adjust the duty cycles based on the difference between the desired and actual wheel speeds.
    - The controller also includes anti-windup logic to prevent the integral term from accumulating too much error.
    - Note the default Kp and Ki arguments.
"""

class PIController:
    def __init__(self, Kp=8, Ki=5, dt=0.1, wheel_radius=WHEEL_RADIUS, wheel_sep=WHEEL_SEP, integral_windup=True, real_time=False):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.dt = dt  # Time step

        self.r = wheel_radius
        self.l = wheel_sep

        self.min_output = -100  # Minimum duty cycle (-100%)
        self.max_output = 100  # Maximum duty cycle (100%)

        self.ml_integral = 0  # Integral term, motor left
        self.mr_integral = 0  # Integral term, motor right
        self.ml_last_update, self.mr_last_update = None, None  # Last update time

        self.anti_integral_windup = integral_windup  # Anti-windup flag
        self.real_time = real_time

    """
    Gets the time of the last measurement/update for the motor ml or mr
    """

    def get_dt(self, motor=None):
        # TODO: Implement smoothing for dt - potentially moving average
        if motor not in ['ml', 'mr']:
            raise ValueError("Motor must be 'ml' or 'mr'.")

        if not self.real_time:
            return self.dt

        last_update = f"{motor}_last_update"
        now = time.time()
        #         print(f"Last update: {getattr(self, last_update)}")
        if getattr(self, last_update) is None:
            setattr(self, last_update, now)
            return self.dt

        dt = now - getattr(self, last_update)
        setattr(self, last_update, now)
        return dt

    def compute(self, w_target, w_actual, motor=None):
        # motor should be either "ml" or "mr" for left or right motor
        if motor not in ['ml', 'mr']:
            raise ValueError("Motor must be 'ml' or 'mr'.")
        integral_attr = f"{motor}_integral"

        dt = self.get_dt(motor)

        error = w_target - w_actual  # Calculate the error
        P_out = self.Kp * error  # Proportional term
        I_out = self.Ki * getattr(self, integral_attr)  # Integral term
        raw_output = P_out + I_out

        if self.anti_integral_windup:
            # Anti-windup - only integrate if output is not saturated
            if self.min_output < raw_output < self.max_output:
                setattr(self, integral_attr, getattr(self, integral_attr) + error * dt)
                # equiv. to self.ml_integral += error * self.dt or self.mr_integral += error * self.dt
        else:
            setattr(self, integral_attr, getattr(self, integral_attr) + error * dt)

        return np.clip(raw_output, self.min_output,
                       self.max_output)  # Clamp the output to the min/max duty cycle limits

    def drive(self, v_desired, w_desired, wl_actual, wr_actual):
        # v_desired: m/s
        # w_desired (rotation), wl_actual, w_actual (rotation): rad/s

        if v_desired == 0 and w_desired == 0:
            return 0, 0, 0, 0

        # Calculate desired wheel angular velocities
        wl_desired = (v_desired + self.l * w_desired / 2) / self.r
        wr_desired = (v_desired - self.l * w_desired / 2) / self.r

        #         print(f"wl_des (rad/s): {wl_desired:.2f}, wr_des: {wr_desired:.2f}\nwl_des (rps): {wl_desired/(2*np.pi):.2f}, wr_des: {wr_desired/(2*np.pi):.2f}")

        # Compute duty cycles for left and right wheels
        duty_cycle_l = self.compute(wl_desired, wl_actual, 'ml')
        duty_cycle_r = self.compute(wr_desired, wr_actual, 'mr')
        return duty_cycle_l, duty_cycle_r, wl_desired, wr_desired