import time
import numpy as np
import RPi.GPIO as GPIO
import gpiozero

from robot_core.utils.filters import LowPassFilter
from robot_core.hardware.pin_config import *
from robot_core.hardware.dimensions import *

class DiffDriveRobot:
    def __init__(
            self,
            dt=0.05,
            real_time=False,
            tau=0.1,
            wheel_radius=WHEEL_RADIUS,
            wheel_sep=WHEEL_SEP,
            max_enc_steps=MAX_ENC_STEPS,
            ml_pwm=ML_ENA,
            mr_pwm=MR_ENB,
            ml_in1=ML_IN1,
            ml_in2=ML_IN2,
            mr_in3=MR_IN3,
            mr_in4=MR_IN4,
            ml_encA=ML_ENC_A,
            ml_encB=ML_ENC_B,
            mr_encA=MR_ENC_A,
            mr_encB=MR_ENC_B
    ):
        self.x = 0.0  # x-position, meters
        self.y = 0.0  # y-position, meters
        self.th = 0.0  # orientation, angle in radians

        self.wl = 0.0  # rotational velocity left wheel, rad/s
        self.wr = 0.0  # rotational velocity right wheel, rad/s

        #         self.wl_prev = 0
        #         self.wr_prev = 0

        self.wl_smoothed_func = LowPassFilter(tau)
        self.wr_smoothed_func = LowPassFilter(tau)

        self.dt = dt  # time delta in seconds. The control loop runs every dt. Faster means the control loop runs more often. We can increase this to reduce CPU load on our robot.
        self.r = wheel_radius  # wheel radius in meters.
        self.l = wheel_sep  # wheel separation in meters
        self.max_enc_steps = max_enc_steps  # steps in the encoder per revolution

        self.real_time = real_time  # if True, the wheel velocity measurements consider the actual time elapsed between measurements. If False, it uses the predefined dt value.
        self.last_update = None  # last time the control loop ran

        # Pin numbers
        self.motor_L_in1 = ml_in1  # Input 1 (motor left)
        self.motor_L_in2 = ml_in2  # Input 2 (motor left)
        self.motor_R_in3 = mr_in3  # Input 3 (motor right)
        self.motor_R_in4 = mr_in4  # Input 4 (motor right)
        GPIO.setup(self.motor_L_in1, GPIO.OUT)
        GPIO.setup(self.motor_L_in2, GPIO.OUT)
        GPIO.setup(self.motor_R_in3, GPIO.OUT)
        GPIO.setup(self.motor_R_in4, GPIO.OUT)

        # Initialize encoders
        self.ML_ENC = gpiozero.RotaryEncoder(a=ml_encA, b=ml_encB, max_steps=max_enc_steps, wrap=True)
        self.MR_ENC = gpiozero.RotaryEncoder(a=mr_encA, b=mr_encB, max_steps=max_enc_steps, wrap=True)
        self.ml_enc_steps = 0  # left motor encoder value (AKA shaft angle) in steps. Note, this is NOT in radians. It counts from 0 -> MAX_ENC_STEPS -> -MAX_ENC_STEPS (wraps around to -ve value)
        self.mr_enc_steps = 0  # right motor encoder value (AKA shaft angle) in steps.

        # Initialize motor control pins
        GPIO.setup(ml_pwm, GPIO.OUT)
        GPIO.setup(mr_pwm, GPIO.OUT)
        self.motor_L_pwm = GPIO.PWM(ml_pwm,
                                    1000)  # 1000 Hz frequency. This works well for our motor. Remember that motor speed is controlled by varying the duty cycle of our PWM frequency, and not the frequency itself.
        self.motor_R_pwm = GPIO.PWM(mr_pwm, 1000)
        self.motor_L_pwm.start(0)
        self.motor_R_pwm.start(0)

    '''
    This method calculates the change in encoder steps between the current and previous time step.
    The logic might seem complicated but this is to deal with the wraparound in value (from 0 -> MAX_ENC_STEPS -> -MAX_ENC_STEPS) that the encoder values have.
    e.g. for the first rotation, the encoder goes from 0 to 3600, then for every rotation thereafter, it goes from -3600 to 3600, then wraps around again.
    '''

    @property
    def pose(self):
        return self.x, self.y, self.th

    @property
    def wl_smoothed(self):
        return self.wl_smoothed_func.value

    @property
    def wr_smoothed(self):
        return self.wr_smoothed_func.value

    def _get_encoder_delta(self, curr_value, prev_value):
        raw_delta = curr_value - prev_value
        delta = raw_delta
        wraparound_thresh = np.ceil((2 * self.max_enc_steps + 1) / 2)

        if raw_delta > wraparound_thresh:
            delta = raw_delta - (2 * self.max_enc_steps + 1)
        elif raw_delta < -wraparound_thresh:
            delta = raw_delta + (2 * self.max_enc_steps + 1)

        return delta

    def _get_dt(self):
        # TODO: Implement smoothing for dt - potentially moving average
        if not self.real_time:
            return self.dt

        now = time.time()
        # print(f"Last update: {self.last_update}")
        if self.last_update is None:
            self.last_update = now
            return self.dt

        dt = now - self.last_update
        self.last_update = now  # this needs to be in between the above and below lines. Don't move it
        return dt

    # Wheel velocities in radians/sec
    '''
    This method reads the encoder values and calculates the wheel velocities in rad/s.
    It uses the get_encoder_delta method to calculate the change in encoder steps between the current and previous time step.
    It then converts this change in steps to radians, and then to radians per second.
    '''

    def _read_wheel_velocities(self, dt):

        ml_enc_now, mr_enc_now = self.ML_ENC.steps, self.MR_ENC.steps

        # Calculate change in steps, accounting for wrap-around
        ml_enc_delta = self._get_encoder_delta(ml_enc_now, self.ml_enc_steps)
        mr_enc_delta = self._get_encoder_delta(mr_enc_now, self.mr_enc_steps)

        # Convert step change to radians
        ml_delta_rad = ml_enc_delta / self.max_enc_steps * 2 * np.pi
        mr_delta_rad = mr_enc_delta / self.max_enc_steps * 2 * np.pi

        # Calculate velocities
        self.wl = ml_delta_rad / dt  # rad/s
        self.wr = mr_delta_rad / dt  # rad/s

        # Calculate smoothed velocities
        self.wl_smoothed_func.update(self.wl, dt)
        self.wr_smoothed_func.update(self.wr, dt)

        # Update previous steps
        self.ml_enc_steps = ml_enc_now
        self.mr_enc_steps = mr_enc_now

        return self.wl, self.wr

    '''
    This method sets the motor speed based on the duty cycle provided. 
    It also sets the direction of the motor based on the sign of the duty cycle.
    The duty cycle is the percentage of time the motor is on, and it MUST be a value between -100 and 100.
    '''

    def set_motor_speed(self, left_duty_cycle, right_duty_cycle):
        # Set direction
        GPIO.output(self.motor_L_in1, GPIO.HIGH if left_duty_cycle >= 0 else GPIO.LOW)
        GPIO.output(self.motor_L_in2, GPIO.LOW if left_duty_cycle >= 0 else GPIO.HIGH)
        GPIO.output(self.motor_R_in3, GPIO.HIGH if right_duty_cycle >= 0 else GPIO.LOW)
        GPIO.output(self.motor_R_in4, GPIO.LOW if right_duty_cycle >= 0 else GPIO.HIGH)

        # Set speed
        self.motor_L_pwm.ChangeDutyCycle(abs(left_duty_cycle))
        self.motor_R_pwm.ChangeDutyCycle(abs(right_duty_cycle))

    """
    This method stops the robot by setting the left and right motor speeds to 0.
    """
    def stop(self):
        self.set_motor_speed(0, 0)

    '''
    This method calculates the linear and angular velocity of the robot based on the wheel velocities.
    It uses the formulae for differential drive robots to calculate the linear and angular velocity.
    This is pretty much identical to Michael's code in the ECE4191 repo.
    '''

    def base_velocity(self, wl, wr):
        v = (wl * self.r + wr * self.r) / 2.0  # linear velocity, m/s, +ve is forward
        w = (wl * self.r - wr * self.r) / self.l  # angular velocity, rad/s, +ve is CCW. Note that the negative sign
        # is due to the way the motors are oriented, and so we need it to 'correct' our w calculation
        # so that +ve w is CCW, adhering to convention.
        return v, w

    """
    This method updates the robot's pose (x, y, theta) based on the wheel velocities.
    It uses the base_velocity method to calculate the linear and angular velocity of the robot.
    It then uses these velocities to update the robot's pose (x, y, theta) based on the kinematic equations.
    """

    def pose_update(self, duty_cycle_ml, duty_cycle_mr):
        # TODO: remove the call to get_dt here, relocate it to the read_wheel_velocities method, and remove the need for it in the pose_update method (below) by having read_wheel_velocities return dt, or the distance travelled (dt*v) directly.
        dt = self._get_dt()

        self.set_motor_speed(duty_cycle_ml, duty_cycle_mr)

        wl, wr = self._read_wheel_velocities(dt)  # get wheel velocities in rad/s
        v, w = self.base_velocity(wl, wr)

        self.x = self.x + dt * v * np.cos(self.th)
        self.y = self.y + dt * v * np.sin(self.th)
        self.th = self.th + w * dt

        return self.x, self.y, self.th


