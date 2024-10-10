from gpiozero import AngularServo
import RPi.GPIO as GPIO
import time
import numpy as np
from robot_core.hardware.pin_config import DOOR_PIN, STAMP_PINS

#######################################################

"""
Provides methods for controlling the servos on the robot. 
- stamp() stamps the stamper
- deposit() opens the door for DOOR_OPEN_TIME
"""

class ServoController:

    # HARDWARE
    MIN_PULSE_WIDTH = 0.0006
    MAX_PULSE_WIDTH = 0.0023
    DOOR_OPEN_TIME = 10

    def __init__(self, door_pin = DOOR_PIN, min_pulse = MIN_PULSE_WIDTH, max_pulse = MAX_PULSE_WIDTH, stamp_pins = STAMP_PINS, debug = False):
        # Initialize the servos
        self.door_servo = AngularServo(door_pin, min_pulse_width = min_pulse, max_pulse_width = max_pulse)
        self.stamp_servos = [AngularServo(pin, min_pulse_width=min_pulse, max_pulse_width=max_pulse) for pin in stamp_pins]
        self.debug = debug

    def stamp(self):
        angle1 = 90
        angle2 = -90

        # Stamping down
        for _ in range(4):
            self._move_servos(angle1, angle2)
            print(f"Down servos: Angle 1 = {angle1}, Angle 2 = {angle2}")
            angle1 += 45
            angle2 -= 45
            time.sleep(1)

        # Ensure the angles stay within the valid range
        angle1 = max(-90, angle1)  # Keep angle1 between -90 and 90
        angle2 = min(90, angle2)   # Keep angle2 between -90 and 90

        # Move both servos up AFTER STAMP TO RETURN
        for _ in range(4):
            self._move_servos(angle1, angle2)
            print(f"Up servos: Angle 1 = {angle1}, Angle 2 = {angle2}")
            angle1 -= 45
            angle2 += 45
            time.sleep(1)

        # Ensure the angles stay within the valid range
        angle1 = min(90, angle1)   # Keep angle1 between -90 and 90
        angle2 = max(-90, angle2)  # Keep angle2 between -90 and 90

    def deposit(self, open_time = DOOR_OPEN_TIME):
        self._open_door()
        time.sleep(open_time)  # Keep the door open for the specified time
        self._close_door()

    def _open_door(self):
        if self.debug: print("Opening door...")
        self.door_servo.angle = -50  # Open position

    def _close_door(self):
        if self.debug: print("Closing door...")
        self.door_servo.angle = 90  # Close position

    def _move_servos(self, angle1, angle2):
        """Set angles for both stamping servos and log the movement."""
        self.stamp_servos[0].angle = angle1
        self.stamp_servos[1].angle = angle2
#         if self.debug == True:
#             print(f"Moving servos: Angle 1 = {angle1}, Angle 2 = {angle2}")

# Example usage
if __name__ == "__main__":

    controller = ServoController(debug = True)
    controller.deposit(open_time=10)  # Open the door for 10 seconds
    controller.stamp()  # Perform the stamping action