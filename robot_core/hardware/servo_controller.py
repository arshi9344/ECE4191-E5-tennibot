# with less jitter
import RPi.GPIO as GPIO
import time

from robot_core.hardware.pin_config import DOOR_PIN, STAMP_PINS, DOOR_OPEN_TIME

class ServoController:
    def __init__(self, door_pin=DOOR_PIN, stamp_pins=STAMP_PINS, debug=False):
        # Initialize the servos using GPIO for door and stamps
        self.door_pin = door_pin
        self.stamp_pins = stamp_pins
        self.debug = debug

        # GPIO setup
        GPIO.setmode(GPIO.BCM)

        # Initialize PWM for the door servo
        GPIO.setup(self.door_pin, GPIO.OUT)
        self.door_pwm = GPIO.PWM(self.door_pin, 50)  # 50Hz frequency
        self.door_pwm.start(0)

        # Initialize PWM for the stamp servos
        self.stamp_pwms = []
        for pin in self.stamp_pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, 50)  # 50Hz frequency for stamp servos
            pwm.start(0)
            self.stamp_pwms.append(pwm)

    # Function to map angle (-90 to 90 degrees) to pulse width and set duty cycle
    def set_servo_angle(self, pwm, angle, min_pulse, max_pulse):
        pulse_width = min_pulse + ((angle + 90) / 180.0) * (max_pulse - min_pulse)
        duty_cycle = pulse_width * 50 * 100  # Convert pulse width to duty cycle percentage
        if self.debug:
            print(f"Setting servo angle to {angle}Â°, Pulse width: {pulse_width}, Duty cycle: {duty_cycle}%")
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(2)  # Wait for the servo to move

    # Move the stamping servos
    def move_servos(self, angle1, angle2):
        """Set angles for both stamping servos and log the movement."""
        self.set_servo_angle(self.stamp_pwms[0], angle1, 0.0009, 0.0021)
        self.set_servo_angle(self.stamp_pwms[1], angle2, 0.0009, 0.0021)

    # Perform the stamping action
    def stamp(self):
        self.move_servos(-90, 90)  # Move down
        time.sleep(2)
        self.move_servos(90, -90)  # Move up
        time.sleep(2)

    # Open the door
    def open_door(self, open_time=DOOR_OPEN_TIME):
        if self.debug:
            print("Opening door...")
        self.set_servo_angle(self.door_pwm, 90, 0.001, 0.002)  # Door open position
        time.sleep(open_time)  # Keep the door open for the specified time
        self.close_door()

    # Close the door
    def close_door(self):
        if self.debug:
            print("Closing door...")
        self.set_servo_angle(self.door_pwm, -70, 0.001, 0.002)  # Door close position
        self.move_servos(90, -90)  # Reset stamp positions after door closes

    # Cleanup GPIO and stop PWM
    def cleanup(self):
        if self.debug:
            print("Cleaning up GPIO and stopping PWM...")
        self.door_pwm.stop()
        for pwm in self.stamp_pwms:
            pwm.stop()
        GPIO.cleanup()


# Example usage
if __name__ == "__main__":
    try:
        controller = ServoController(DOOR_PIN, STAMP_PINS, debug=True)
        controller.stamp()  # Perform the stamping action
        controller.open_door(open_time=10)  # Open the door for 10 seconds
    except KeyboardInterrupt:
        pass
