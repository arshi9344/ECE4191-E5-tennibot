# with jitter but faster
# problem besides jittering (just found out 7/10): door open a bit before stamping, without being called

from gpiozero import AngularServo
import time

#DEFINE THIS IN PIN_CONFIG
DOOR_PIN = 16
STAMP_PINS = [21, 19] # STAMP PIN LEFT IN LIST IS 
####################################################
#HARDWARE
MIN_PULSE_WIDTH = 0.0006
MAX_PULSE_WIDTH = 0.0023
DOOR_OPEN_TIME = 10
####################################################

class ServoController:
    def __init__(self, door_pin = DOOR_PIN, min_pulse = MIN_PULSE_WIDTH, max_pulse = MAX_PULSE_WIDTH, stamp_pins = STAMP_PINS, debug = False):
        # Initialize the servos
        self.door_servo = AngularServo(door_pin, min_pulse_width = min_pulse, max_pulse_width = max_pulse)
        self.stamp_servos = [AngularServo(pin) for pin in stamp_pins]
        self.debug = debug

    def stamp(self):
            self.move_servos(-90, 90)   # down
            time.sleep(2)
        
            self.move_servos(90, -90)   # up
            time.sleep(2)

    def move_servos(self, angle1, angle2):
        """Set angles for both stamping servos and log the movement."""
        self.stamp_servos[0].angle = angle1
        self.stamp_servos[1].angle = angle2
#         if self.debug == True:
#             print(f"Moving servos: Angle 1 = {angle1}, Angle 2 = {angle2}")

    def open_door(self, open_time = DOOR_OPEN_TIME):
        self.stamp_servos[0].angle = 75
        self.stamp_servos[1].angle = -75
        if self.debug == True:
            print("Opening door...")
        self.door_servo.angle = -70  # Open position
        time.sleep(open_time)  # Keep the door open for the specified time
        self.close_door()

    def close_door(self):
        if self.debug == True:
            print("Closing door...")
        self.door_servo.angle = 90  # Close position
        self.stamp_servos[0].angle = 90
        self.stamp_servos[1].angle = -90

# Example usage
if __name__ == "__main__":

    controller = ServoController(DOOR_PIN, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH, STAMP_PINS, debug = True)
    controller.stamp()  # Perform the stamping action
    controller.open_door(open_time=10)  # Open the door for 10 seconds

    