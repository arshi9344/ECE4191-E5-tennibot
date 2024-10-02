
"""
    Pin assignments for the Raspberry Pi 4B + physical dimensions + motor encoder steps.
"""

# Encoder pins
# ML = motor left, MR = motor right

ML_ENC_A = 15 # yellow encoder wire
ML_ENC_B = 14 # white encoder wire

MR_ENC_A = 24 # yellow encoder wire
MR_ENC_B = 23 # white encoder wire

# The number of encoder steps per revolution.
MAX_ENC_STEPS = 900

# Motor Pins
ML_IN1 = 17 # IN1
ML_IN2 = 27 # IN2
ML_ENA = 11 # Used for PWM

MR_IN3 = 22 # IN3
MR_IN4 = 10 # IN4
MR_ENB = 9 # Used for PWM

#Ultrasonic
TRIGGER = [5,20]  # GPIO 5 for Trigger of Set 1 (right) and 20 for set 2 (left)
ECHO = [6,26]     # GPIO 6 for Echo of Set 1 (right) and 26 for set 2 (left)

if __name__ == '__main__':
    try:
        import RPi.GPIO as GPIO
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        print("Pin assignments imported successfully.")
    except ImportError:
        print("RPi.GPIO not installed. GPIO operations not run.")
        pass
