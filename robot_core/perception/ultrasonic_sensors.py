import RPi.GPIO as GPIO
import time
import math

current_dir = os.getcwd()
cur__ = os.path.abspath(os.path.join(current_dir,"..",".."))
sys.path.append(cur__)

from robot_core.hardware.dimensions import *
from robot_core.hardware.pin_config import *



class UltrasonicSensor:
    """
    A class to interact with an ultrasonic sensor for distance measurement with optional debugging and alignment correction.
    
    Attributes:
        trigger_pins (list of int): GPIO pins connected to the sensor's trigger pins.
        echo_pins (list of int): GPIO pins connected to the sensor's echo pins.
        sensors (int): Number of ultrasonic sensors being used.
        debug (bool): Whether to print debug information.
        rotate (function): Function to issue rotation commands to the robot.
    """
    
    def __init__(self, trigger_pins=TRIGGER, echo_pins=ECHO, sensor_count=2, rotate_function=None, debug=False, sensor_distance = ULTRA_DIST_BTWN, num_samples=30):
        """
        Initializes the ultrasonic sensors.

        Args:
            trigger_pins (list of int): GPIO pins connected to the trigger pins of the sensors.
            echo_pins (list of int): GPIO pins connected to the echo pins of the sensors.
            rotate_function (function): Function to control the robot's rotation. Defaults to None.
            debug (bool): If True, enables debug mode to print sensor readings and other information.
        
        Sets up GPIO pins and initializes the sensor system.
        """
        self.trigger_pins = trigger_pins
        self.echo_pins = echo_pins
        self.sensor_count = len(trigger_pins) # Track the number of sensors
        self.debug = debug
        self.rotate = rotate_function # Set desired rotate function
        GPIO.setmode(GPIO.BCM) # Set GPIO mode to Broadcom pin-numbering scheme
        self.sensor_distance = sensor_distance
        self.num_samples = num_samples
        
        for trigger_pin, echo_pin in zip(trigger_pins, echo_pins): 
            GPIO.setup(trigger_pin, GPIO.OUT) # Set trigger pins as output
            GPIO.setup(echo_pin, GPIO.IN) # Set echo pins as input
            GPIO.output(trigger_pin, GPIO.LOW) # Set to low when not being used in normal state
        
        #time.sleep(2)  # Allow sensors to settle if necessary
    
    def _send_pulse(self, trigger_pin):
        """
        Sends a pulse from the specified trigger pin to initiate the ultrasonic measurement.

        Args:
            trigger_pin (int): The GPIO pin number connected to the trigger.
        
        The pulse duration is very short (10 microseconds).
        """
        GPIO.output(trigger_pin, GPIO.HIGH) # Signals high for trigger pin
        time.sleep(0.00001)  # 10 microseconds as per the datasheet for correct operation
        GPIO.output(trigger_pin, GPIO.LOW) # Back to normal state
    
    def _measure_pulse(self, echo_pin, timeout=1.0):
        """
        Measures the duration of the pulse received on the specified echo pin.

        Args:
            echo_pin (int): The GPIO pin number connected to the echo.

        Returns:
            float: Duration of the pulse in seconds.
        """
        while GPIO.input(echo_pin) == GPIO.LOW: # Waiting for start of the pulse
            pass

        start_time = time.time() # Records start time

        while GPIO.input(echo_pin) == GPIO.HIGH:
            if time.time() - start_time > timeout:
             print("Failed Reading, didn't receive pulse after 1 second")
             return -1  # Timeout reached, return invalid reading
            
        end_time = time.time() # Time at end of pulse
        
        # if self.debug:
        #     print(f"time taken: {end_time - start_time}")

            
        return end_time - start_time # Time taken to receive pulse being sent back to sensor
    
    def get_distances(self, delay=0.1):
        """
        Retrieves the distances measured by all ultrasonic sensors, with averaging and filtering.
        
        Args:
            num_samples (int): Number of samples to take for each sensor to improve accuracy.
            delay (float): Delay between each sample in seconds.
        
        Returns:
            list of float: List of filtered and averaged distances measured by each sensor in centimeters.
        """
        distances = []  # List for storing filtered distances
        
        for trigger_pin, echo_pin in zip(self.trigger_pins, self.echo_pins):  # Loop over each sensor's respective trigger and echo pins
            samples = []  # Collect multiple samples for each sensor
            
            for _ in range(self.num_samples):
                self._send_pulse(trigger_pin)  # Sending pulse
                pulse_duration = self._measure_pulse(echo_pin)  # Get duration of pulse
                
                # Check if the pulse duration is valid before appending
                if pulse_duration > 0:
                    distance = pulse_duration * 17150  # Convert pulse duration to distance in cm
                    samples.append(distance)
                    
                time.sleep(delay)  # Wait between samples
            
            # Apply filtering to remove outliers
            filtered_samples = self._filter_outliers(samples)
            
            if filtered_samples:
                avg_distance = round(sum(filtered_samples) / len(filtered_samples), 2)  # Average the filtered distances
                distances.append(avg_distance)  # Add the averaged, filtered distance to the list
            else:
                distances.append(None)  # Append None if no valid distance is measured
            
        if self.debug:
            #print(f"unFiltered Distances: {samples}")
            print(f"Filtered Distances: {distances}")
            
        
        return distances

    
    def detect_obstacle(self, threshold_distance): # Purely going to be used for detecting drop off container
        """
        Checks if any obstacle is detected within the specified distance threshold.

        Args:
            threshold_distance (float): The distance threshold in centimeters.

        Returns:
            bool: True if any sensor detects an object within the threshold distance, otherwise False.
        """
        distances = self.get_distances() # Distance from all sensors
        obstacle_detected = any(distance < threshold_distance for distance in distances) # Checking distance below threshold
        
        if self.debug:
            if obstacle_detected:
                print(f"Obstacle detected within {threshold_distance} cm")
            else:
                print(f"No obstacles detected within {threshold_distance} cm")
        
        return obstacle_detected
    
    def check_alignment(self, depot_distance_threshold, alignment_tolerance):
        """
        Checks if the robot is properly aligned with the depot based on the distances
        measured by the two front ultrasonic sensors, and commands the robot to drive straight
        until aligned.
    
        Args:
            depot_distance_threshold (float): Target distance from the depot in centimeters.
            alignment_tolerance (float): Acceptable deviation from the target distance in centimeters.
    
        Returns:
            str: A string indicating the alignment status or action taken.
        """
        while True:
            distances = self.get_distances()
    
            if len(distances) != 2:
                raise ValueError("This function is designed to work with exactly two sensors for alignment.")
    
            right_distance, left_distance  = distances  # Assuming right is first and left is second
            
            # Check alignment
            if abs(left_distance - right_distance) < alignment_tolerance:
                if self.debug:
                    print(f"Robot is aligned with the depot (left: {left_distance} cm, right: {right_distance} cm)")
                    print("alligned")

                # Check if both distances are below the depot distance threshold
                if left_distance < depot_distance_threshold and right_distance < depot_distance_threshold:
                    print("you have arrived and are alligned!")
                else:
                    # Command to drive straight until both distances are below the threshold
                    if self.debug:
                        print("Driving straight...")
                        return (right_distance + left_distance) / 2
    
            else:
                # If right sensor is closer, the robot needs to rotate right
                
                distance_difference = abs(left_distance - right_distance)
                angle_radians = math.atan(distance_difference / self.sensor_distance)
                angle_degrees = math.degrees(angle_radians)

                
                if right_distance < left_distance:
                    if self.debug:
                        print(f"Robot needs to rotate right (left: {left_distance} cm, right: {right_distance} cm)")
                        print(f"rotating by: {angle_radians} rad which is {angle_degrees} deg")  # Call the function to rotate right
                    rotation = angle_radians
                    return rotation
                
                else:  # left_distance < right_distance
                    if self.debug:
                        print(f"Robot needs to rotate left (left: {left_distance} cm, right: {right_distance} cm)")
                        print(f"rotating by: {-angle_radians} rad which is {angle_degrees} deg")  # Call the function to rotate left
                    rotation = -angle_radians
                    return rotation

            # ADD A DELAY HERE TO ALLOW FOR ROTATION TO BE PERFORMED

    def _filter_outliers(self, distances, threshold=1):
        """
        Filters out extreme outlier distances that may be errors.
        
        Args:
            distances (list of float): List of distance readings.
            threshold (float): Maximum allowable deviation from the median to consider a value valid.
        
        Returns:
            list of float: List of filtered distance readings.
        """
        if not distances:
            return []
        
        median = sorted(distances)[len(distances) // 2]  # Calculate the median
        return [d for d in distances if abs(d - median) < threshold]  # Keep values close to the median
    

    
    def close(self):
        """
        Cleans up GPIO settings and releases resources.

        This method should be called when the sensor is no longer needed to ensure proper cleanup.
        """
        GPIO.cleanup()
