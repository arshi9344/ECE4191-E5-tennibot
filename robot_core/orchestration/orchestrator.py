
import time
import threading
import numpy as np
import random
from robot_core.control.PI_controller import PIController
from robot_core.motion.tentacle_planner import TentaclePlanner
from robot_core.perception.opencv_detector import TennisBallDetector
from robot_core.hardware.simulated_diff_drive_robot import DiffDriveRobot

"""
Orchestrator - coordinates PI motion control, perception, and motion planning. 
Single-threaded version of Orchestrator. 
"""
#TODO: Remove the TentaclePlanner and PIController classes from the orchestrator and pass them as arguments to the start() method. (Depdendency Injection)

class Orchestrator:
    def __init__(self):

        self.running = True
        self.last_update = None
        self.dt = 0.1

        self.robot = None
        self.controller = None
        self.detector = None
        self.planner = None
        self.control_thread = None

        self.poses = []
        self.goal_positions = []
        self.velocities = []
        self.desired_velocities = []
        self.duty_cycle_commands = []
        self.error_sums = []
        self.errors = []
        self.actual_dts = []
        self.scan_locations = []

        self.duration = 0
        self.start_time = 0


    def get_dt(self):
        now = time.time()
        if self.last_update is None:
            self.last_update = now
            return self.dt

        dt = now - self.last_update
        self.last_update = now # this needs to be in between the above and below lines. Don't move it
        return dt


    """
    Notes to myself (Patrick): 
    - Outer tennis court quadrant is 5.48m deep, 4.12m wide. Inner quadrants are 6.40m deep.
    - 
    """
    def control_loop(self, width=4.12, depth=5.48):
        try:
            # Log data
            x, y, th = self.robot.pose
            self.poses.append([x, y, th])
            self.velocities.append([self.robot.wl, self.robot.wr])

            # Calculate number of vertical scan lines based on maximum camera detection distance
            max_scan_distance = .8
            num_scan_lines = int(np.ceil((width -( 2 *max_scan_distance)) / ( 2 *max_scan_distance)))
            scan_lines = np.linspace(max_scan_distance, width -max_scan_distance, num_scan_lines)
            scan_lines = [-v for v in scan_lines]

            # Calculate vertical stop points
            num_scan_points = int(np.ceil((depth -( 2 *max_scan_distance)) / (2 * max_scan_distance)))
            scan_points = np.linspace(max_scan_distance, depth -max_scan_distance, num_scan_points)

            scan_line_idx = 0
            scan_point_idx = 0

            # Define scan directions
            UP = 1
            DOWN = -1
            current_direction = UP
            while self.running:

                scan_direction = "anticlockwise" if current_direction == UP else "clockwise"
                scan_line_distance = scan_lines[scan_line_idx]
                scan_point_distance = scan_points[scan_point_idx]
                print(f"Scan point X: {scan_line_distance}, Y: {scan_point_distance}, th: {scan_direction}")

                # Move to the next scanning position
                print(f"Moving to next scanning position: col: {scan_line_idx}, row: {scan_point_idx}")
                self.move_to_scan_position(scan_point_distance, scan_line_distance, current_direction)

                # Scan for ball
                ball_found = self.scan_for_ball(scan_direction=scan_direction, simulate_camera=False)
                if ball_found:
                    self.approach_ball(50)
                    print("Going home now.")
                    self.navigate(0, 0, self.robot.th) # go home
                    return  # End the control loop

                # Move to the next scan point or line
                scan_point_idx, scan_line_idx, current_direction = self.update_scan_indices(scan_point_idx, scan_line_idx, scan_lines, scan_points, current_direction)

                if scan_line_idx >= len(scan_lines):
                    # This doesn't actually run because self.update_scan_indices doesn't let scan_line_idx > len()
                    print("Scanning complete")
                    self.navigate(0, 0, self.robot.th) # return to origin
                    break

        except Exception as e:
            print(f"\nUnhandled Exception in Orchestrator control loop: {e}. Stopping Robot.\n")
            self.stop()



    def move_to_scan_position(self, scan_point_distance, scan_line_distance, direction):

        """# Proper way to move to our next scan location"""
        try:
            goal_angle = 0 if direction == 1 else -np.pi
            self.navigate(scan_point_distance, scan_line_distance, goal_angle)
            self.rotate(goal_angle, absolute=True)

        except Exception:
            raise


    def navigate(self, goal_x, goal_y, goal_th, strategy="tentacles"):
        try:
            while True and self.running:
                inputs = self.planner.get_control_inputs(goal_x, goal_y, goal_th, *self.robot.pose, strategy=strategy)
                self.apply_control_inputs(inputs)
                self.goal_positions.append([goal_x, goal_y, goal_th])
                if inputs['linear_velocity'] == 0.0 and inputs['angular_velocity'] == 0.0:
                    # print("Rotation complete: planner indicates no more movement needed.")
                    break

                time.sleep(self.dt)

        except KeyboardInterrupt:
            print("Keyboard interrupt caught inside self.navigate() - reraising.")
            raise

        except Exception:
            print("Unhandled exception inside self.navigate()")
            raise Exception

        return


    def update_scan_indices(self, scan_point_idx, scan_line_idx, scan_lines, scan_points, current_direction):
        UP, DOWN = 1, -1

        scan_point_idx += current_direction # Increment scan_point_idx according to the current direction

        if scan_point_idx < 0 or scan_point_idx >= len \
                (scan_points): # Check if we've reached the end of the vertical scan

            if current_direction == UP: # We've completed a vertical scan, move to the next scan line
                # We were moving up, so now we need to start moving down
                # and move to the next scan line to the left
                current_direction = DOWN
                scan_line_idx += 1
                scan_point_idx = len(scan_points) - 1  # Start from the top
            else:  # current_direction == DOWN
                # We were moving down, so now we need to start moving up
                # and move to the next scan line to the left
                current_direction = UP
                scan_line_idx += 1
                scan_point_idx = 0  # Start from the bottom

            # Check if we've completed all scan lines
            if scan_line_idx >= len(scan_lines):
                # We've completed all scan lines, reset to start a new sweep if needed
                scan_line_idx = 0
                current_direction = UP
                scan_point_idx = 0

        return scan_point_idx, scan_line_idx, current_direction



    """
    This is a subroutine that rotates on the spot and scans for the ball. 
    """
    def scan_for_ball(self,
                      scan_direction='clockwise',
                      simulate_camera=False,
                      rotation_thres=6.27, # just below 360 degrees
                      rotation_increment = 0.5236 # 30 * np.pi/180  # 30 degrees
                      ):

        rotation_angle = 0
        initial_theta = self.robot.th
        self.scan_locations.append([*self.robot.pose]) # log data
        print("\nBeginning rotational scan now.")
        while rotation_angle <= rotation_thres and self.running:
            if simulate_camera:
                print("Simulating camera wait...")
                time.sleep(2)  # Wait for 2 seconds to simulate camera processing
                # Simulate ball detection (randomly for demonstration)

                #                 ball_detected = random.choice([True, False])
                ball_detected = False
                is_between = True

                is_between = random.choice([True, False]) if ball_detected else False
                distance = random.uniform(-1, 1) if ball_detected else 0
                ball_y_dist = 30
            else:
                # Take an image and detect tennis balls
                ball_detected = self.detector.detect()
                #                 self.detector.display_frame()
                distance = self.detector.get_ball_distance_from_lines()
                ball_y_dist = self.detector.get_ball_vertical_y()
                time.sleep(1)
            if ball_detected:
                if distance == 0:
                    print("Ball detected between lines. Exiting scan_for_ball(). Returning True")
                    return ball_y_dist
                else:
                    print(f"Ball detected. Distance from center: {distance:.2f}")
                    rotation_direction = 1 if (distance < 0) else -1
                    scaling_factor = 0.02
                    rotation_adjustment = rotation_direction * 10 * np.pi /180
                    print(f"Rotating by {rotation_adjustment} radians to adjust.\n")
                    self.rotate(rotation_adjustment)  # Rotate by 10 degrees
                    return True
            else:
                rotation_angle += rotation_increment
                if rotation_angle > rotation_thres:
                    break
                else:
                    print(f"No ball detected. Rotating by {rotation_increment * 180 / np.pi:.2f} degrees.")
                    self.rotate(rotation_increment if scan_direction == 'clockwise' else -rotation_increment)


        # If we've rotated full threshold and haven't found a ball, rotate back to initial position
        print("No ball found. Exiting scan routine.")
        #         self.rotate(initial_theta - self.robot.th) # We can REMOVE THIS if using proper motion planning
        return False

    """
    approach ball()
    """

    def approach_ball(self, ball_y=30, y_threshold=30):
        print("Attempting to approach ball now. ")
        max_iterations = 10  # Prevent infinite loop
        iterations = 0

        while iterations < max_iterations and self.running:
            # Calculate distance to move
            # Combine linear and exponential scaling
            linear_factor = 0.001  # Adjust this value to change the linear component
            exp_factor = 0.0  # Adjust this value to change the exponential component
            # Take a new picture

            # Get the ball's vertical position

            if ball_y is None:

                print("Ball not detected. Cannot approach.")
                return False

            if ball_y < y_threshold:
                print(f"Ball is close enough (y = {ball_y}). Approach successful.")
                return True

            distance = linear_factor * ball_y + exp_factor * np.exp(ball_y / 50)


            # Limit the maximum movement to prevent overshooting
            max_movement = 0.5  # meters
            distance = min(distance, max_movement)

            print(f"Moving forward {distance:.2f} meters (ball_y = {ball_y})")
            new_x = self.robot.x + distance * np.cos(self.robot.th)
            new_y = self.robot.y + distance * np.sin(self.robot.th)

            self.navigate(new_x, new_y, self.robot.th)

            self.detector.detect()
            ball_y = self.detector.get_ball_vertical_y(0)


            iterations += 1

        print(f"Could not approach ball within {max_iterations} iterations.")

        return False


    """
    EMERGENCY FIX methods are methods that we would ideally not need, but are required as workarounds
    due to the limited implementation and software development human resources that we have.
    - Ideally, our motion planner would be good enough to accept x, y, th coords without having funky outputs
    - Additionally, the motion planner (e.g. TentaclePlanner) would ideally have a good enough interface that we wouldn't
    need to have yet another abstraction of an abstraction
    """

    """
    EMERGENCY FIX METHOD FOR MILESTONE 1
    Rotates the robot. 
    - Absolute set False means that the robot rotates by the angle parameter. 
    - Absolute set True means that the robot will rotate to the angle supplied. Actual rotation may or may not be angle.

    """
    def rotate(self, angle, absolute=False):
        print("Rotate method entered.")
        goal_theta = angle
        if not absolute:
            goal_theta += self.robot.th
        while True and self.running:
            inputs = self.planner.get_control_inputs(
                self.robot.x, self.robot.y, goal_theta,
                self.robot.x, self.robot.y, self.robot.th,
                strategy="rotate"
            )
            self.apply_control_inputs(inputs)
            if inputs['linear_velocity'] == 0.0 and inputs['angular_velocity'] == 0.0:
                #                 print("Rotation complete: planner indicates no more movement needed.")
                break
            time.sleep(self.dt)
        return

    """ 
    Applies control inputs from the Motion Planner (e.g. TentaclePlanner) into the robot.
    """
    def apply_control_inputs(self, inputs):
        if len(inputs) == 2:
            linear = inputs[0]
            angular = inputs[1]
        else:
            linear = inputs['linear_velocity']
            angular = inputs['angular_velocity']

        duty_cycle_l, duty_cycle_r, wl_desired, wr_desired = self.controller.drive(
            linear,
            angular,
            self.robot.wl_smoothed,
            self.robot.wr_smoothed
        )
        # Log data
        x, y, th = self.robot.pose
        self.poses.append([x, y, th])
        self.velocities.append([self.robot.wl, self.robot.wr])
        self.duty_cycle_commands.append([duty_cycle_l, duty_cycle_r])
        self.desired_velocities.append([wl_desired, wr_desired])

        self.robot.pose_update(duty_cycle_l, duty_cycle_r)

    """
    Start the Orchestrator, initialise children class instances, and begin operation.
    """
    def start(
            self,
            robot=DiffDriveRobot(real_time=True),
            controller=PIController(real_time=True),
            detector=None,
            planner=TentaclePlanner(max_linear_velocity=0.2, max_angular_velocity=2)
    ):
        if self.robot is None:
            print("Initialising robot.")
            self.robot = robot

        if self.controller is None:
            print("Initialising controller.")
            self.controller = controller

        if self.detector is None:
            print("Initialising tennis ball detector.")
            self.detector = detector

        if self.planner is None:
            print("Initialising tentacle planner.")
            self.planner = planner


        self.control_thread = threading.Thread \
            (target=self.control_loop) # you can change which control loop is called. Supply the function name, no parentheses.
        self.start_time = time.time()
        self.control_thread.start()

    """
    Stops the Orchestrator and stops the robot.
    """
    def stop(self):
        self.running = False
        self.control_thread.join()
        self.robot.set_motor_speed(0, 0)
        self.duration = time.time() - self.start_time
        print(f"Ended run. Robot ran for: {self.duration:.2f} s")