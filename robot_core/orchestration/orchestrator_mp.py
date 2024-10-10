import time
import numpy as np
import json
import random
from IPython import display
import multiprocessing as mp
import logging
import queue
import os
import psutil
from queue import Empty
import traceback
from typing import Optional
import cv2

from matplotlib import pyplot as plt

from robot_core.control.PI_controller import PIController
from robot_core.hardware.diff_drive_robot import DiffDriveRobot
from robot_core.hardware.dimensions import COURT_YLIM, COURT_XLIM
from robot_core.motion.tentacle_planner import TentaclePlanner
from robot_core.perception.detection_results import BallDetection
from robot_core.perception.ultrasonic_sensors import UltrasonicSensor
from robot_core.hardware.servo_controller import ServoController
from robot_core.perception.vision_model.tennis_YOLO import TennisBallDetector

from robot_core.utils.command_utils import StatefulCommandQueue
from robot_core.utils.logging_utils import setup_logging
from robot_core.utils.robot_log_point import RobotLogPoint
from robot_core.orchestration.scan_point_utils import ScanPointGenerator
from robot_core.coordinator.commands import RobotCommands
from robot_core.utils.position import Position, PositionTypes
# from robot_core.perception.ultrasonic_sensors import UltrasonicSensor # moved import inside the Orchestrator.run() method
"""
How Orchestrator works is that it accepts a command, and then acts upon it. It checks the command queue on EVERY iteration, 
and grabs the latest command and immediately acts upon it. The responsibility for checking if the command is done is on the DecisionMaker, and NOT the Orchestrator.

Orchestrator is DUMB: It gets a command, acts upon it, and then notifies the StatefulCommandQueue that the command is done. It doesn't care about the command after that.

"""

class Orchestrator:
    def __init__(
            self,
            dt=0.1

    ):

        # Parameters
        self.dt = dt
        self.default_camera_idx = 0

        self.quadrant_bounds = (0, COURT_XLIM, 0, COURT_YLIM)
        self.scan_points = [(2.0, -2.0),(4.0, -2.0),(4.0, -4.0),(2.0, -4.0)]
        self.curr_scan_point = 0

        #### Internal state variables
        self.last_update = None
        self.start_time = None
        self.last_scan_time = None
        self.mode = None
        self.target_pose = Position(0,0,0,PositionTypes.ROBOT)

        ###### Subcomponents ######
        # These are initialised inside run()
        self.robot = DiffDriveRobot(0.1, real_time=True)  # Default
        self.controller = PIController(real_time=True)  # Default
        self.planner = TentaclePlanner()  # Default
        self.ultrasonic = UltrasonicSensor(num_samples=20)  # Default
        self.servo = ServoController()
        self.camera = self.open_camera()
        self.ball_detector = TennisBallDetector(collection_zone=(200, 150, 400, 350))



    def run(self):
        rotate_at_start_angle = np.pi/2
        track_rotate_angle_start = 0
        self.mode = 'rotate_at_start' # 'rotate_at_start', 'search', 'drive', 'stamp', 'deposit', 'align', 'scan'

        try:
            # Main loop and logic
            while True:

                if self.mode == 'rotate_at_start':
                    # Rotate the robot at the start
                    track_rotate_angle_start += self.robot.x
                    # Rotate the robot
                    if track_rotate_angle_start >= rotate_at_start_angle:
                        self.robot.pose_update(0, 0)
                        self.mode = 'search'
                        self.start_time = time.time()
                        print("Starting to drive")
                    else:
                        self.robot.pose_update(-50, 50)

                        rotate_angle = np.pi / 2
                        print("Starting to rotate")

                if self.start_time is None:
                    self.start_time = time.time()
                if rotate_at_start:
                    rotate_angle = np.pi / 2
                    rotate_at_start = False

                # Get the current command from the command queue




                    # print("### Orchestrator: in STOP command")
                    self.robot.pose_update(0, 0)
                    self.log_data(0,0,0,0,
                        Position(self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['z'], PositionTypes.ROBOT)
                    )
                    self.mark_command_done()


                elif self.mode == 'search':
                    # print("### Orchestrator: in DRIVE command")
                    # Get the robot's goal position from the shared goal_position cmd_queue
                    goal = self.get_latest_goal()
                    res = self.movement(goal.x, goal.y, goal.th)
                    self.log_data(
                        res['wl_desired'],
                        res['wr_desired'],
                        res['duty_cycle_l'],
                        res['duty_cycle_r'],
                        goal
                    )

                    # If the goal has been reached, mark the command as done
                    if self.is_goal_reached(goal): # TODO: position type-aware is_goal_reached IS NOT IMPLEMENTED YET
                        self.mark_command_done()

                elif self.mode == 'drive':
                    # print("### Orchestrator: in STAMP command")
                    # Stop the robot and collect the ball
                    self.robot.pose_update(0, 0) # stop the robot if it isn't already
                    self.servo.stamp()  # Activate the collection mechanism
                    self.mark_command_done()

                elif self.mode == 'stamp':

                elif self.mode == "deposit":

                elif self.mode == "align":

                elif self.mode == "scan":
                    # print("### Orchestrator: in SCAN command")
                    # Get the next scan point
                    goal = self.scan_points[self.curr_scan_point]
                    res = self.movement(goal[0], goal[1], 0)

                    # If the goal has been reached, mark the command as done
                    goal_check = Position(goal[0], goal[1], 0, PositionTypes.SCAN_POINT)
                    if self.is_goal_reached(goal_check):
                        self.curr_scan_point += 1
                        if self.curr_scan_point >= len(self.scan_points):
                            self.curr_scan_point = 0
                            self.mode = 'search'
                            print("Finished scanning. Returning to search mode.")
                        self.mark_command_done()

                elif command == RobotCommands.DEPOSIT:
                    # print("### Orchestrator: in DEPOSIT command")
                    # We're now depositing the balls, so insert servo control logic here
                    self.robot.pose_update(0, 0)
                    self.servo.open_door()
                    self.log_data(
                        0,
                        0,
                        0,
                        0,
                        Position(self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['z'], PositionTypes.ROBOT)
                    )
                    self.mark_command_done()

                elif command == RobotCommands.ALIGN:
                    # print("### Orchestrator: in ALIGN command")

                    # This should be run after the box has been detected, now aligning the robot via ultrasonic sensors
                    #                     self.robot.set_motor_speed(0, 0)

                    # Once in desired location to begin scanning the drop off box
                    check_return = self.ultrasonic.check_alignment(depot_distance_threshold=15, alignment_tolerance=1.2)
                    print(f'woop: {check_return}')
                    if check_return[0] == 'distance':
                        # Drive set distance in a straight line
                        distance = check_return[1]
                        print(distance)
                        self.robot.pose_update(1,1)


                    elif check_return[0] == 'rotate':
                        # Rotate on the spot by the desired angle
                        angle_rad = check_return[1]
                        print(angle_rad)
                        self.movement(self.robot.x,self.robot.y,self.robot.th + angle_rad )

                    elif check_return == 'arrived':
                        # In the robot state cmd_queue, notify the cmd_queue itself that the object has been processed
                        self.mark_command_done()

                    self.log_data(
                        0,
                        0,
                        0,
                        0,
                        Position(self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['z'], PositionTypes.ROBOT)
                    )

                elif command == RobotCommands.ROTATE:

                    # The RobotCommands.ROTATE means that the robot should rotate on the spot to scan. For now, we always do 360 degrees.
                    # If we wanted to specify a specific angle, we would pass that in the command data, OR:
                    #    we add an additional position type, like PositionTypes.ROTATE, and set the desired angular rotation as the .th (angle) in the Position object.
                    #    then, inside this method, we just use the angle from the position object, and not care about the Position.x and Position.y

                    # TODO: Insert rotate logic here!!!
                    # Set the angle to rotate by (72 degrees in radians)
                    angle_to_rotate = (2 * np.pi) / 5  # 72-degree rotation

                    # On the first call, initialize the starting angle and target angle
                    if self.starting_angle == None:
                        self.starting_angle = self.robot.th  # Store the initial angle
                        self.theta_target = (self.starting_angle + angle_to_rotate)
                        print(f"Starting rotation. Initial theta: {np.rad2deg(self.starting_angle):.2f} degrees, Target theta: {np.rad2deg(self.theta_target):.2f} degrees")

                    # Check if the current angle has reached the target (with tolerance)
                    current_angle = self.robot.th
                    if abs(current_angle - self.theta_target) > 0.01:  # Tolerance of 0.01 radians (~0.57 degrees)
                        # Rotate by setting the motor speeds for in-place rotation
                        self.robot.pose_update(-0.5, 0.5)

                        # Debugging output to show the progress
                        print(f"Rotating... Current angle: {np.rad2deg(current_angle):.2f} degrees")

                    else:
                        # Stop the robot when the target angle is reached
                        self.robot.pose_update(0, 0)
                        print("Rotation complete.")

                        # Clear the starting angle and theta_target attributes for future rotations
                        self.starting_angle = None
                        self.theta_target = None

                        # Mark the command as done
                        self.mark_command_done()



                # Sleep for 0.1s before the next iteration
                time.sleep(self.dt)


        except KeyboardInterrupt:
            print("Keyboard interrupt. Stopping robot.")

        except Exception as e:
            print(f"Error in Orchestrator, Stopping robot: {e}")
            self.robot.set_motor_speed(0, 0)
        return


    #### PLACEHOLDER PLACEHOLDER PLACEHOLDER PLACEHOLDER NEED TO EDIT
    def is_goal_reached(self, goal: Position) -> bool:
        x, y, th = self.robot.pose

        # TODO: This function needs to be aware of the Position type. If it's the box, then it needs to be closer. If it's a ball, it needs to be 10-14 inches behind. etc.
        #
        # there's a is_goal_reached function in orchestrator_mp that should check this using different criteria depending on the type (PositionTypes) of the goal (Position) that's currently set (set in goal_position, which is the shared multiprocessing.manager.dict()).
        #
        # For example: If it's a ball, we need to make sure we're 8-14 inches behind it but also pointing towards it.
        # If it's a box, we need to make sure that we're close enough for the ultrasonic sensors to take over before returning True. I assume this would be like max 30cm or something.
        # If it's a scan point or a ball position, then we apply the same type of check - are we close enough given some sort of tolerance
        distance_to_goal = np.hypot(goal.x - self.robot.x, goal.y - self.robot.y)
        angle_to_goal = np.arctan2(goal.y - self.robot.y, goal.x - self.robot.x) - self.robot.th # TODO: MAKE SURE THIS CALCULATION IS RIGHT

        if goal.type == PositionTypes.BALL:
            # PLACEHOLDER PLACEHOLDER PLACEHOLDER. DEFINITELY NOT FINAL. WE NEED TO CONSIDER THE ANGLE TOO.
            # THESE SHOULD NOT BE MAGIC NUMBERS. THEY ALSO NEED TO BE LESS THAN THE TOLERANCES IN TENTACLEPLANNER
            if distance_to_goal < 0.12:  # 35cm
                return True

        elif goal.type == PositionTypes.SCAN_POINT:
            # PLACEHOLDER PLACEHOLDER PLACEHOLDER. DEFINITELY NOT FINAL
            if distance_to_goal < 0.12:
                return True

        elif goal.type == PositionTypes.BOX:
            # PLACEHOLDER PLACEHOLDER PLACEHOLDER. DEFINITELY NOT FINAL
            if distance_to_goal < 0.12:
                return True

        return False

    def get_dt(self):
        now = time.time()
        if self.last_update is None:
            self.last_update = now
            return self.dt

        dt = now - self.last_update
        self.last_update = now
        return dt

    def _is_within_bounds(self, detection: BallDetection):
        x, y = detection.x, detection.y
        xmin, xmax, ymin, ymax = self.quadrant_bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def open_camera(self):
        MAX_IND = 3
        camera_idxs = [self.default_camera_idx] + [x for x in range(MAX_IND) if x != self.default_camera_idx]
        for idx in camera_idxs:
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                self.camera = camera
                self.default_camera_idx = idx
                print(f"VisionRunner: Camera opened successfully using idx {idx}")
                return True

        print(f"VisionRunner: Error: Could not open USB camera. Tried {camera_idxs}")
        return False

    def movement(self, x, y, th):
        inputs = self.planner.get_control_inputs(x, y, th, *self.robot.pose, strategy='tentacles')
        linear_vel, angular_vel = inputs['linear_velocity'], inputs['angular_velocity']
        goal_reached = inputs['goal_reached']
        # if goal_reached: print(f"Goal reached! x:{goal.x}, y:{goal.y}, th:{goal.th}")

        duty_cycle_l, duty_cycle_r, wl_desired, wr_desired = self.controller.drive(
            linear_vel,
            angular_vel,
            self.robot.wl,
            self.robot.wr
        )

        # print(f"\nGoal: {goal.x}, {goal.y}, {goal.th}, Current: {self.robot.x}, {self.robot.y}, {self.robot.th}")
        # Apply the duty cycles to the robot wheels
        # print(f"Duty Cycle: {duty_cycle_l}, {duty_cycle_r}\n\n")
        self.robot.pose_update(duty_cycle_l, duty_cycle_r)

        return {
            "goal_reached": goal_reached,
            "inputs": inputs,
            "duty_cycle_l": duty_cycle_l,
            "duty_cycle_r": duty_cycle_r,
            "wl_desired": wl_desired,
            "wr_desired": wr_desired
        }

    def get_latest_goal(self) -> Optional[Position]:
        res = self.goal_position.get('goal')
        if self.debug:
            # print(f"************ Orchestrator: Got goal: {res}")
            pass
        return res

    # Gets the current command from the robot_command_q. It always gets the latest command.
    # If you don't want orchestrator to do something, then don't issue anything. It will do as it's told.
    def get_command(self) -> Optional[RobotCommands]:
        if not self.robot_command_q.empty():
            self.current_command, self.curr_command_id = self.robot_command_q.get()
            # print(f"--#### Orchestrator: got NEW command {self.current_command}")
        return self.current_command

    # Marks the current command as done in the robot_command_q.
    def mark_command_done(self):
        if self.curr_command_id is not None:
            cmd_name = self.robot_command_q.get_data(self.curr_command_id)
            self.robot_command_q.mark_done(self.curr_command_id)
            self.curr_command_id = None
            self.current_command = None
            # print(f"----- Orchestrator: command {cmd_name} marked DONE")

    def print_process(self):
        # Get the current process ID
        pid = os.getpid()
        # Get the CPU core this process is running on
        process = psutil.Process(pid)
        print(f"Orchestrator Process (PID: {pid}) running with: {process.num_threads()} threads")
