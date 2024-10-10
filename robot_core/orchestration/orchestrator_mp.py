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

from matplotlib import pyplot as plt

from robot_core.control.PI_controller import PIController
from robot_core.motion.tentacle_planner import TentaclePlanner
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

class Orchestrator(mp.Process):
    def __init__(
            self,
            running, # the running flag, which is mp.manager.Value('b', True) initalised in ProcessCoordinator
            robot_command_q : StatefulCommandQueue, # the robot command queue, which is a StatefulCommandQueue object initalised in ProcessCoordinator
            goal_position, # the goal position, which is a mp.manager.dict initalised in ProcessCoordinator
            robot_pose, # the robot pose, which is a mp.manager.dict initalised in ProcessCoordinator
            robot_graph_data, # the robot graph data, which is a mp.manager.list initalised in ProcessCoordinator
            log_queue, # the log queue, which is a mp.Queue
            simulated_robot, # boolean, whether the robot is simulated or not. True or False
            controller=None,
            planner=None,
            dt=None,
            log=False,
            debug=False
    ):
        super().__init__()
        if log: setup_logging(log_queue)

        self.logger = logging.getLogger(f'{__name__}.Orchestrator')
        print(f"Logger name: {self.logger.name}")
        print(f"Logger level: {self.logger.level}")
        print(f"Logger handlers: {self.logger.handlers}")
        print(f"Logger parent: {self.logger.parent}")

        ##### Shared Data (all mp.manager objects or wrappers for them) #####
        self.running = running  # mp.manager.Value('b', True)
        self.robot_command_q : StatefulCommandQueue = robot_command_q  # StatefulCommandQueue object
        self.goal_position = goal_position  # Dict, Consumed (read) by Orchestrator to adjust robot's pose
        self.robot_pose = robot_pose  # Updated by Orchestrator
        self.robot_graph_data = robot_graph_data  # Updated by Orchestrator

        #### Internal state variables
        self.last_update = None
        self.start_time = None
        self.curr_command_id = None
        self.current_command = None
        self.simulated_robot: bool = simulated_robot  # boolean
        self.debug = debug
        self.starting_angle = None
        self.theta_target = None

        self.logger.info(f"Initialising Orchestrator:")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")
        # print(f"Process ID: {os.getpid()} - Running worker: {self.name}")
        if not dt:
            self.dt = 0.1  # Everything runs at 0.1s. The TentaclePlanner and PIController are also run at this interval.
        else:
            self.dt = dt
        self.logger.info(f"    Using dt={self.dt}")

        ###### Subcomponents ######
        # These are initialised inside run()
        self.robot = None
        self.ultrasonic = None
        self.servo = None

        if not controller:
            self.controller = PIController(real_time=True)  # Default
        else:
            self.controller = controller
        controller_default = 'default' if controller else 'supplied'
        self.logger.info(
            f"    Initialised {controller_default} controller: Kp: {self.controller.Kp:.2f}, Ki: {self.controller.Ki:.2f}")

        if not planner:
            self.planner = TentaclePlanner()  # Default
        else:
            self.planner = planner
        planner_default = 'default' if planner else 'supplied'
        self.logger.info(f"    Initialised {planner_default} planner:")
        self.logger.info(f"        Max linear velocity: {self.planner.max_linear_velocity:.2f}")
        self.logger.info(f"        Max linear acceleration: {self.planner.max_acceleration:.2f}")
        self.logger.info(f"        Max linear tolerance: {self.planner.max_linear_tolerance:.2f}")
        self.logger.info(f"        Max angular velocity: {self.planner.max_angular_velocity:.2f}")
        self.logger.info(f"        Max angular acceleration: {self.planner.max_angular_acceleration:.2f}")
        self.logger.info(f"        Max angular tolerance: {self.planner.max_angular_tolerance:.2f}")

        # print(f"Logger name: {self.logger.name}")
        # print(f"Logger level: {self.logger.level}")
        # print(f"Logger handlers: {self.logger.handlers}")
        # print(f"Logger parent: {self.logger.parent}")

    def get_dt(self):
        now = time.time()
        if self.last_update is None:
            self.last_update = now
            return self.dt

        dt = now - self.last_update
        self.last_update = now
        return dt

    def movement(self, x, y, th):
        # Calculate control outputs (robot base linear and angular velocities) using the planner
        inputs = self.planner.get_control_inputs(x, y, th, *self.robot.pose, strategy='tentacles')
        # Calculate the duty cycles for the left and right wheels using the controller
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

    def run(self):
        try:
            ### Initialisation ###
            self.print_process()
            self.start_time = time.time()

            # Initialise Robot and Ultrasonic sensors
            if not self.simulated_robot:
                reality = 'real'
                from robot_core.hardware.diff_drive_robot import DiffDriveRobot
                self.robot = DiffDriveRobot(0.03, real_time=True)

                from robot_core.perception.ultrasonic_sensors import UltrasonicSensor
                self.ultrasonic = UltrasonicSensor(num_samples=20)

                from robot_core.hardware.servo_controller import ServoController
                self.servo = ServoController()

                self.servo.stamp()

            else:
                reality = 'simulated'
                from robot_core.hardware.simulated_diff_drive_robot import DiffDriveRobot
                self.robot = DiffDriveRobot(0.03, real_time=True)

            print(f"    Initialised {reality} robot.")

            # Main loop and logic
            while self.running.value:
                # self.logger.setLevel(logging.DEBUG)  # or logging.INFO
                # print(f"Orchestrator running. dt = {self.get_dt():.2f}. Time: {time.time():.2f}")
                command = self.get_command()

                if command == RobotCommands.STOP:
                    # print("### Orchestrator: in STOP command")
                    self.robot.pose_update(0, 0)
                    self.log_data(0,0,0,0,
                        Position(self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['z'], PositionTypes.ROBOT)
                    )
                    self.mark_command_done()


                elif command == RobotCommands.DRIVE:
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

                elif command == RobotCommands.STAMP:
                    # print("### Orchestrator: in STAMP command")
                    # Stop the robot and collect the ball
                    self.robot.pose_update(0, 0) # stop the robot if it isn't already
                    self.servo.stamp()  # Activate the collection mechanism
                    ## NEED A WAY TO CHECK IF BALL IS STILL PRESENT TO CONTROL IF ITS REMOVED FROM QUEUE (in process coordinator)
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

                    elif check_return[0] == 'arrived':
                        # In the robot state cmd_queue, notify the cmd_queue itself that the object has been processed
                        distance = check_return[1] - 2
                        self.movement(self.robot.x + distance,self.robot.y,self.robot.th)
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
                    angle_to_rotate = (2 * np.pi) / 8  # 45-degree rotation

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


                # Updating globally shared robot pose
                self.robot_pose.update({
                    'x': self.robot.x,
                    'y': self.robot.y,
                    'th': self.robot.th
                })

                # Sleep for 0.1s before the next iteration
                if not self.simulated_robot:
                    time.sleep(self.dt)
                else:
                    time.sleep(self.dt / 20)

            # We only reach this point if the shared_data['running'] flag is False
            self.logger.info("Orchestrator stopping, running Flag is false")
            if self.robot is not None and not self.simulated_robot:
                self.robot.set_motor_speed(0, 0)
            self.robot.set_motor_speed(0, 0)
            return

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt. Stopping robot.")

        except Exception as e:
            self.logger.error(f"Error in Orchestrator, Stopping robot: {e}")
        if self.robot is not None and not self.simulated_robot:
            self.robot.set_motor_speed(0, 0)
        return

    def log_data(self, wl_desired, wr_desired, duty_cycle_l, duty_cycle_r, goal: Position):
        # Logging everything
        log_point = RobotLogPoint(
            pose=self.robot.pose,
            current_wheel_w=(self.robot.wl, self.robot.wr),
            target_wheel_w=(wl_desired, wr_desired),
            duty_cycle_commands=(duty_cycle_l, duty_cycle_r),
            goal_position=goal,
            time=time.time()
        )
        self.robot_graph_data.append(log_point)

    #### PLACEHOLDER PLACEHOLDER PLACEHOLDER PLACEHOLDER NEED TO EDIT
    def is_goal_reached(self, goal: Position) -> bool:
        x, y, th = self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['th']

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
            # when driving to the ball we should first angle ourselves appropriately before driving straight, everything seems to be handled easier without tentacle planner but idk
            # otherwise we will have to re-allign afterwards
            # THESE SHOULD NOT BE MAGIC NUMBERS. THEY ALSO NEED TO BE LESS THAN THE TOLERANCES IN TENTACLEPLANNER
            if distance_to_goal > 0.22 or distance_to_goal <0.36:  # anywhere from 22cm to 36cm
                return True

        elif goal.type == PositionTypes.SCAN_POINT:
            # PLACEHOLDER PLACEHOLDER PLACEHOLDER. DEFINITELY NOT FINAL
            if distance_to_goal < 0.01:
                return True

        elif goal.type == PositionTypes.BOX:
            # PLACEHOLDER PLACEHOLDER PLACEHOLDER. DEFINITELY NOT FINAL
            # once the
            if distance_to_goal < 0.12:
                return True

        return False


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