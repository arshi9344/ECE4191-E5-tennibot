
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


from matplotlib import pyplot as plt

from robot_core.control.PI_controller import PIController
from robot_core.motion.tentacle_planner import TentaclePlanner
from robot_core.utils.logging_utils import setup_logging
from robot_core.utils.robot_log_point import RobotLogPoint
from robot_core.orchestration.scan_point_utils import ScanPointGenerator
from robot_core.coordinator.robot_states import RobotStates
from robot_core.utils.position import Position, PositionTypes
# from robot_core.perception.ultrasonic_sensors import UltrasonicSensor # moved import inside the Orchestrator.run() method
"""

"""

class Orchestrator(mp.Process):
    def __init__(
            self,
            shared_data,
            goal_position,
            robot_pose,
            robot_graph_data,
            log_queue,
            simulated_robot, # boolean
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

        self.shared_data = shared_data # Shared data like robot pose (updated by orchestrator) and also motion state (read by orchestrator)
        self.goal_position = goal_position # Dict, Consumed (read) by Orchestrator to adjust robot's pose
        self.robot_pose = robot_pose # Updated by Orchestrator
        self.robot_graph_data = robot_graph_data # Updated by Orchestrator

        self.last_update = None
        self.start_time = None
        self.simulated_robot : bool = simulated_robot # boolean
        self.debug = debug
        # Initialising subcomponents (robot, controller, planner)
        self.logger.info(f"Initialising Orchestrator:")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")
        # print(f"Process ID: {os.getpid()} - Running worker: {self.name}")
        if not dt:
            self.dt = 0.1 # Everything runs at 0.1s. The TentaclePlanner and PIController are also run at this interval.
        else:
            self.dt = dt
        self.logger.info(f"    Using dt={self.dt}")

        # These are initialised inside run()
        self.robot = None
        self.ultrasonic = None
        self.servo = None



        if not controller:
            self.controller = PIController(real_time=True) # Default
        else:
            self.controller = controller
        controller_default = 'default' if controller else 'supplied'
        self.logger.info(f"    Initialised {controller_default} controller: Kp: {self.controller.Kp:.2f}, Ki: {self.controller.Ki:.2f}")


        if not planner:
            self.planner = TentaclePlanner() # Default
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
        linear_vel, angular_vel  = inputs['linear_velocity'], inputs['angular_velocity']
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
            self.print_process()
            self.start_time = time.time()
            wl_desired, wr_desired, duty_cycle_l, duty_cycle_r, goal_x, goal_y, goal_th = None, None, None, None, None, None, None
            goal = Position(0, 0, 0, PositionTypes.ROBOT)

            # Initialise Robot and Ultrasonic sensors
            if not self.simulated_robot:
                    reality = 'real'
                    from robot_core.hardware.diff_drive_robot import DiffDriveRobot
                    self.robot = DiffDriveRobot(0.03, real_time=True)

                    from robot_core.perception.ultrasonic_sensors import UltrasonicSensor
                    self.ultrasonic = UltrasonicSensor(num_samples=20)

                    from robot_core.hardware.servo_controller import ServoController
                    self.servo = ServoController()
            else:
                reality = 'simulated'
                from robot_core.hardware.simulated_diff_drive_robot import DiffDriveRobot
                self.robot = DiffDriveRobot(0.03, real_time=True)

            print(f"    Initialised {reality} robot.")

            while self.shared_data['running']:
                # self.logger.setLevel(logging.DEBUG)  # or logging.INFO
                # print(f"Orchestrator running. dt = {self.get_dt():.2f}. Time: {time.time():.2f}")

                if self.shared_data['robot_state'].get() == RobotStates.SEARCH:
                    # Get the robot's goal position from the shared goal_position queue
                    goal = self.get_latest_goal(goal)

                    res = self.movement(goal.x, goal.y, goal.th)

                    self.log_data(
                        res['wl_desired'],
                        res['wr_desired'],
                        res['duty_cycle_l'],
                        res['duty_cycle_r'],
                        goal
                    )

                elif self.shared_data['robot_state'].get() == RobotStates.STOP:
                    self.robot.set_motor_speed(0, 0)
                    self.log_data(
                        0,
                        0,
                        0,
                        0,
                        Position(None, None, None, PositionTypes.ROBOT)
                    )

                elif self.shared_data['robot_state'].get() == RobotStates.COLLECT:
                    # Move towards the collection point
                    goal = self.get_latest_goal(goal)
                    res = self.movement(goal.x, goal.y, goal.th)
                    goal_reached = res['goal_reached']
                    
                    if goal_reached:
                        # Stop the robot and collect the object
                        self.robot.set_motor_speed(0, 0)
                        self.servo.stamp()  # Activate the collection mechanism



                elif self.shared_data['robot_state'].get() == RobotStates.DEPOSIT:
                    # We're now depositing the balls, so insert servo control logic here
                    self.robot.set_motor_speed(0, 0)
                    self.servo.deposit()
                    self.log_data(
                        0,
                        0,
                        0,
                        0,
                        Position(None, None, None, PositionTypes.ROBOT)
                    )

                elif self.shared_data['robot_state'].get() == RobotStates.ALIGN:
                    # This should be run after the box has been detected, now aligning the robot via ultrasonic sensors
#                     self.robot.set_motor_speed(0, 0)


                    # Once in desired location to begin scanning the drop off box
                    check_return = self.ultrasonic.check_alignment(depot_distance_threshold=15, alignment_tolerance=1.2)
                    if check_return[0] == 'distance':
                        # Drive set distance in a straight line
                        distance = check_return[1]
                        self.robot.set_motor_speed(1,1)

                    elif check_return[0] == 'rotate':
                        # Rotate on the spot by the desired angle
                        angle_rad = check_return[1]
                        self.movement(self.robot.x,self.robot.y,self.robot.th + angle_rad )

                    elif check_return == 'arrived':
                        # In the robot state queue, notify the queue itself that the object has been processed
                            pass


                    self.log_data(
                        0,
                        0,
                        0,
                        0,
                        Position(None, None, None, PositionTypes.ROBOT)
                    )




                # print(self.robot_graph_data[-1])
                # print(json.dumps(data))

                # Updating globally shared robot pose
                self.robot_pose.update({
                    'x': self.robot.x,
                    'y': self.robot.y,
                    'th': self.robot.th
                })

                # Sleep for 0.1s before the next iteration
                if not self.simulated_robot: time.sleep(self.dt)
                else: time.sleep(self.dt/20)

            # We only reach this point if the shared_data['running'] flag is False
            self.logger.info("Orchestrator stopping, running Flag is false")
            if self.robot is not None and not self.simulated_robot:
                self.robot.set_motor_speed(0, 0)
            self.robot.set_motor_speed(0, 0)
            return

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt. Stopping robot.")

        except Exception as e:
            self.logger.error(f"Error in Orchestrator, Stopping robot: {traceback.print_exc()}")
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

    def get_latest_goal(self, current_goal) -> Position:
        try:
            res = self.goal_position['goal']
            # print(f"get_latest_goal: {res}")
        except Exception:
            # If queue is empty, return the current goal (no change)
            res = current_goal

        if self.debug and current_goal != res:
            print(f"************ Orchestrator: New goal received! {res}")
        return res

    def print_process(self):
        # Get the current process ID
        pid = os.getpid()
        # Get the CPU core this process is running on
        process = psutil.Process(pid)
        print(f"Orchestrator Process (PID: {pid}) running with: {process.num_threads()} threads")
