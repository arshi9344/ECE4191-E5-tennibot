
import time
import threading
import numpy as np
import json
import random
import multiprocessing as mp
import logging
import queue
import os
from robot_core.control.PI_controller import PIController
from robot_core.motion.tentacle_planner import TentaclePlanner
from robot_core.hardware.simulated_diff_drive_robot import DiffDriveRobot
from robot_core.utils.logging_utils import setup_logging
from robot_core.orchestration.scan_point_utils import ScanPointGenerator
from robot_core.coordinator.robot_states import RobotStates
"""

"""

class Orchestrator(mp.Process):
    def __init__(
            self,
            shared_data,
            goal_position,
            robot_pose,
            log_queue,
            robot=None,
            controller=None,
            planner=None,
            dt=None
    ):
        super().__init__()
        setup_logging(log_queue)

        self.logger = logging.getLogger(f'{__name__}.Orchestrator')
        print(f"Logger name: {self.logger.name}")
        print(f"Logger level: {self.logger.level}")
        print(f"Logger handlers: {self.logger.handlers}")
        print(f"Logger parent: {self.logger.parent}")

        self.shared_data = shared_data # Shared data like robot pose (updated by orchestrator) and also motion state (read by orchestrator)
        self.goal_position = goal_position # Consumed (read) by Orchestrator to adjust robot's pose
        self.robot_pose = robot_pose # Updated by Orchestrator


        self.last_update = None
        # Initialising subcomponents (robot, controller, planner)
        self.logger.info(f"Initialising Orchestrator:")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")

        if not dt:
            self.dt = 0.1 # Everything runs at 0.1s. The TentaclePlanner and PIController are also run at this interval.
        else:
            self.dt = dt
        self.logger.info(f"    Using dt={self.dt}")

        reality = 'real' if robot else 'simulated'
        if not robot:
            self.robot = DiffDriveRobot(real_time=True)
        else:
            self.robot = robot
        self.logger.info(f"    Initialised {reality} robot.")


        if not controller:
            self.controller = PIController(real_time=True) # Default
        else:
            self.controller = controller
        controller_default = 'default' if controller else 'supplied'
        self.logger.info(f"    Initialised {controller_default} controller: Kp: {self.controller.Kp:.2f}, Ki: {self.controller.Ki:.2f}")


        if not planner:
            self.planner = TentaclePlanner(max_linear_velocity=0.2, max_angular_velocity=2) # Default
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

        print(f"Logger name: {self.logger.name}")
        print(f"Logger level: {self.logger.level}")
        print(f"Logger handlers: {self.logger.handlers}")
        print(f"Logger parent: {self.logger.parent}")

    def get_dt(self):
        now = time.time()
        if self.last_update is None:
            self.last_update = now
            return self.dt

        dt = now - self.last_update
        self.last_update = now
        return dt

    def run(self):
        try:
            wl_desired, wr_desired, duty_cycle_l, duty_cycle_r, goal_x, goal_y, goal_th = None, None, None, None, None, None, None
            counter = 0
            while self.shared_data['running']:

                counter += 1

                if self.shared_data['motion_state'].get() == RobotStates.DRIVE:
                    self.logger.setLevel(logging.DEBUG)  # or logging.INFO

                    # print(f"Orchestrator running {counter}. dt = {self.get_dt():.2f}. Time: {time.time():.2f}")
                    self.logger.info(f"Orchestrator running {counter}. dt = {self.get_dt():.2f}. Time: {time.time():.2f}")

                    # Get the robot's goal position from the shared goal_position dict
                    goal_x, goal_y, goal_th = self.goal_position['x'], self.goal_position['y'], self.goal_position['th']
                    # Calculate control inputs (robot base linear and angular velocities) using the planner
                    inputs = self.planner.get_control_inputs(goal_x, goal_y, goal_th, *self.robot.pose, strategy='tentacles')
                    # Calculate the duty cycles for the left and right wheels using the controller
                    linear_vel, angular_vel  = inputs['linear_velocity'], inputs['angular_velocity']
                    duty_cycle_l, duty_cycle_r, wl_desired, wr_desired = self.controller.drive(
                        linear_vel,
                        angular_vel,
                        self.robot.wl_smoothed,
                        self.robot.wr_smoothed
                    )
                    # Apply the duty cycles to the robot wheels
                    self.robot.pose_update(duty_cycle_l, duty_cycle_r)



                elif self.shared_data['motion_state'].get() == RobotStates.STOP:
                    self.robot.set_motor_speed(0, 0)

                elif self.shared_data['motion_state'].get() == RobotStates.COLLECT:
                    # We're now collecting a ball, so insert servo control logic here
                    self.robot.set_motor_speed(0, 0)

                elif self.shared_data['motion_state'].get() == RobotStates.DEPOSIT:
                    # We're now depositing the balls, so insert servo control logic here
                    self.robot.set_motor_speed(0, 0)


                # Logging everything
                data = {
                    'pose': self.robot.pose,
                    'current_wheel_w': (self.robot.wl, self.robot.wr),
                    'target_wheel_w': (wl_desired, wr_desired),
                    'duty_cycle_commands': (duty_cycle_l, duty_cycle_r),
                    'goal_position': (goal_x, goal_y, goal_th),

                }
                # print(data)
                self.logger.info("Attempting to log data...")

                self.logger.info(json.dumps(data))
                # print(json.dumps(data))

                # Updating globally shared robot pose
                self.robot_pose.update({
                    'x': self.robot.x,
                    'y': self.robot.y,
                    'th': self.robot.th
                })

                # Sleep for 0.1s before the next iteration
                time.sleep(self.dt)

            # We only reach this point if the shared_data['running'] flag is False
            self.logger.info("Orchestrator stopping, running Flag is false")
            self.robot.set_motor_speed(0, 0)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt. Stopping robot.")
            self.robot.set_motor_speed(0, 0)

        except Exception as e:
            self.logger.error(f"Error in Orchestrator, Stopping robot: {e}")
            self.robot.set_motor_speed(0, 0)
