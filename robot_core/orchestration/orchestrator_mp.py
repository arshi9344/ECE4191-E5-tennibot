
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

from matplotlib import pyplot as plt

from robot_core.control.PI_controller import PIController
from robot_core.motion.tentacle_planner import TentaclePlanner
from robot_core.utils.logging_utils import setup_logging
from robot_core.orchestration.scan_point_utils import ScanPointGenerator
from robot_core.coordinator.robot_states import RobotStates
from robot_core.utils.position_data import PositionData
"""

"""

class Orchestrator(mp.Process):
    def __init__(
            self,
            shared_data,
            goal_position_q,
            robot_pose,
            robot_graph_data,
            log_queue,
            simulated_robot, # boolean
            robot=None,
            controller=None,
            planner=None,
            dt=None,
            log=False
    ):
        super().__init__()
        if log: setup_logging(log_queue)

        self.logger = logging.getLogger(f'{__name__}.Orchestrator')
        print(f"Logger name: {self.logger.name}")
        print(f"Logger level: {self.logger.level}")
        print(f"Logger handlers: {self.logger.handlers}")
        print(f"Logger parent: {self.logger.parent}")

        self.shared_data = shared_data # Shared data like robot pose (updated by orchestrator) and also motion state (read by orchestrator)
        self.goal_position_q = goal_position_q # Queue, Consumed (read) by Orchestrator to adjust robot's pose
        self.robot_pose = robot_pose # Updated by Orchestrator
        self.robot_graph_data = robot_graph_data # Updated by Orchestrator

        self.last_update = None
        self.start_time = None
        self.simulated_robot : bool = simulated_robot # boolean
        # Initialising subcomponents (robot, controller, planner)
        self.logger.info(f"Initialising Orchestrator:")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")
        # print(f"Process ID: {os.getpid()} - Running worker: {self.name}")
        if not dt:
            self.dt = 0.1 # Everything runs at 0.1s. The TentaclePlanner and PIController are also run at this interval.
        else:
            self.dt = dt
        self.logger.info(f"    Using dt={self.dt}")

        self.robot = None
        # reality = 'real' if robot else 'simulated'
        # if not robot:
        #     self.robot = DiffDriveRobot(0.03, real_time=True)
        # else:
        #     self.robot = robot
        # self.logger.info(f"    Initialised {reality} robot.")


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

    def run(self):
        try:
            print("Inside robot run")
            self.print_process()
            self.start_time = time.time()
            wl_desired, wr_desired, duty_cycle_l, duty_cycle_r, goal_x, goal_y, goal_th = None, None, None, None, None, None, None
            counter = 0
            goal = PositionData(0, 0, 0, False)

            # Initialise Robot
            if not self.simulated_robot:
                    print("not simulated robot, initalising real")
                    reality = 'real'
                    from robot_core.hardware.diff_drive_robot import DiffDriveRobot
                    self.robot = DiffDriveRobot(0.03, real_time=True)
            else:
                reality = 'simulated'
                from robot_core.hardware.simulated_diff_drive_robot import DiffDriveRobot
                self.robot = DiffDriveRobot(0.03, real_time=True)

            print(f"    Initialised {reality} robot.")

            while self.shared_data['running']:
                counter += 1
                print("inside running loop")

                if self.shared_data['robot_state'].get() == RobotStates.DRIVE:
                    self.logger.setLevel(logging.DEBUG)  # or logging.INFO

                    # print(f"Orchestrator running {counter}. dt = {self.get_dt():.2f}. Time: {time.time():.2f}")
                    self.logger.info(f"Orchestrator running {counter}. dt = {self.get_dt():.2f}. Time: {time.time():.2f}")

                    # Get the robot's goal position from the shared goal_position dict
                    goal = self.get_latest_goal(goal)
                    # print(f"Goal Position: {goal_x:.2f}, {goal_y:.2f}, {goal_th:.2f}, Robot Pose: {self.robot.x:.2f}, {self.robot.y:.2f}, {self.robot.th:.2f}")
                    # Calculate control inputs (robot base linear and angular velocities) using the planner
                    inputs = self.planner.get_control_inputs(goal.x, goal.y, goal.th, *self.robot.pose, strategy='tentacles')
                    # Calculate the duty cycles for the left and right wheels using the controller
                    linear_vel, angular_vel  = inputs['linear_velocity'], inputs['angular_velocity']
                    duty_cycle_l, duty_cycle_r, wl_desired, wr_desired = self.controller.drive(
                        linear_vel,
                        angular_vel,
                        self.robot.wl,
                        self.robot.wr
                    )
                    # Apply the duty cycles to the robot wheels
                    self.robot.pose_update(duty_cycle_l, duty_cycle_r)
                    # self.robot.pose_update(90, 90)



                elif self.shared_data['robot_state'].get() == RobotStates.STOP:
#                     self.robot.set_motor_speed(0, 0)
                    pass

                elif self.shared_data['robot_state'].get() == RobotStates.COLLECT:
                    # We're now collecting a ball, so insert servo control logic here
#                     self.robot.set_motor_speed(0, 0)
                    pass

                elif self.shared_data['robot_state'].get() == RobotStates.DEPOSIT:
                    # We're now depositing the balls, so insert servo control logic here
#                     self.robot.set_motor_speed(0, 0)
                    pass


                # Logging everything
                data = {
                    'pose': self.robot.pose,
                    'current_wheel_w': (self.robot.wl, self.robot.wr),
                    'target_wheel_w': (wl_desired, wr_desired),
                    'duty_cycle_commands': (duty_cycle_l, duty_cycle_r),
                    'goal_position': (goal.x, goal.y, goal.th),

                }
                # print(data)
                self.logger.info(json.dumps(data))
                self.robot_graph_data.append(data)
                # print(self.robot_graph_data[-1])
                # print(json.dumps(data))

                # Updating globally shared robot pose
                self.robot_pose.update({
                    'x': self.robot.x,
                    'y': self.robot.y,
                    'th': self.robot.th
                })

                # Sleep for 0.1s before the next iteration
                time.sleep(self.dt)
                # time.sleep(1)

            # We only reach this point if the shared_data['running'] flag is False
            self.logger.info("Orchestrator stopping, running Flag is false")
            self.robot.set_motor_speed(0, 0)
            return

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt. Stopping robot.")

        except Exception as e:
            self.logger.error(f"Error in Orchestrator, Stopping robot: {e}")
        if self.robot is not None:
            self.robot.set_motor_speed(0, 0)
        return

    def get_latest_goal(self, current_goal):
        try:
            res = self.goal_position_q.get_nowait()
        except Empty:
            # If queue is empty, return the current goal (no change)
            res = current_goal
        return res

    def print_process(self):
        # Get the current process ID
        pid = os.getpid()
        # Get the CPU core this process is running on
        process = psutil.Process(pid)
        print(f"Orchestrator Process (PID: {pid}) running with: {process.num_threads()} threads")

    def update_plot(self, fig, axes, clear_output=False):
        # assert that fig and azes are a subplot of 2x2
        plt.ion()
        plt.close(fig)

        if fig is None or axes is None:
            raise ValueError("Please provide a figure and axes to update the plot, e.g. \nfig, axes = plt.subplots(2, 2, figsize=(12, 10))")
        if isinstance(self.robot_graph_data, type(None)):
            return
        try:
            if len(self.robot_graph_data) == 0:
                return
        except TypeError:
            print("Inside update_plot, robot graph data is none.")
            return

        plt.ion()
        # axes_flat = axes.flatten()  # Flatten the 2D array of axes
        for ax in axes:
            ax.clear()

        data = self.robot_graph_data
        # Plot 1: Robot path and orientation
        poses = np.array([ele['pose'] for ele in data])
        if len(poses) > 0:
            axes[0].clear()
            axes[0].plot(np.array(poses)[:, 0], np.array(poses)[:, 1])
            x, y, th = poses[-1]
            axes[0].plot(x, y, 'k', marker='+')
            axes[0].quiver(x, y, 0.1 * np.cos(th), 0.1 * np.sin(th))
        axes[0].set_xlabel('x-position (m)')
        axes[0].set_ylabel('y-position (m)')
        axes[0].set_title(
            f"Robot Pose Over Time. Kp: {self.controller.Kp}, Ki: {self.controller.Ki}")
        axes[0].axis('equal')
        axes[0].grid()

        # Plot 2: Duty cycle commands
        duty_cycle_commands = np.array([ele['duty_cycle_commands'] for ele in data])
        if len(duty_cycle_commands) > 0:
            axes[1].clear()
            duty_cycle_commands = np.array(duty_cycle_commands)
            axes[1].plot(duty_cycle_commands[:, 0], label='Left Wheel')
            axes[1].plot(duty_cycle_commands[:, 1], label='Right Wheel')

        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Duty Cycle')
        axes[1].set_title('Duty Cycle Commands Over Time')
        axes[1].legend()
        axes[1].grid()

        # Plot 3: Wheel velocities
        velocities = np.array([ele['current_wheel_w'] for ele in data])
        desired_velocities = np.array([ele['target_wheel_w'] for ele in data])
        if len(velocities) > 0 and len(desired_velocities) > 0:
            axes[2].clear()
            axes[2].plot(velocities[:, 0], label='Left Wheel')
            axes[2].plot(velocities[:, 1], label='Right Wheel')
            axes[2].plot(desired_velocities[:, 0], label='Desired Left Wheel')
            axes[2].plot(desired_velocities[:, 1], label='Desired Right Wheel')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Wheel Velocity (rad/s)')
        axes[2].set_title('Wheel Velocity vs. Time')
        axes[2].legend()
        axes[2].grid()

        # Plot 4: Goal Positions vs. actual position
        goal_positions = np.array([ele['goal_position'] for ele in data])

        # Add (0, 0) to both goal_positions and poses
        goal_positions = np.vstack(((0, 0, 0), goal_positions))
        poses = np.vstack(([0, 0, 0], poses))
        # scan_locations = np.array(orchestrator.scan_locations

        axes[3].clear()
        axes[3].plot(0, 0, 'ko', markersize=10, label='Start (0, 0)')  # Add point at (0, 0)

        if len(poses) > 0:
            axes[3].plot(poses[:, 0], poses[:, 1], 'b-', label='Actual Path')
            axes[3].scatter(poses[:, 0], poses[:, 1], color='b', s=5)  # Add dots for each position with custom size
        if len(goal_positions) > 1:
            axes[3].plot(goal_positions[:, 0], goal_positions[:, 1], 'r--', label='Goal Path')
            axes[3].plot(goal_positions[:, 0], goal_positions[:, 1], 'r.')  # Add dots for each goal position
        # if len(scan_locations) > 1:
        #     axes[1, 1].scatter(scan_locations[:, 0], scan_locations[:, 1], color='g', s=20,
        #                        label='Scan Locations')  # Add dots for each scan position

        axes[3].set_xlabel('x-position (m)')
        axes[3].set_ylabel('y-position (m)')
        if self.start_time is not None:
            duration = time.time() - self.start_time
        else:
            duration = 0
        axes[3].set_title(f"Robot Positions. t={duration:.2f} sec")
        axes[3].axis('equal')
        axes[3].grid(True)
        axes[3].legend()

        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(0.001)

        if clear_output:
            display.clear_output(wait=True)
        plt.show()
        display.display(fig)