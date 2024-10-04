import multiprocessing as mp
from multiprocessing import Manager
import time
import logging
import traceback

import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from robot_core.hardware.diff_drive_robot import DiffDriveRobot
from robot_core.motion.tentacle_planner import TentaclePlanner
from robot_core.control.PI_controller import PIController
from robot_core.coordinator.robot_states import RobotStates, StateWrapper, VisionStates
from robot_core.orchestration.scan_point_utils import ScanPointGenerator, ScanPoint
from robot_core.orchestration.orchestrator_mp import Orchestrator
from robot_core.perception.vision_runner import VisionRunner
from robot_core.utils.logging_utils import setup_logging, create_log_listener
from robot_core.utils.robot_log_point import RobotLogPoint
from robot_core.utils.position import Position, PositionTypes
from robot_core.utils.robot_plotter import RobotPlotter
import matplotlib.pyplot as plt
import os
import psutil
import numpy as np


# TODO: Add function to determine if detected ball is beyond boundaries of court

class Coordinator:
    def __init__(
            self,
            simulate=False,
            live_graphs=False,
            graph_interval=1,
            log=False,
            clear_output=False,
            court_dimensions = (4.12, 5.48),
            debug=False,
            plot_time_window=5
    ):
        self.manager = Manager()
        self.shared_data = {
            'running': self.manager.Value('b', True),
            'robot_state': StateWrapper(self.manager, RobotStates, RobotStates.STOP),
            'vision_state': StateWrapper(self.manager, VisionStates, VisionStates.NONE),
        }
        self.robot_pose = self.manager.dict({'x': 0, 'y': 0, 'th': 0})
        self.goal_position_q = self.manager.Queue() # This should always only contain one or zero Position objects from robot_core.utils.position

        """
        mini-explainer for the goal_position Queue:
            - Orchestrator ALWAYS moves towards the one Position in goal_position. Orchestrator does not care about anything else.
                - If there's nothing inside goal_position Queue, then Orchestrator maintains its current heading.
            - On startup, goal_position will contain a Position for a scanning point. Robot will start moving towards this.
            - If VisionRunner sees a ball, then the Coordinator inserts a Position for ball coordinates into goal_position. 
                - The robot hence starts moving toward the ball.
                - If VisionRunner refines its estimate of the ball's position, then Coordinator inserts this new coord into the queue.
                - Once the ball is collected, Coordinator then inserts the next scanning point into goal_position

        """
        # [[x, y, th], [x, y, th] ]

        # Robot graph data
        # robot_graph_data may ONLY be written to by Orchestrator and read by everything else.
        self.robot_graph_data = self.manager.list([]) # Contains a list of RobotLogPoint objects from robot_core.utils.logging_utils.
        self.graph_interval = graph_interval
        self.last_graph_time = 0
        self.live_graphs = live_graphs
        self.clear_output = clear_output
        self.plotter = None
        if live_graphs: self.plotter = RobotPlotter(max_time_window=plot_time_window)

        # Shared image data. may ONLY be written to by VisionRunner and read by everything else. This is only necessary for plotting.
        self.latest_image = self.manager.dict({'time': None, 'frame': None})

        # Other variables
        self.debug = debug
        self.court_dimensions = court_dimensions
        self.scan_point_generator = ScanPointGenerator(x_lim=4.12, y_lim=5.48, scan_radius=2, flip_x=True, flip_y=False)
        self.scan_points = self.scan_point_generator.points
        self.curr_scan_point = 0
        self.prev_scan_point = None
        # Setting up Logging
        self.log = log
        self.log_queue = self.manager.Queue(-1)
        if self.log:
            self.log_listener = create_log_listener(self.log_queue)
            self.log_listener.start()
        setup_logging(self.log_queue)

        self.logger = logging.getLogger(f'{__name__}.Coordinator')

        # Instantiating child processes
        # Orchestrator
        self.orchestrator = Orchestrator(
            shared_data=self.shared_data,
            goal_position_q=self.goal_position_q,
            robot_pose=self.robot_pose,
            robot_graph_data=self.robot_graph_data,
            log_queue=self.log_queue,
            simulated_robot=simulate,
            # robot=DiffDriveRobot(real_time=True) if not simulate else None,
            log=self.log
        )

        # VisionRunner
        self.vision_runner = VisionRunner(
            shared_data=self.shared_data,
            shared_image=self.latest_image,
            log_queue=self.log_queue,
            log=self.log,
            camera_idx=1,
            use_simulated_video=False
        )

    def start(self):
        self.print_process()
        self.orchestrator.start()
        self.vision_runner.start()

        if self.live_graphs: self.plotter.start()


    def stop(self):
        self.shared_data['running'] = False
        self.orchestrator.join()
        self.vision_runner.join()

        if self.live_graphs: self.plotter.stop()
        if self.log: self.log_listener.stop()

    def run(self):
        self.start() # Starting Orchestrator and VisionRunner

        self.shared_data['robot_state'].set(RobotStates.SEARCH)
        self.shared_data['vision_state'].set(VisionStates.DETECT_BALL)

        # start with curr_scan_point = 0, prev_scan_point = None
        try:
            while self.shared_data['running']:
                """Searching for a ball / moving to scanning point"""
                scan_point_goal = Position(
                    self.scan_points[self.curr_scan_point].x,
                    self.scan_points[self.curr_scan_point].y,
                    self._angle_between_points([self.robot_pose['x'], self.robot_pose['y']],
                                               [self.scan_points[self.curr_scan_point].x,
                                                self.scan_points[self.curr_scan_point].y]),
                    PositionTypes.SCAN_POINT
                )

                if self.prev_scan_point != self.curr_scan_point:
                    self.goal_position_q.put(scan_point_goal) # Insert the new scan point into the goal_position queue
                    if self.debug: print(f"************ New scan point: {scan_point_goal}")

                    if self.prev_scan_point is None: # Means we're at the first scan point
                        self.prev_scan_point = 0
                    else:
                        self.prev_scan_point = self.curr_scan_point


                if self._is_goal_reached(*scan_point_goal.coords, self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['th']):
                    if self.debug: print(f"************ New scan point: Scanning point reached: {scan_point_goal.x}, {scan_point_goal.y}")
                    self.curr_scan_point += 1
                    if self.curr_scan_point >= len(self.scan_points):
                        self.curr_scan_point = 0
                    if self.debug: print(f"Scanning point incremented: {self.curr_scan_point}")


                # print(self.shared_data['robot_state'])

                # time.sleep(5)

                # Graphing
                self.plot() # Update the plot
                # self.logger.info("Coordinator is running")

        except KeyboardInterrupt:
            self.logger.info("Exiting Coordinator")
        except Exception:
            self.logger.error(f"Error in Coordinator, exiting: {traceback.print_exc()}")

        finally:
            # print(self.axes)
            # print(self.fig)
            if self.live_graphs: self.plot()
            # print(self.robot_graph_data)
            plt.ioff()
            plt.show()
            self.stop()

    def plot(self):
        assert self.live_graphs is True, "Cannot plot if live_graphs is False"
        assert self.plotter is not None, "Cannot plot if self.plotter is None"
        if time.time() - self.last_graph_time > self.graph_interval and len(self.robot_graph_data) > 0:
            self.last_graph_time = time.time()
            self.plotter.update_plot(self.robot_graph_data, self.latest_image, clear_output=self.clear_output)
            #TODO: Add vision data into update_plot - make robotPlotter also accept vision data

            # print(f"Length of robot_graph_data: {len(self.robot_graph_data)}")

    def _is_goal_reached(self, goal_x, goal_y, goal_th, x, y, th, max_linear_tolerance=0.1, max_angular_tolerance=0.05):
        distance_to_goal = np.hypot(goal_x - x, goal_y - y)
        angular_error = np.arctan2(np.sin(goal_th - th), np.cos(goal_th - th))
        return distance_to_goal <= max_linear_tolerance and angular_error <= max_angular_tolerance  # and abs(angular_error) <= self.max_angular_tolerance

    def _angle_between_points(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.arctan2(dy, dx)

    def print_process(self):
        # Get the current process ID
        pid = os.getpid()
        # Get the CPU core this process is running on
        process = psutil.Process(pid)
        print(f"Coordinator Process (PID: {pid}) running with: {process.num_threads()} threads")


if __name__ == '__main__':
    coordinator = Coordinator(simulate=True)
    coordinator.run()