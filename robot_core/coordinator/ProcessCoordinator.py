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
from robot_core.coordinator.robot_states import RobotStates, StateWrapper
from robot_core.orchestration.orchestrator_mp import Orchestrator
from robot_core.utils.logging_utils import setup_logging, create_log_listener
import matplotlib.pyplot as plt
import os
import psutil


class Coordinator:
    def __init__(self, simulate=False, fig=None, axes=None, live_graphs=False, graph_interval=0.5, log=False):
        self.manager = Manager()
        self.shared_data = {
            'running': self.manager.Value('b', True),
            'motion_state': StateWrapper(self.manager, RobotStates.STOP),
            'detected_objects': self.manager.list() # not used for now
        }
        self.robot_pose = self.manager.dict({'x': 0, 'y': 0, 'th': 0})
        self.goal_position = self.manager.dict({'x': 0, 'y': 0, 'th': 0})

        # Robot graph data
        self.robot_graph_data = self.manager.list([])
        self.graph_interval = graph_interval
        self.fig = fig
        self.axes = axes
        self.live_graphs = live_graphs

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
            goal_position=self.goal_position,
            robot_pose=self.robot_pose,
            robot_graph_data=self.robot_graph_data,
            log_queue=self.log_queue,
            robot=DiffDriveRobot(real_time=True) if not simulate else None
        )

        # VisionRunner
        # Insert this here

    def start(self):
        self.orchestrator.start()
        # self.vision_runner.start()

    def stop(self):
        self.shared_data['running'] = False
        self.orchestrator.join()
        # self.vision_runner.join()
        if self.log: self.log_listener.stop()

    def run(self):
        # print(f"Running Coordinator: Process ID: {os.getpid()}")
        self.print_process()
        self.start()

        last_graph_time = 0

        self.goal_position.update({'x': .8, 'y': -.8, 'th': 0})

        try:
           while self.shared_data['running']:
                self.shared_data['motion_state'].set(RobotStates.DRIVE)
                # print(self.shared_data['motion_state'])

                # time.sleep(5)

                # Graphing
                if time.time() - last_graph_time > self.graph_interval and len(self.robot_graph_data) > 0 and self.live_graphs:
                    last_graph_time = time.time()
                    self.orchestrator.update_plot(self.fig, self.axes)
                # self.logger.info("Coordinator is running")

        except KeyboardInterrupt:
            self.logger.info("Exiting Coordinator")
        except Exception:
            self.logger.error(f"Error in Coordinator, exiting: {traceback.print_exc()}")

        finally:
            # print(self.axes)
            # print(self.fig)
            if self.fig is not None and self.axes is not None:
                self.orchestrator.update_plot(self.fig, self.axes)
            # print(self.robot_graph_data)
            plt.ioff()
            plt.show()
            self.stop()

    def print_process(self):
        # Get the current process ID
        pid = os.getpid()
        # Get the CPU core this process is running on
        process = psutil.Process(pid)
        print(f"Coordinator Process (PID: {pid}) running with: {process.num_threads()} threads")


if __name__ == '__main__':
    coordinator = Coordinator(simulate=True)
    coordinator.run()