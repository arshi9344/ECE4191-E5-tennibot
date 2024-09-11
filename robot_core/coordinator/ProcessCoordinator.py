import multiprocessing as mp
from multiprocessing import Manager
import time
import logging
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from robot_core.hardware.diff_drive_robot import DiffDriveRobot
from robot_core.motion.tentacle_planner import TentaclePlanner
from robot_core.control.PI_controller import PIController
from robot_core.coordinator.robot_states import RobotStates
from robot_core.orchestration.orchestrator_mp import Orchestrator
from robot_core.utils.logging_utils import setup_logging, setup_file_logging, log_listener

import os



class Coordinator:
    def __init__(self, simulate=False):
        self.manager = Manager()
        self.shared_data = {
            'running': self.manager.Value('b', True),
            'motion_state': self.manager.Value('s', RobotStates.STOP),
            'detected_objects': self.manager.list() # not used for now
        }
        self.robot_pose = self.manager.dict({'x': 0, 'y': 0, 'th': 0})
        self.goal_position = self.manager.dict({'x': 0, 'y': 0, 'th': 0})

        # Setting up Logging
        self.log_queue = self.manager.Queue()
        self.logger = logging.getLogger(f'{__name__}.Coordinator')
        # self.listener = setup_file_logging(self.log_queue)
        self.listener = log_listener(self.log_queue)

        setup_logging(self.log_queue)

        # Instantiating child processes
        # Orchestrator
        self.orchestrator = Orchestrator(
            shared_data=self.shared_data,
            goal_position=self.goal_position,
            robot_pose=self.robot_pose,
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
        self.listener.stop()

    def run(self):
        self.start()
        try:
            while self.shared_data['running']:
                time.sleep(5)
                self.logger.info("Coordinator is running")
        except KeyboardInterrupt:
            self.logger.info("Exiting Coordinator")
            self.stop()
        except Exception as e:
            self.logger.error(f"Error in Coordinator, exiting: {e}")
            self.stop()


if __name__ == '__main__':
    coordinator = Coordinator(simulate=True)
    coordinator.run()