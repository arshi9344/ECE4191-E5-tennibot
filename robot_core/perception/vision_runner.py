
import time
import traceback
import numpy as np
import json
import random
from IPython import display
import multiprocessing as mp

import logging
import queue
import os
import psutil
import cv2


from matplotlib import pyplot as plt

from robot_core.utils.logging_utils import setup_logging
from robot_core.coordinator.robot_states import RobotStates, VisionStates, StateWrapper


class VisionRunner(mp.Process):
    def __init__(
            self,
            shared_data,
            log_queue,
            camera_idx,
            log=False,
            scanning_interval=0.5,
            use_simulated_video=False
    ):
        super().__init__()
        if log: setup_logging(log_queue)
        self.logger = logging.getLogger(f'{__name__}.Orchestrator')

        self.scanning_interval = scanning_interval # The duration in seconds between each image
        self.shared_data = shared_data

        self.logger.info(f"Initialising VisionRunner, scanning interval: {self.scanning_interval}s")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")

        self.last_update = None

        # TODO: Add class instances here for TennisBallDetector and BoxDetector
        self.camera_idx = camera_idx
        self.camera = None
        self.frame = None




    def run(self):
        self.camera = cv2.VideoCapture(self.camera_idx)

        if not self.camera.isOpened():
            self.logger.error("Error: Could not open USB camera.")
            raise Exception("Error: Could not open USB camera.")
        else:
            self.logger.info("VisionRunner: Camera opened successfully")

        try:
            while self.shared_data['running']:
                if self.shared_data['vision_state'].get() != VisionStates.NONE:
                    ret, self.frame = self.camera.read() # capture image
                    if not ret:
                        print("VisionRunner: Failed to capture image.")
                        continue
                    else:
                        print("VisionRunner: Image captured")

                    self.last_update = time.time()
                    print(f"Frame: {self.frame}")

                    # cv2.imshow('camera', self.frame)



                    # Now, dependent on what we're checking for (either box or ball), we can add the appropriate logic here

                    # Add scanning here, dependent on self.shared_data['robot_state']




                    time.sleep(self.scanning_interval) # The only problem with this approach is that

        except KeyboardInterrupt:
            self.logger.info("VisionRunner: Keyboard Interrupt")
        except Exception as e:
            self.logger.error(f"Error in VisionRunner, Stopping robot: {traceback.print_exc()}")
        self.release_camera()

    def release_camera(self):
        print('Releasing camera')
        self.camera.release()
        cv2.destroyAllWindows()

    def show_image(self, axis):
        print(self.frame)

        if self.frame is not None:
            print('Showing image')
            axis.imshow(self.frame)
            plt.show()
        else:
            print('No image to show')


    # def print_process(self):
    #     # Get the current process ID
    #     pid = os.getpid()
    #     # Get the CPU core this process is running on
    #     process = psutil.Process(pid)
    #     print(f"Orchestrator Process (PID: {pid}), {self.name}, with: {process.num_threads()} threads")


# Test usage
if __name__ == '__main__':
    manager = mp.Manager()
    shared_data = {
        'running': True,
        'vision_state': StateWrapper(manager, VisionStates, VisionStates.DETECT_BALL),
        'robot_state': StateWrapper(manager, RobotStates, RobotStates.STOP)
    }
    fig, ax = plt.subplots()

    log_queue = mp.Queue()
    camera_idx = 1
    vision_runner = VisionRunner(shared_data, log_queue, camera_idx, log=True)
    try:
        vision_runner.start()
        while True:
            vision_runner.show_image(ax)
            time.sleep(1)


    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    except Exception as e:
        print(f"Error in main {e}")

    vision_runner.join()
    print("VisionRunner process joined")
    print("Exiting...")