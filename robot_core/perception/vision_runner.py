
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
            shared_image,
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
        self.shared_image = shared_image
        self.logger.info(f"Initialising VisionRunner, scanning interval: {self.scanning_interval}s")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")

        self.last_update = None

        # TODO: Add class instances here for TennisBallDetector and BoxDetector
        self.camera_idx = camera_idx
        self.camera = None
        self.frame = None

        self.simulate = use_simulated_video



    def run(self):
        if not self.simulate:
            if not self.open_camera():
                print("Exiting VisionRunner run().")
                return

            try:
                while self.shared_data['running']:
                    if self.shared_data['vision_state'].get() != VisionStates.NONE:
                        ret, self.frame = self.camera.read() # capture image
                        if not ret:
                            print("VisionRunner: Failed to capture image.")
                            continue
                        else:
                            # print("VisionRunner: Image captured")
                            pass

                        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                        self.last_update = time.time()
                        self.shared_image.update({
                            'time': self.last_update,
                            'frame': frame_rgb
                        })
                        # print(f"Frame: {self.frame}")

                        # cv2.imshow('camera', self.frame)



                        # Now, dependent on what we're checking for (either box or ball), we can add the appropriate logic here

                        # Add scanning here, dependent on self.shared_data['robot_state']

                        time.sleep(self.scanning_interval) # The only problem with this approach is that we always need to wait for the interval to pass, we can't immediately request an image

            except KeyboardInterrupt:
                self.logger.info("VisionRunner: Keyboard Interrupt")
            except Exception as e:
                self.logger.error(f"Error in VisionRunner, Stopping robot: {traceback.print_exc()}")
            self.release_camera()

        else:
            # Insert code here to simulate video feed
            pass

        print("VisionRunner: Exiting run method")

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

    def open_camera(self):
        camera_idxs = [self.camera_idx, self.camera_idx + 1, self.camera_idx + 2, self.camera_idx - 1]
        for idx in camera_idxs:
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                self.camera = camera
                self.camera_idx = idx
                print(f"VisionRunner: Camera opened successfully using idx {idx}")
                return True

        print(f"VisionRunner: Error: Could not open USB camera. Tried {camera_idxs}")
        return False

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