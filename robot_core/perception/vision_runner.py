
import time
import traceback
import numpy as np
import json
import random
from IPython import display
import multiprocessing as mp
import traceback


import logging
import queue
import os
import psutil
import cv2
from pathlib import Path

from matplotlib import pyplot as plt

from robot_core.utils.logging_utils import setup_logging
from robot_core.coordinator.robot_states import RobotStates, VisionStates, StateWrapper
from robot_core.utils.position import Position, PositionTypes
from robot_core.perception.vision_model.tennis_YOLO import TennisBallDetectorHeight

class VisionRunner(mp.Process):
    def __init__(
            self,
            shared_data,
            shared_image,
            log_queue,
            goal_position,
            camera_idx,
            collection_zone=(200, 150, 400, 350),
            camera_height=0.02,
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
        self.goal_position = goal_position # shared dictionary object, read by Orchestrator
        # TODO: Add class instances here for TennisBallDetector and BoxDetector
        self.default_camera_idx = camera_idx
        self.camera = None
        self.frame = None
        self.collection_zone = collection_zone
        self.simulate = use_simulated_video

        self.tennis_ball_detector = None

        # Load the YOLO model and calibration data
        # current_dir = Path(__file__).parent # Get the directory of the current script
        # Construct the path to the .npz file
        # npz_file_path = current_dir / 'vision_model/calib6_matrix.npz'
        # model_path = current_dir / 'vision_model/best.pt'
        # custom_cache_dir = current_dir / 'vision_model/cache'
        # # Set the custom cache directory
        # os.environ['TORCH_HOME'] = custom_cache_dir

        # Now you can load the .npz file
        # calibration_data = np.load(npz_file_path)
        # camera_matrix = calibration_data['camera_matrix']
        # distortion_coeffs = calibration_data['dist_coeffs']
        self.collection_zone = collection_zone # Define a region for collection, x_min, y_min, x_max, y_max
        self.camera_height = camera_height  # Camera height in meters (2 cm)



    def run(self):
        # Initialize the TennisBallDetector with camera height
        try:
            self.tennis_ball_detector = TennisBallDetectorHeight(
                collection_zone=self.collection_zone,
                camera_height=self.camera_height,
                cache=True,  # Use the cache for the YOLO model if it exists
                windows=False,  # Path handling for Windows
                verbose=True  # Enable verbose logging for debugging
            )
        except Exception:
            print(f"Error loading TennisBallDetector: {traceback.print_exc()}")

        if not self.simulate:
            try:
                if not self.open_camera():
                    print("Exiting VisionRunner run().")
                    return
            except:
                print(f"Error opening camera: {traceback.print_exc()}")
                return

            try:
                while self.shared_data['running']:
                    if self.shared_data['vision_state'].get() != VisionStates.NONE:
                        ret, self.frame = self.camera.read() # capture image
                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                        if not ret:
                            print("VisionRunner: Failed to capture image.")
                            continue
                        else:
                            # print("VisionRunner: Image captured")
                            pass

                        """
                        For now, we will just update the shared image with the raw frame for debugging purposes so we can 
                        figure out what's going on with the TennisBallDetector and why Torch is having that 'module not found' error.
                        """
                        detection = self.tennis_ball_detector.detect(self.frame)
                        self.last_update = time.time()
                        self.shared_image.update({
                            'time': self.last_update,
                            'frame': detection["annotated_frame"]
                        })

                        """
                        Directly updating the goal position here with ball coords, but this should be passed via a shared queue object into the occupancy map
                        """
                        coords = detection["cartesian_coords"]
                        if coords is not None:
                            x, y = coords
                            self.goal_position.update({
                                'goal': Position(x, y, 0, PositionTypes.BALL),
                                'time': time.time()
                            })


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
        'robot_state': StateWrapper(manager, RobotStates, RobotStates.SEARCH)
    }
    goal_position = manager.dict({
        'goal': Position(None, None, None, PositionTypes.BALL),
        'time': None
    })

    log_queue = mp.Queue()
    camera_idx = 1
    shared_image = manager.dict({'time': None, 'frame': None})

    vision_runner = VisionRunner(
        shared_data=shared_data,
        shared_image=shared_image,
        goal_position=goal_position,
        log_queue=log_queue,
        camera_idx=0,
        log=False
    )
    try:
        vision_runner.start()
        while True:
            if shared_image['frame'] is not None:
                plt.imshow(shared_image['frame'])
            plt.show()
            time.sleep(1)



    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    except Exception as e:
        print(f"Error in main {traceback.print_exc()}")

    vision_runner.join()
    print("VisionRunner process joined")
    print("Exiting...")