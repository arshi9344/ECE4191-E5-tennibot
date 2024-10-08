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
from robot_core.perception.detection_results import DetectionResult

class VisionRunner(mp.Process):
    def __init__(
            self,
            shared_data,
            shared_image,
            log_queue,
            detection_results_q,
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

        self.scanning_interval = scanning_interval  # The duration in seconds between each image
        self.shared_data = shared_data
        self.shared_image = shared_image
        self.logger.info(f"Initialising VisionRunner, scanning interval: {self.scanning_interval}s")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")

        self.last_update = None
        self.detection_results_q = detection_results_q  # shared queue, containing the latest detection results
        self.default_camera_idx = camera_idx
        self.camera = None
        self.frame = None
        self.collection_zone = collection_zone
        self.simulate = use_simulated_video
        self.tennis_ball_detector = None

    def run(self):
        """Main logic of the VisionRunner process."""
        try:
            # Initialize the TennisBallDetector with camera height
            self.tennis_ball_detector = TennisBallDetectorHeight(
                collection_zone=self.collection_zone,
                camera_height=self.camera_height,
                cache=True,  # Use cached YOLO model if available
                windows=False,  # Path handling for Windows
                verbose=True  # Enable verbose logging for debugging
            )
        except Exception:
            self.logger.error(f"Error loading TennisBallDetector: {traceback.format_exc()}")

        if not self.simulate:
            try:
                if not self.open_camera():
                    self.logger.error("Exiting VisionRunner run() - Failed to open camera.")
                    return
            except Exception:
                self.logger.error(f"Error opening camera: {traceback.format_exc()}")
                return

            try:
                while self.shared_data['running']:
                    if self.shared_data['vision_state'].get() != VisionStates.NONE:
                        ret, self.frame = self.camera.read()  # Capture image
                        if not ret:
                            self.logger.warning("VisionRunner: Failed to capture image.")
                            continue

                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        self.last_update = time.time()

                        # Run the detection
                        detection = self.tennis_ball_detector.detect(self.frame)

                        # Handle detection results and pass them to the shared queue
                        if detection.ball_detection or detection.box_detection:
                            frame = detection.frame
                            self.detection_results_q.put({
                                'time': self.last_update,
                                'frame': frame,
                                'ball_detection': detection.ball_detection,
                                'box_detection': detection.box_detection
                            })
                        else:
                            frame = self.frame

                        # Update shared image data
                        self.shared_image.update({
                            'time': self.last_update,
                            'frame': frame
                        })

                        time.sleep(self.scanning_interval)  # Respect the scanning interval
            except KeyboardInterrupt:
                self.logger.info("VisionRunner: Keyboard Interrupt")
            except Exception as e:
                self.logger.error(f"Error in VisionRunner, stopping robot: {traceback.format_exc()}")
            finally:
                self.release_camera()
        else:
            # Simulated video feed handling (not implemented)
            pass

        self.logger.info("VisionRunner: Exiting run method")

    def release_camera(self):
        """Safely release the camera and close OpenCV windows."""
        self.logger.info('Releasing camera')
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

    def open_camera(self):
        """Attempt to open the camera with multiple indices."""
        MAX_IND = 3
        camera_idxs = [self.default_camera_idx] + [x for x in range(MAX_IND) if x != self.default_camera_idx]
        for idx in camera_idxs:
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                self.camera = camera
                self.default_camera_idx = idx
                self.logger.info(f"VisionRunner: Camera opened successfully using idx {idx}")
                return True
        self.logger.error(f"VisionRunner: Could not open USB camera. Tried {camera_idxs}")
        return False


# Test usage
if __name__ == '__main__':
    manager = mp.Manager()
    shared_data = {
        'running': True,
        'vision_state': StateWrapper(manager, VisionStates, VisionStates.DETECT_BALL),
        'robot_state': StateWrapper(manager, RobotStates, RobotStates.SEARCH)
    }
    detection_results_q = manager.Queue()
    log_queue = mp.Queue()
    camera_idx = 1
    shared_image = manager.dict({'time': None, 'frame': None})

    vision_runner = VisionRunner(
        shared_data=shared_data,
        shared_image=shared_image,
        detection_results_q=detection_results_q,
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
        print(f"Error in main: {traceback.format_exc()}")
    finally:
        vision_runner.join()
        print("VisionRunner process joined")
        print("Exiting...")
