
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
from robot_core.coordinator.commands import RobotCommands, VisionCommands, StateWrapper
from robot_core.utils.position import Position, PositionTypes
from robot_core.perception.detection_results import BallDetection, BoxDetection, DetectionResult

class VisionRunner(mp.Process):
    def __init__(
            self,
            shared_data,
            shared_image,
            log_queue,
            detection_results_q,
            camera_idx,
            robot_pose,
            collection_zone=(200, 150, 400, 350),
            deposition_zone=(200, 150, 400, 350),
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
        self.robot_pose = robot_pose

        self.logger.info(f"Initialising VisionRunner, scanning interval: {self.scanning_interval}s")
        self.logger.info(f"Process ID: {os.getpid()} - Running worker: {self.name}")

        self.last_update = None
        self.detection_results_q = detection_results_q # shared cmd_queue, containing the latest detection results, read by ProcessCoordinator
        # TODO: Add class instances here for TennisBallDetector and BoxDetector
        
        self.default_camera_idx = camera_idx
        self.camera = None
        self.camera_height = camera_height
        self.frame = None
        self.collection_zone = collection_zone
        self.deposition_zone = deposition_zone
        self.simulate = use_simulated_video

        self.ball_detector = None
        self.box_detector = None

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
        

    def combine_frames(self, ball_frame, box_frame):
        combined_frame = ball_frame  # Use ball_frame as the base frame
        combined_frame = cv2.addWeighted(ball_frame, 0.5, box_frame, 0.5, 0)
        
        return combined_frame

    def run(self):
        # Initialize the TennisBallDetector with camera height
        try:
            from robot_core.perception.vision_model.tennis_YOLO import TennisBallDetector, BoxDetector
            self.ball_detector = TennisBallDetector(collection_zone= self.collection_zone)
            self.box_detector = BoxDetector(deposition_zone=self.deposition_zone)

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
                    if self.shared_data['vision_command'].get() != VisionCommands.NONE:
                        ret, self.frame = self.camera.read() # capture image
                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                        if not ret:
                            print("VisionRunner: Failed to capture image.")
                            continue
                        else:
                            # print("VisionRunner: Image captured")
                            self.last_update = time.time()
                        
                        
                        # Copy the original frame for each detector
                        ball_frame = self.frame.copy()
                        box_frame = self.frame.copy()

                        # Run the detection model
                        ball_detections, ball_frame = self.ball_detector.detect(ball_frame)
                        box_detections, box_frame = self.box_detector.detect(box_frame)

                        # Convert the relative position of the ball to global position
                        ball_detections = [self._estimate_ball_global_position(detection) for detection in ball_detections]

                        if ball_detections or box_detections:
                            combined_frame = self.combine_frames(ball_frame, box_frame)

                            
                            self.detection_results_q.put({
                                'time': self.last_update,
                                'frame': combined_frame, #  CHECK 
                                'ball_detection': ball_detections,
                                'box_detection': box_detections
                            })
                        else:
                            combined_frame = self.frame

                        self.shared_image.update({
                            'time': self.last_update,
                            'frame': combined_frame
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


    def _estimate_ball_global_position(self, detection: BallDetection) -> BallDetection:

        # Estimate the global position using the robot's current pose and the detection's relative position
        robot_x = self.robot_pose['x']
        robot_y = self.robot_pose['y']
        robot_th = self.robot_pose['th']

        dx = detection.x
        dy = detection.y
        cos_th = np.cos(robot_th)
        sin_th = np.sin(robot_th)
        gx = robot_x + dx * cos_th - dy * sin_th
        gy = robot_y + dx * sin_th + dy * cos_th

        detection.x = gx
        detection.y = gy
        return detection


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
        'vision_command': StateWrapper(manager, VisionCommands, VisionCommands.DETECT_BALL),
        'robot_command': StateWrapper(manager, RobotCommands, RobotCommands.DRIVE)
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
        print(f"Error in main {traceback.print_exc()}")

    vision_runner.join()
    print("VisionRunner process joined")
    print("Exiting...")