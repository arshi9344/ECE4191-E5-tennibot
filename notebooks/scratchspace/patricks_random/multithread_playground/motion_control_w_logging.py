import multiprocessing as mp
from multiprocessing import Manager
import time
import logging
from logging.handlers import QueueHandler, QueueListener
import queue

def setup_logging(queue):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(QueueHandler(queue))


def log_listener(queue):
    handler = logging.StreamHandler()
    listener = QueueListener(queue, handler)
    listener.start()
    return listener


class MotionControl(mp.Process):
    def __init__(self, shared_data, command_queue, status_queue, log_queue):
        mp.Process.__init__(self)
        self.shared_data = shared_data
        self.command_queue = command_queue
        self.status_queue = status_queue
        setup_logging(log_queue)
        self.logger = logging.getLogger(f'{__name__}.MotionControl')

    def run(self):
        while self.shared_data['running']:
            try:
                command = self.command_queue.get_nowait()
                self.logger.info(f"Received command: {command}")
                # Process command
                if command == 'STOP':
                    self.shared_data['motion_state'] = 'STOPPED'
                elif command == 'MOVE':
                    self.shared_data['motion_state'] = 'MOVING'
            except Exception:
                pass

            # Update position if moving
            if self.shared_data['motion_state'] == 'MOVING':
                with self.shared_data['position'].get_lock():
                    self.shared_data['position']['x'] += 1
                    self.shared_data['position']['y'] += 1

            # Send status update
            self.status_queue.put({
                'motion_state': self.shared_data['motion_state'],
                'position': dict(self.shared_data['position'])
            })

            time.sleep(0.1)


class ComputerVision(mp.Process):
    def __init__(self, shared_data, vision_queue, log_queue):
        mp.Process.__init__(self)
        self.shared_data = shared_data
        self.vision_queue = vision_queue
        # setup_logging(log_queue)
        self.logger = logging.getLogger(f'{__name__}.ComputerVision')

    def run(self):
        while self.shared_data['running']:
            # Simulate object detection
            detected_object = {'type': 'obstacle', 'distance': 5.0}
            self.shared_data['detected_objects'].append(detected_object)

            # Send vision data
            self.vision_queue.put(detected_object)

            self.logger.info(f"Detected object: {detected_object}")

            time.sleep(0.5)