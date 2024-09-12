import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import time
# class LogData:
#     def __init__(self):
#         self.poses = []
#         self.velocities = []
#         self.desired_velocities = []
#         self.duty_cycle_commands = []
#         self.error_sums = []
#         self.errors = []
#         self.actual_dts = []


def setup_logging(queue):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Remove all existing handlers to avoid duplicates
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    # Add new QueueHandler
    queue_handler = QueueHandler(queue)
    root.addHandler(queue_handler)

def create_log_listener(queue):
    run_time = time.strftime("%d-%m-%Y %H:%M:%S")
    file_handler = RotatingFileHandler(f'Robot Run {run_time}.log', maxBytes=20*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    listener = QueueListener(queue, file_handler, console_handler)
    return listener
