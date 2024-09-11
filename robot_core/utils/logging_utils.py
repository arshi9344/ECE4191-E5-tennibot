import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

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
    root.addHandler(QueueHandler(queue))


def setup_file_logging(log_queue):
    file_handler = RotatingFileHandler('coordinator.log', maxBytes=20*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()  # Keep console logging if desired
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    listener = QueueListener(log_queue, file_handler, console_handler)
    listener.daemon = True
    listener.start()
    return listener


def log_listener(queue):
    handler = logging.StreamHandler()
    listener = QueueListener(queue, handler)
    listener.start()
    return listener

