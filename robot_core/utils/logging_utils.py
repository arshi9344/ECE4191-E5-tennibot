import logging
from logging.handlers import QueueHandler, QueueListener

class LogData:
    def __init__(self):
        self.poses = []
        self.velocities = []
        self.desired_velocities = []
        self.duty_cycle_commands = []
        self.error_sums = []
        self.errors = []
        self.actual_dts = []


def setup_logging(queue):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(QueueHandler(queue))

def log_listener(queue):
    handler = logging.StreamHandler()
    listener = QueueListener(queue, handler)
    listener.start()
    return listener