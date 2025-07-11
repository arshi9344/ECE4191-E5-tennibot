import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

import time


def setup_logging(queue):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
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
