import multiprocessing as mp
import queue
import time


class MotionControl(mp.Process):
    def __init__(self, command_queue, status_queue):
        super().__init__()
        self.command_queue = command_queue
        self.status_queue = status_queue
        self.running = mp.Event()
        self.running.set()

    def run(self):
        while self.running.is_set():
            try:
                command = self.command_queue.get_nowait()
                print(f"Motion control received: {command}")
            except queue.Empty:
                pass

            time.sleep(0.01)
            self.status_queue.put("Motion control status")

    def stop(self):
        self.running.clear()


class ComputerVision(mp.Process):
    def __init__(self, vision_queue):
        super().__init__()
        self.vision_queue = vision_queue
        self.running = mp.Event()
        self.running.set()

    def run(self):
        while self.running.is_set():
            time.sleep(0.1)
            self.vision_queue.put("Vision data")

    def stop(self):
        self.running.clear()