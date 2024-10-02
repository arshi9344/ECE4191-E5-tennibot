import cv2
import multiprocessing as mp
import numpy as np
from enum import Enum
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class VisionStates(Enum):
    NONE = 0
    DETECT_BALL = 1
    DETECT_BOX = 2


class RobotStates(Enum):
    STOP = 0
    MOVE = 1


class StateWrapper:
    def __init__(self, manager, enum_class, initial_state):
        self._value = manager.Value('i', initial_state.value)
        self._enum_class = enum_class

    def get(self):
        return self._enum_class(self._value.value)

    def set(self, new_state):
        if isinstance(new_state, self._enum_class):
            self._value.value = new_state.value
        else:
            raise ValueError(f"State must be an instance of {self._enum_class}")


class FrameCaptureProcess(mp.Process):
    def __init__(self, frame_queue, shared_data, camera_idx=1):
        super().__init__()
        self.frame_queue = frame_queue
        self.shared_data = shared_data
        self.camera_idx = camera_idx

    def run(self):
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_idx}.")
            self.shared_data['running'] = False
            return

        print("FrameCaptureProcess: Camera opened successfully")

        while self.shared_data['running']:
            if self.shared_data['vision_state'].get() != VisionStates.NONE:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except mp.queues.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frame_rgb)
                except mp.queues.Full:
                    pass

            time.sleep(0.01)  # Small sleep to prevent excessive CPU usage

        cap.release()
        print("FrameCaptureProcess: Camera released")


def update_plot(frame, ax, frame_queue, shared_data):
    if not frame_queue.empty():
        try:
            img = frame_queue.get_nowait()
            ax.clear()
            ax.imshow(img)
            ax.set_title(f"Vision State: {shared_data['vision_state'].get().name}")
            ax.axis('off')
        except mp.queues.Empty:
            pass
    return ax,


def on_key_press(event, shared_data):
    if event.key == 'q':
        shared_data['running'] = False
        plt.close('all')
    elif event.key == 'b':
        shared_data['vision_state'].set(VisionStates.DETECT_BALL)
        print("Switched to ball detection mode")
    elif event.key == 'x':
        shared_data['vision_state'].set(VisionStates.DETECT_BOX)
        print("Switched to box detection mode")
    elif event.key == 'n':
        shared_data['vision_state'].set(VisionStates.NONE)
        print("Switched to no detection mode")


if __name__ == "__main__":
    manager = mp.Manager()
    frame_queue = mp.Queue(maxsize=2)
    shared_data = {
        'running': True,
        'vision_state': StateWrapper(manager, VisionStates, VisionStates.DETECT_BALL),
        'robot_state': StateWrapper(manager, RobotStates, RobotStates.STOP)
    }

    capture_process = FrameCaptureProcess(frame_queue, shared_data)
    capture_process.start()

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, shared_data))

    ani = FuncAnimation(fig, update_plot, fargs=(ax, frame_queue, shared_data),
                        interval=33, blit=True)

    print("Press 'q' to quit, 'b' for ball detection, 'x' for box detection, 'n' for no detection.")
    plt.show()

    capture_process.join()
    print("Exiting...")