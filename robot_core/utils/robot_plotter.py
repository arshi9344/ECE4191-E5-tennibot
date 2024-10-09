import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bisect import bisect_right
import numpy as np
import time
from IPython import display
import threading
import queue
from typing import List, Optional, NamedTuple
from robot_core.utils.robot_log_point import RobotLogPoint


"""
- The RobotPlotter class is responsible for plotting the robot's pose, duty cycle, wheel velocity, and position data. It's instantiated by Coordinator and used to plot data in real-time.

NOTES:
- Because matplotlib is blocking and also not thread-safe, we need to create a separate thread for plotting so that if we plot lots of data, it doesn't slow down the main thread
    - this is important because slowing the main thread down in Coordinator can cause the robot to not respond to commands in time, not move as expected, and not react as quickly to new data
- The RobotPlotter class uses a cmd_queue to pass data to the plotting thread, which then allows us to plot data in real-time with minimal impact on the main thread
"""

class RobotPlotter:
    def __init__(self, max_time_window, save_figs=True, efficient_mode=True):
        self.max_time_window = max_time_window
        self.start_time = None
        self.last_plotted_time = 0
        self.last_image_time = 0
        self.data_queue = queue.Queue()  # cmd_queue for passing data to the plotting thread
        self.plotting_thread = threading.Thread(target=self.plotting_loop, daemon=True) # Create the plotting thread
        self.running = False
        self.save_figs = save_figs
        self.efficient_mode=efficient_mode

    def start(self):
        # start the plotting thread
        self.running = True
        self.plotting_thread.start()
        self.data_queue.put(('init_plot', None)) # initialise the plots (will happen in a separate thread)

    def stop(self):
        self.running = False

    def init_plot(self):
        self.fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])

        self.ax1 = plt.subplot(gs[0, 0])
        self.ax2 = plt.subplot(gs[0, 1])
        self.ax3 = plt.subplot(gs[1, 0])
        self.ax4 = plt.subplot(gs[1, 1])
        self.ax5 = plt.subplot(gs[:, 2])

        self.path_line, = self.ax1.plot([], [], 'b-')
        self.orientation_quiver = self.ax1.quiver([], [], [], [])

        self.duty_cycle_lines = [
            self.ax2.plot([], [], label='Left Wheel')[0],
            self.ax2.plot([], [], label='Right Wheel')[0]
        ]

        self.velocity_lines = [
            self.ax3.plot([], [], label='Left Wheel')[0],
            self.ax3.plot([], [], label='Right Wheel')[0]
        ]
        self.desired_velocity_lines = [
            self.ax3.plot([], [], '--', label='Desired Left Wheel')[0],
            self.ax3.plot([], [], '--', label='Desired Right Wheel')[0]
        ]

        self.actual_path_line, = self.ax4.plot([], [], 'b-', label='Actual Path')
        self.goal_path_line, = self.ax4.plot([], [], 'r--', label='Goal Path')


        self.label_plots()

    def label_plots(self):
        efficient_explainer = " (PLOT DISABLED, efficiency_mode=True)" if self.efficient_mode else ""
        self.ax1.set_xlabel('x-position (m)')
        self.ax1.set_ylabel('y-position (m)')
        self.ax1.set_title('Robot Pose Over Time')
        self.ax1.grid(True)

        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Duty Cycle')
        self.ax2.set_title(f'Duty Cycle Commands Over Time{efficient_explainer}')
        self.ax2.set_title('Duty Cycle Commands Over Time')
        self.ax2.legend()
        self.ax2.grid(True)

        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Wheel Velocity (rad/s)')
        self.ax3.set_title(f'Wheel Velocity vs. Time{efficient_explainer}')
        self.ax3.legend()
        self.ax3.grid(True)

        self.ax4.set_xlabel('x-position (m)')
        self.ax4.set_ylabel('y-position (m)')
        self.ax4.set_title('Robot Positions')
        self.ax4.grid(True)
        self.ax4.legend()


    def update_plot(self, graph_data, image_data, clear_output: bool = False):
        if not self.running:
            raise RuntimeError("RobotPlotter is not running. Call start() first.")

        new_graph_data = self._get_new_data(graph_data, self.last_plotted_time)
        new_image_data = image_data if image_data.get('frame') is not None and image_data.get('time', 0) > self.last_image_time else None

        if new_graph_data or new_image_data:
            if new_graph_data:
                self.start_time = self.start_time or new_graph_data[0].time
                self.last_plotted_time = new_graph_data[-1].time
            if new_image_data:
                # print(f"New image data: {new_image_data['time']}")
                # print(f"New image data: {new_image_data}")
                self.last_image_time = new_image_data['time']
            self.data_queue.put(('update_plot', (new_graph_data, new_image_data, clear_output)))

    @staticmethod
    def _get_new_data(data, last_time: float):
        return data[bisect_right([d.time for d in data], last_time):] if data and data[-1].time > last_time else None


    def plotting_loop(self, downsample_rate=2):
        # A downsample rate of 2 means we only plot every other data point. This can help reduce the computational load from plotting.

        while True:
            command, data = self.data_queue.get()
            if command == 'init_plot':
                self.init_plot()
            elif command == 'update_plot':
                graph_data, image_data, clear_output = data

                if graph_data:
                    downsampled_data = graph_data[::downsample_rate] # Downsample the data
                    if downsampled_data:  # Check if there's any data after downsampling
                        self.update_path_plot(downsampled_data)
                        if not self.efficient_mode: self.update_duty_cycle_plot(downsampled_data)
                        if not self.efficient_mode: self.update_velocity_plot(downsampled_data)
                        self.update_position_plot(downsampled_data)

                        duration = graph_data[-1].time - self.start_time
                        self.ax4.set_title(f"Robot Positions. t={duration:.2f} sec")
                if image_data:
                    self.update_image_plot(image_data)

                self.fig.canvas.draw()
                plt.pause(0.001)

                if clear_output:
                    display.clear_output(wait=True)
                display.display(self.fig)

                if self.save_figs:
                    self.fig.savefig(f'robot_plot_{int(time.time())}.png')

            self.data_queue.task_done()

    def update_image_plot(self, image_data):
        self.ax5.imshow(image_data['frame'])
        self.ax5.axis('off')
        t = image_data['time'] - self.start_time
        self.ax5.set_title(f"Camera t={t:.2f} sec")


    def update_path_plot(self, new_data):
        if not new_data:
            return
        new_poses = np.array([d.pose for d in new_data])
        current_data = self.path_line.get_data()
        updated_x = np.append(current_data[0], new_poses[:, 0])
        updated_y = np.append(current_data[1], new_poses[:, 1])
        self.path_line.set_data(updated_x, updated_y)

        if len(new_poses) > 0:
            x, y, th = new_poses[-1]
            self.orientation_quiver.remove()
            self.orientation_quiver = self.ax1.quiver(x, y, 0.1 * np.cos(th), 0.1 * np.sin(th))

        self.ax1.relim()
        self.ax1.autoscale_view()

    def update_duty_cycle_plot(self, new_data):
        if not new_data:
            return

        new_times = np.array([d.time - self.start_time for d in new_data]) # convert time since epoch into time since robot start
        newest_time = new_times[-1]
        new_duty_cycles = np.array([d.duty_cycle_commands for d in new_data])

        for i, line in enumerate(self.duty_cycle_lines): # Loop through left and right wheels
            current_x, current_y = line.get_data()
            updated_x = np.append(current_x, new_times)
            updated_y = np.append(current_y, new_duty_cycles[:, i])

            # Apply time window
            mask = updated_x >= (newest_time - self.max_time_window)
            line.set_data(updated_x[mask], updated_y[mask])

        # Bound the plot to the time window
        ax_left_limit = 0 if newest_time - self.max_time_window < 0 else newest_time - self.max_time_window
        if ax_left_limit != newest_time: self.ax2.set_xlim(ax_left_limit, newest_time)
        self.ax2.relim()
        self.ax2.autoscale_view()


    def update_velocity_plot(self, new_data):
        if not new_data:
            return
        new_times = np.array([d.time - self.start_time for d in new_data])
        newest_time = new_times[-1]
        new_velocities = np.array([d.current_wheel_w for d in new_data])
        new_desired_velocities = np.array([d.target_wheel_w for d in new_data])

        for i, (v_line, dv_line) in enumerate(zip(self.velocity_lines, self.desired_velocity_lines)):
            current_x, current_v = v_line.get_data()
            _, current_dv = dv_line.get_data()

            updated_x = np.append(current_x, new_times)
            updated_v = np.append(current_v, new_velocities[:, i])
            updated_dv = np.append(current_dv, new_desired_velocities[:, i])

            # Apply time window
            mask = updated_x >= (new_times[-1] - self.max_time_window)
            v_line.set_data(updated_x[mask], updated_v[mask])
            dv_line.set_data(updated_x[mask], updated_dv[mask])

        # Bound the plot to the time window
        ax_left_limit = 0 if newest_time - self.max_time_window < 0 else newest_time - self.max_time_window
        if ax_left_limit != newest_time: self.ax3.set_xlim(ax_left_limit, newest_time)
        self.ax3.relim()
        self.ax3.autoscale_view()

    def update_position_plot(self, new_data, only_show_recent=True, recent_thresh=1000):
        if not new_data:
            return

        new_poses = np.array([d.pose for d in new_data])
        new_goals = np.array([(d.goal_position.x, d.goal_position.y, d.goal_position.type) for d in new_data])
        new_goal_x, new_goal_y, new_goal_types = new_goals.T
        #TODO: Colour goal points based on type - add legend. Types are defined in robot_core.utils.position PositionTypes class

        current_actual = self.actual_path_line.get_data()
        current_goal = self.goal_path_line.get_data()

        # If this is the first update, start with (0, 0)
        if len(current_actual[0]) == 0:
            updated_actual_x = np.concatenate(([0], new_poses[:, 0]))
            updated_actual_y = np.concatenate(([0], new_poses[:, 1]))

            updated_goal_x = np.concatenate(([0], new_goal_x))
            updated_goal_y = np.concatenate(([0], new_goal_y))
        else:
            # If not the first update, just append new data
            updated_actual_x = np.append(current_actual[0], new_poses[:, 0])
            updated_actual_y = np.append(current_actual[1], new_poses[:, 1])
            updated_goal_x = np.append(current_goal[0], new_goal_x)
            updated_goal_y = np.append(current_goal[1], new_goal_y)

        if only_show_recent:
            # Apply time window
            updated_actual_x = updated_actual_x[-recent_thresh:]
            updated_actual_y = updated_actual_y[-recent_thresh:]
            updated_goal_x = updated_goal_x[-recent_thresh:]
            updated_goal_y = updated_goal_y[-recent_thresh:]
        self.actual_path_line.set_data(updated_actual_x, updated_actual_y)
        self.goal_path_line.set_data(updated_goal_x, updated_goal_y)

        self.ax4.relim()
        self.ax4.autoscale_view()

# Usage remains the same
if __name__ == '__main__':
    plotter = RobotPlotter(max_time_window=10)
    plotter.start()
    # In your main loop or update function:
    # plotter.update_plot(robot_graph_data)