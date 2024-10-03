# Plots data from Orchestrator_mp and VisionRuner_mp processes

import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display
from robot_core.utils.logging_utils import RobotLogPoint

class RobotPlotter:
    def __init__(self, max_points=1000):
        self.max_points = max_points
        self.robot_graph_data = None
        self.start_time = None
        self.controller = None  # Assuming this is set elsewhere

        # New attributes to track plotting state
        self.last_plotted_index = 0
        self.poses = np.zeros((max_points, 3))
        self.duty_cycle_commands = np.zeros((max_points, 2))
        self.velocities = np.zeros((max_points, 2))
        self.desired_velocities = np.zeros((max_points, 2))
        self.goal_positions = np.zeros((max_points, 3))

        # Initialize the plot
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.setup_plots()

    def setup_plots(self):
        # Initialize all subplots here
        # (Similar to your original setup, but create Line2D objects)
        self.path_line, = self.axes[0, 0].plot([], [], 'b-')
        self.orientation_quiver = self.axes[0, 0].quiver([], [], [], [])
        # ... (initialize other lines for each subplot)

    def update_plot(self, clear_output=False):
        if self.robot_graph_data is None or len(self.robot_graph_data) == 0:
            return

        new_data = self.robot_graph_data[self.last_plotted_index:]
        if not new_data:
            return

        # Update data arrays
        for i, data_point in enumerate(new_data, start=self.last_plotted_index):
            idx = i % self.max_points
            self.poses[idx] = data_point['pose']
            self.duty_cycle_commands[idx] = data_point['duty_cycle_commands']
            self.velocities[idx] = data_point['current_wheel_w']
            self.desired_velocities[idx] = data_point['target_wheel_w']
            self.goal_positions[idx] = data_point['goal_position']

        # Update plots
        self.update_path_plot()
        self.update_duty_cycle_plot()
        self.update_velocity_plot()
        self.update_position_plot()

        self.last_plotted_index = len(self.robot_graph_data)

        self.fig.canvas.draw()
        plt.pause(0.001)

        if clear_output:
            display.clear_output(wait=True)
        display.display(self.fig)

    def update_path_plot(self):
        valid_poses = self.poses[~np.all(self.poses == 0, axis=1)]
        self.path_line.set_data(valid_poses[:, 0], valid_poses[:, 1])
        if len(valid_poses) > 0:
            x, y, th = valid_poses[-1]
            self.orientation_quiver.remove()
            self.orientation_quiver = self.axes[0, 0].quiver(x, y, 0.1 * np.cos(th), 0.1 * np.sin(th))
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()

    def update_duty_cycle_plot(self):
        valid_duty_cycles = self.duty_cycle_commands[~np.all(self.duty_cycle_commands == 0, axis=1)]
        # Update duty cycle lines
        # ... (similar to path plot update)

    def update_velocity_plot(self):
        valid_velocities = self.velocities[~np.all(self.velocities == 0, axis=1)]
        valid_desired_velocities = self.desired_velocities[~np.all(self.desired_velocities == 0, axis=1)]
        # Update velocity lines
        # ... (similar to path plot update)

    def update_position_plot(self):
        valid_poses = self.poses[~np.all(self.poses == 0, axis=1)]
        valid_goals = self.goal_positions[~np.all(self.goal_positions == 0, axis=1)]
        # Update position plot
        # ... (similar to path plot update)


# Usage
plotter = RobotPlotter()
# ... populate robot_graph_data ...
plotter.update_plot()
