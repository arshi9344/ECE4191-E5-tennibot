import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
from IPython import display


class RobotPlotter:
    def __init__(self, max_points=500):
        self.max_points = max_points
        self.start_time = None
        self.last_plotted_index = 0

        # Create the figure and subplots
        self.fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])

        self.ax1 = plt.subplot(gs[0, 0])  # Robot pose vs. time
        self.ax2 = plt.subplot(gs[0, 1])  # Duty cycle vs. time
        self.ax3 = plt.subplot(gs[1, 0])  # Wheel velocity vs. time
        self.ax4 = plt.subplot(gs[1, 1])  # Robot positions, goal paths
        self.ax5 = plt.subplot(gs[:, 2])  # Camera view (spans both rows)

        # Initialize plot lines
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
        # Set up labels, titles, and legends for each subplot
        self.ax1.set_xlabel('x-position (m)')
        self.ax1.set_ylabel('y-position (m)')
        self.ax1.set_title('Robot Pose Over Time')
        self.ax1.grid(True)
        # self.ax1.axis('equal')


        self.ax2.set_xlabel('Time Step')
        self.ax2.set_ylabel('Duty Cycle')
        self.ax2.set_title('Duty Cycle Commands Over Time')
        self.ax2.legend()
        self.ax2.grid(True)
        # self.ax2.axis('equal')


        self.ax3.set_xlabel('Time Step')
        self.ax3.set_ylabel('Wheel Velocity (rad/s)')
        self.ax3.set_title('Wheel Velocity vs. Time')
        self.ax3.legend()
        self.ax3.grid(True)
        # self.ax3.axis('equal')


        self.ax4.set_xlabel('x-position (m)')
        self.ax4.set_ylabel('y-position (m)')
        self.ax4.set_title('Robot Positions')
        self.ax4.grid(True)
        self.ax4.legend()
        # self.ax4.axis('equal')


    def update_plot(self, robot_graph_data, clear_output=False):
        if not robot_graph_data or robot_graph_data is None or len(robot_graph_data) <= self.last_plotted_index:
            return

        new_data = robot_graph_data[self.last_plotted_index:]

        # Update plots with only the new data
        self.update_path_plot(new_data)
        self.update_duty_cycle_plot(new_data)
        self.update_velocity_plot(new_data)
        self.update_position_plot(new_data)

        self.last_plotted_index = len(robot_graph_data)

        # Update title with current time
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.ax4.set_title(f"Robot Positions. t={duration:.2f} sec")

        self.fig.canvas.draw()
        plt.pause(0.001)

        if clear_output:
            display.clear_output(wait=True)
        display.display(self.fig)

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
        self.ax1.axis('equal')
        self.ax1.autoscale_view()

    def update_duty_cycle_plot(self, new_data):
        if not new_data:
            return
        new_duty_cycles = np.array([d.duty_cycle_commands for d in new_data])
        current_data = self.duty_cycle_lines[0].get_data()
        updated_x = np.append(current_data[0], range(len(current_data[0]), len(current_data[0]) + len(new_duty_cycles)))

        for i, line in enumerate(self.duty_cycle_lines):
            updated_y = np.append(line.get_ydata(), new_duty_cycles[:, i])
            if len(updated_y) > self.max_points:
                updated_x = updated_x[-self.max_points:]
                updated_y = updated_y[-self.max_points:]
            line.set_data(updated_x, updated_y)

        self.ax2.relim()
        self.ax2.autoscale_view()

    def update_velocity_plot(self, new_data):
        if not new_data:
            return
        new_velocities = np.array([d.current_wheel_w for d in new_data])
        new_desired_velocities = np.array([d.target_wheel_w for d in new_data])
        current_data = self.velocity_lines[0].get_data()
        updated_x = np.append(current_data[0], range(len(current_data[0]), len(current_data[0]) + len(new_velocities)))

        for i, (v_line, dv_line) in enumerate(zip(self.velocity_lines, self.desired_velocity_lines)):
            updated_v = np.append(v_line.get_ydata(), new_velocities[:, i])
            updated_dv = np.append(dv_line.get_ydata(), new_desired_velocities[:, i])
            if len(updated_v) > self.max_points:
                updated_x = updated_x[-self.max_points:]
                updated_v = updated_v[-self.max_points:]
                updated_dv = updated_dv[-self.max_points:]
            v_line.set_data(updated_x, updated_v)
            dv_line.set_data(updated_x, updated_dv)

        self.ax3.relim()
        self.ax3.autoscale_view()


    def update_position_plot(self, new_data):
        if not new_data:
            return
        new_poses = np.array([d.pose for d in new_data])
        new_goals = np.array([d.goal_position for d in new_data])

        current_actual = self.actual_path_line.get_data()
        current_goal = self.goal_path_line.get_data()

        updated_actual_x = np.append(current_actual[0], new_poses[:, 0])
        updated_actual_y = np.append(current_actual[1], new_poses[:, 1])
        updated_goal_x = np.append(current_goal[0], new_goals[:, 0])
        updated_goal_y = np.append(current_goal[1], new_goals[:, 1])

        self.actual_path_line.set_data(updated_actual_x, updated_actual_y)
        self.goal_path_line.set_data(updated_goal_x, updated_goal_y)

        self.ax4.relim()
        self.ax4.axis('equal')
        # self.ax4.autoscale_view()


# Usage
if __name__ == '__main__':
    plotter = RobotPlotter()
    # In your main loop or update function:
    # plotter.update_plot(robot_graph_data)