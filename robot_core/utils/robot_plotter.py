import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
from IPython import display

"""
- The RobotPlotter class is responsible for plotting the robot's pose, duty cycle, wheel velocity, and position data.

NOTES:
- Plotting is a blocking operation as it stands. This is because an instance of RobotPlotter is held by Coordinator, and the update_plot method is called in the main loop.
- This means that plotting can affect the performance of the high-level decision logic, including the robot's ability to respond to new data by setting goals.
- I was thinking we could get around this by using Python's threading module to run the plotting in a separate thread. This would allow the main loop to continue running without being blocked by the plotting code.
- Alternately, we could move the plotting code to a separate process using Python's multiprocessing module. This would also allow the main loop to continue running without being blocked by the plotting code.
- I'm not sure which approach would be better, but I think threading might be simpler to implement. We could try that first and see how it goes.

"""

class RobotPlotter:
    def __init__(self, max_time_window=10):  # Changed from max_points to max_time_window (in seconds)
        self.max_time_window = max_time_window # Time window in seconds for duty cycle and wheel velocity plots (otherwise gets crowded).
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

        self.counter = 0

    def label_plots(self):
        self.ax1.set_xlabel('x-position (m)')
        self.ax1.set_ylabel('y-position (m)')
        self.ax1.set_title('Robot Pose Over Time')
        self.ax1.grid(True)

        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Duty Cycle')
        self.ax2.set_title('Duty Cycle Commands Over Time')
        self.ax2.legend()
        self.ax2.grid(True)

        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Wheel Velocity (rad/s)')
        self.ax3.set_title('Wheel Velocity vs. Time')
        self.ax3.legend()
        self.ax3.grid(True)

        self.ax4.set_xlabel('x-position (m)')
        self.ax4.set_ylabel('y-position (m)')
        self.ax4.set_title('Robot Positions')
        self.ax4.grid(True)
        self.ax4.legend()

    def update_plot(self, robot_graph_data, clear_output=False):
        if not robot_graph_data or len(robot_graph_data) <= self.last_plotted_index:
            return

        new_data = robot_graph_data[self.last_plotted_index:]

        if self.start_time is None:
            self.start_time = new_data[0].time

        self.update_path_plot(new_data)
        self.update_duty_cycle_plot(new_data)
        self.update_velocity_plot(new_data)
        self.update_position_plot(new_data)

        self.last_plotted_index = len(robot_graph_data)

        duration = new_data[-1].time - self.start_time
        self.ax4.set_title(f"Robot Positions. t={duration:.2f} sec")

        self.fig.canvas.draw()
        plt.pause(0.001)

        ### debug
        # if self.counter % 5 == 0:
        #     print(f"length of new_data: {len(new_data)}")
        #     print(f"new_data[-1] pose: {new_data[-1].pose}")
        #     print(f"new_data[-1] duty_cycle_commands: {new_data[-1].duty_cycle_commands}")
        #     print(f"last_plotted_index: {self.last_plotted_index}")

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
        self.ax1.autoscale_view()

    def update_duty_cycle_plot(self, new_data):
        if not new_data:
            return

        new_times = np.array([d.time - self.start_time for d in new_data]) # convert time since epoch into time since robot start
        new_duty_cycles = np.array([d.duty_cycle_commands for d in new_data]) #

        for i, line in enumerate(self.duty_cycle_lines): # Loop through left and right wheels
            current_x, current_y = line.get_data()
            updated_x = np.append(current_x, new_times)
            updated_y = np.append(current_y, new_duty_cycles[:, i])

            # Apply time window
            mask = updated_x >= (new_times[-1] - self.max_time_window)
            line.set_data(updated_x[mask], updated_y[mask])

            ## debug
            # if self.counter % 5 == 0:
            #     print(f"new_times: {new_times[-10:]}")
            #     print(f"updated_y: {updated_y[-10:]}")

        ax_left_limit = 0 if new_times[-1] - self.max_time_window < 0 else new_times[-1] - self.max_time_window
        # print(f"ax_left_limit {ax_left_limit}")
        self.ax2.set_xlim(ax_left_limit, new_times[-1])
        self.ax2.relim()
        self.ax2.autoscale_view()
        ## debug
        self.counter += 1


    def update_velocity_plot(self, new_data):
        if not new_data:
            return
        new_times = np.array([d.time - self.start_time for d in new_data])
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

        ax_left_limit = 0 if new_times[-1] - self.max_time_window < 0 else new_times[-1] - self.max_time_window
        self.ax3.set_xlim(ax_left_limit, new_times[-1])
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
        self.ax4.autoscale_view()


# Usage
if __name__ == '__main__':
    plotter = RobotPlotter(max_time_window=60)  # Show last 60 seconds of data
    # In your main loop or update function:
    # plotter.update_plot(robot_graph_data)