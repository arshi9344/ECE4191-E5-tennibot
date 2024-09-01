"""
This file contains the function to update a matplotlib.plt plot in real-time.
"""

import numpy as np
from IPython import display


def update_plot(orchestrator, fig, axes):
    axes_flat = axes.flatten()  # Flatten the 2D array of axes
    for ax in axes_flat:
        ax.clear()

    # Plot 1: Robot path and orientation
    poses = np.array(orchestrator.poses)
    if len(poses) > 0:
        axes[0, 0].plot(np.array(poses)[:, 0], np.array(poses)[:, 1])
        x, y, th = poses[-1]
        axes[0, 0].plot(x, y, 'k', marker='+')
        axes[0, 0].quiver(x, y, 0.1 * np.cos(th), 0.1 * np.sin(th))
    axes[0, 0].set_xlabel('x-position (m)')
    axes[0, 0].set_ylabel('y-position (m)')
    axes[0, 0].set_title(f"Robot Pose Over Time. Kp: {orchestrator.controller.Kp}, Ki: {orchestrator.controller.Ki}")
    axes[0, 0].axis('equal')
    axes[0, 0].grid()

    # Plot 2: Duty cycle commands
    duty_cycle_commands = np.array(orchestrator.duty_cycle_commands)
    if len(duty_cycle_commands) > 0:
        duty_cycle_commands = np.array(duty_cycle_commands)
        axes[0, 1].plot(duty_cycle_commands[:, 0], label='Left Wheel')
        axes[0, 1].plot(duty_cycle_commands[:, 1], label='Right Wheel')

    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Duty Cycle')
    axes[0, 1].set_title('Duty Cycle Commands Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid()

    # Plot 3: Wheel velocities
    velocities = np.array(orchestrator.velocities)
    desired_velocities = np.array(orchestrator.desired_velocities)
    if len(velocities) > 0 and len(desired_velocities) > 0:
        axes[1, 0].plot(velocities[:, 0], label='Left Wheel')
        axes[1, 0].plot(velocities[:, 1], label='Right Wheel')
        axes[1, 0].plot(desired_velocities[:, 0], label='Desired Left Wheel')
        axes[1, 0].plot(desired_velocities[:, 1], label='Desired Right Wheel')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Wheel Velocity (rad/s)')
    axes[1, 0].set_title('Wheel Velocity vs. Time')
    axes[1, 0].legend()
    axes[1, 0].grid()

    # Plot 4: Goal Positions vs. actual position
    goal_positions = np.array(orchestrator.goal_positions)
    poses = np.array(orchestrator.poses)

    # Add (0, 0) to both goal_positions and poses
    goal_positions = np.vstack(((0, 0, 0), goal_positions))
    poses = np.vstack(([0, 0, 0], poses))
    scan_locations = np.array(orchestrator.scan_locations)

    axes[1, 1].plot(0, 0, 'ko', markersize=10, label='Start (0, 0)')  # Add point at (0, 0)

    if len(poses) > 0:
        axes[1, 1].plot(poses[:, 0], poses[:, 1], 'b-', label='Actual Path')
        axes[1, 1].scatter(poses[:, 0], poses[:, 1], color='b', s=5)  # Add dots for each position with custom size
    if len(goal_positions) > 1:
        axes[1, 1].plot(goal_positions[:, 0], goal_positions[:, 1], 'r--', label='Goal Path')
        axes[1, 1].plot(goal_positions[:, 0], goal_positions[:, 1], 'r.')  # Add dots for each goal position
    if len(scan_locations) > 1:
        axes[1, 1].scatter(scan_locations[:, 0], scan_locations[:, 1], color='g', s=20,
                           label='Scan Locations')  # Add dots for each scan position

    axes[1, 1].set_xlabel('x-position (m)')
    axes[1, 1].set_ylabel('y-position (m)')
    axes[1, 1].set_title(f"Robot Positions. {orchestrator.duration:.2f} sec run.")
    axes[1, 1].axis('equal')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.canvas.draw()
    #     display.clear_output(wait=True)
    display.display(fig)