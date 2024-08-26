import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow


"""
This is the TentaclePlanner class that allows us to plan a path for a robot to follow.
This is based off of Michael's TentaclePlanner, except with a few additional methods to visualise the paths.
"""
class TentaclePlanner:

    def __init__(self, obstacles, dt=0.1, steps=5, alpha=1, beta=0.1):

        self.dt = dt
        self.steps = steps
        # Tentacles are possible trajectories to follow
        self.tentacles = [(0.0, 1.0), (0.0, -1.0), (0.1, 1.0), (0.1, -1.0), (0.1, 0.5), (0.1, -0.5), (0.1, 0.0),
                          (0.0, 0.0)]

        self.alpha = alpha
        self.beta = beta

        self.obstacles = obstacles

    # Play a trajectory and evaluate where you'd end up
    def roll_out(self, v, w, goal_x, goal_y, goal_th, x, y, th):

        for j in range(self.steps):

            x = x + self.dt * v * np.cos(th)
            y = y + self.dt * v * np.sin(th)
            th = (th + w * self.dt)

            if (self.check_collision(x, y)):
                return np.inf

        # Wrap angle error -pi,pi
        e_th = goal_th - th
        e_th = np.arctan2(np.sin(e_th), np.cos(e_th))

        cost = self.alpha * ((goal_x - x) ** 2 + (goal_y - y) ** 2) + self.beta * (e_th ** 2)

        return cost

    def check_collision(self, x, y):

        min_dist = np.min(np.sqrt((x - self.obstacles[:, 0]) ** 2 + (y - self.obstacles[:, 1]) ** 2))

        if (min_dist < 0.1):
            return True
        return False

    # Choose trajectory that will get you closest to the goal
    def plan(self, goal_x, goal_y, goal_th, x, y, th):

        costs = []
        for v, w in self.tentacles:
            costs.append(self.roll_out(v, w, goal_x, goal_y, goal_th, x, y, th))

        best_idx = np.argmin(costs)

        return self.tentacles[best_idx]

    def visualize_tentacles(self, x, y, th, extrapolation_steps=100):
        plt.figure(figsize=(12, 12))
        plt.scatter(x, y, color='green', s=100, label='Start')

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.tentacles)))

        for (v, w), color in zip(self.tentacles, colors):
            x_traj, y_traj = [x], [y]
            th_traj = th
            for _ in range(self.steps):
                x_new = x_traj[-1] + self.dt * v * np.cos(th_traj) * extrapolation_steps
                y_new = y_traj[-1] + self.dt * v * np.sin(th_traj) * extrapolation_steps
                x_traj.append(x_new)
                y_traj.append(y_new)
                th_traj += w * self.dt

                # Add an arrow to show direction
                if _ == self.steps // 2:  # Add arrow in the middle of the trajectory
                    dx = x_new - x_traj[-2]
                    dy = y_new - y_traj[-2]
                    arrow = Arrow(x_traj[-2], y_traj[-2], dx, dy, width=0.2, color=color)
                    plt.gca().add_patch(arrow)

            plt.plot(x_traj, y_traj, '-', color=color, linewidth=2, label=f'v={v:.2f}, w={w:.2f}')

        plt.scatter(self.obstacles[:, 0], self.obstacles[:, 1], color='red', s=50, label='Obstacles')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Tentacle Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def visualize_plan(self, goal_x, goal_y, goal_th, x, y, th, extrapolation_steps=100):
        v, w = self.plan(goal_x, goal_y, goal_th, x, y, th)

        plt.figure(figsize=(10, 10))
        plt.scatter(x, y, color='green', s=100, label='Start')
        plt.scatter(goal_x, goal_y, color='blue', s=100, label='Goal')

        x_traj, y_traj = [x], [y]
        th_traj = th
        for _ in range(self.steps):
            x_traj.append(x_traj[-1] + self.dt * v * np.cos(th_traj) * extrapolation_steps)
            y_traj.append(y_traj[-1] + self.dt * v * np.sin(th_traj) * extrapolation_steps)
            th_traj += w * self.dt
        plt.plot(x_traj, y_traj, 'g-', linewidth=2, label='Chosen Path')

        plt.scatter(self.obstacles[:, 0], self.obstacles[:, 1], color='red', s=50, label='Obstacles')
        plt.legend()
        plt.title(f'Planned Path (v={v:.2f}, w={w:.2f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


"""
# Create a TentaclePlanner instance
obstacles = np.array([[1, 1], [2, 2], [3, 1]])  # Example obstacles
planner = TentaclePlanner(obstacles)

# Visualize all tentacles from a starting position
planner.visualize_tentacles(0, 0, 0)

# Visualize the planned path to a goal
planner.visualize_plan(5, 5, np.pi / 4, 0, 0, 0)
"""

# # Create a TentaclePlanner instance
# obstacles = np.array([[1, 1], [2, 2], [3, 1]])  # Example obstacles
# planner = TentaclePlanner(obstacles)

# # Visualize all tentacles from a starting position
# planner.visualize_tentacles(0, 0, 0)

# # Visualize the planned path to a goal
# planner.visualize_plan(5, 5, np.pi / 4, 0, 0, 0)