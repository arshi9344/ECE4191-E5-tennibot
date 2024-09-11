import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow




class TentaclePlanner:
    def __init__(
            self,
            obstacles=[],
            dt=0.1,
            steps=15,  # Number of steps the planner uses when simulating a path #CHANGE
            alpha=1,  # Weighs the positional error in the cost function. Higher value prioritises goal positions.
            beta=0.1,
            # Weighs the rotational error of the cost function. higher value prioritises correct final orientation.
            max_linear_velocity=0.23,  # 0.23 m/s is at about 75% duty cycle.
            max_angular_velocity=2.1,  # 2.1 rad/s is about 78% duty cycle
            max_linear_tolerance=0.1,  # meters
            max_angular_tolerance=0.1,  # radians, 0.15 rad = 8.6 deg, 0.08 = 4.6 deg
            max_acceleration=0.3,
            max_angular_acceleration=0.5,
            current_linear_velocity=0.0,
            current_angular_velocity=0.0

    ):
        self.dt = dt
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.obstacles = np.array(obstacles)
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_tolerance = max_linear_tolerance
        self.max_angular_tolerance = max_angular_tolerance
        self.max_acceleration = max_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        self.current_linear_velocity = current_linear_velocity
        self.current_angular_velocity = current_angular_velocity

        self.tentacles = self._generate_tentacles()

    def _generate_tentacles(self):
        # Increased resolution for velocities
        linear_velocities = np.linspace(-self.max_linear_velocity, self.max_linear_velocity, 30)
        angular_velocities = np.linspace(-self.max_angular_velocity, self.max_angular_velocity, 30)

        # Basic tentacles with more density
        tentacles = [(v, w) for v in linear_velocities for w in angular_velocities]

        # Fine granularity near zero for more precise control
        fine_linear = np.linspace(-0.1 * self.max_linear_velocity, 0.1 * self.max_linear_velocity, 5)
        fine_angular = np.linspace(-0.1 * self.max_angular_velocity, 0.1 * self.max_angular_velocity, 5)
        tentacles.extend([(v, w) for v in fine_linear for w in fine_angular])

        # Pure linear and pure rotational motions
        tentacles.extend([(v, 0) for v in linear_velocities if v != 0])
        tentacles.extend([(0, w) for w in angular_velocities if w != 0])

        # Random perturbations for exploration
        random_perturbations = [(v + np.random.uniform(-0.05, 0.05), w + np.random.uniform(-0.05, 0.05))
                                for v, w in tentacles]
        tentacles.extend(random_perturbations)

        # Remove duplicates and return
        return list(set(tentacles))


    def _is_goal_reached(self, goal_x, goal_y, goal_th, x, y, th):
        distance_to_goal = np.hypot(goal_x - x, goal_y - y)
        angular_error = np.arctan2(np.sin(goal_th - th), np.cos(goal_th - th))
        return distance_to_goal <= self.max_linear_tolerance  # and abs(angular_error) <= self.max_angular_tolerance


    def _trapezoidal_profile(self, distance, current_velocity, max_velocity, max_acceleration):
        acceleration_distance = (max_velocity ** 2) / (2 * max_acceleration)
        if distance == 0:
            velocity = current_velocity + max_acceleration * self.dt
        elif 0 < distance <= 2 * acceleration_distance:
            t_accel = np.sqrt(distance / max_acceleration)
            velocity = min(current_velocity + max_acceleration * t_accel, max_velocity)
        else:
            velocity = max_velocity
        return velocity

    def _roll_out(self, v, w, goal_x, goal_y, goal_th, x, y, th):
        for _ in range(self.steps):
            x += self.dt * v * np.cos(th)
            y += self.dt * v * np.sin(th)
            th += w * self.dt

            if self._check_collision(x, y):
                return np.inf

        dist2goal = np.hypot(goal_x - x, goal_y - y)

        e_th = np.arctan2(np.sin(goal_th - th), np.cos(goal_th - th))

        cost = self.alpha * dist2goal + self.beta * abs(e_th)

        return cost

    def _check_collision(self, x, y):
        if len(self.obstacles) == 0:
            return False
        min_dist = np.min(np.hypot(x - self.obstacles[:, 0], y - self.obstacles[:, 1]))
        return min_dist < 0.1

    def _angle_between_points(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.arctan2(dy, dx)

    def _normalise_angle(self, angle):  # in radians
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _shortest_angular_distance(self, from_angle, to_angle):
        """
        Calculate the shortest angular distance between two angles.
        """
        diff = self._normalise_angle(to_angle - from_angle)
        if diff > np.pi:
            diff -= 2 * np.pi
        return diff


    def _plan_tentacles(self, goal_x, goal_y, goal_th, x, y, th):

        #         current_goal = goals[current_goal_index]
        #         goal_x, goal_y, goal_th = current_goal

        #         if self.is_goal_reached(goal_x, goal_y, goal_th, x, y, th):
        #             # Move to the next goal if available
        #             if current_goal_index < len(goals) - 1:
        #                 current_goal_index += 1
        #                 print(f"Reached goal {current_goal_index-1}, moving to goal {current_goal_index}")
        #                 current_goal = goals[current_goal_index]
        #                 goal_x, goal_y, goal_th = current_goal
        #             else:
        #             return (0.0, 0.0)  # No more goals to process

        if self._is_goal_reached(goal_x, goal_y, goal_th, x, y, th):
            return (0.0, 0.0)

        tentacle_num = 15
        angle_to_goal = self._angle_between_points((x, y), (goal_x, goal_y))
        angle_diff = self._shortest_angular_distance(th, angle_to_goal)

        # Define tentacles for turning
        turning_tentacles = [(0, w) for w in
                             np.linspace(-self.max_angular_velocity, self.max_angular_velocity, tentacle_num) if w != 0]
        best_turn_tentacle = min(turning_tentacles, key=lambda t: abs(angle_diff - t[1]))
        best_turn_rate = best_turn_tentacle[1]

        if abs(angle_diff) > self.max_angular_tolerance:
            return (0.0, best_turn_rate)

        # If aligned, use linear tentacles
        linear_tentacles = [(v, 0) for v in np.linspace(0, self.max_linear_velocity, tentacle_num)]
        best_linear_tentacle = min(linear_tentacles,
                                   key=lambda t: self._roll_out(t[0], 0, goal_x, goal_y, goal_th, x, y, th))

        if abs(angle_diff) <= self.max_angular_tolerance:
            return (best_linear_tentacle[0], 0.0)

        return (best_linear_tentacle[0], best_turn_rate)

    def _rotate(self, goal_x, goal_y, goal_th, x, y, th):
        # Calculate the shortest angular distance

        value = 0.5

        diff = goal_th - th
        angle_diff = (diff + np.pi) % (2 * np.pi) - np.pi

        if abs(angle_diff) > 0.05:
            w = -abs(value) if angle_diff < 0 else abs(value)
            return 0.0, w

        else:
            return 0.0, 0.0

    def get_control_inputs(self, goal_x, goal_y, goal_th, x, y, th, strategy='tentacles'):
        if strategy == 'rotate':
            v, w = self._rotate(goal_x, goal_y, goal_th, x, y, th)
        elif strategy == 'tentacles':
            v, w = self._plan_tentacles(goal_x, goal_y, goal_th, x, y, th)
        else:
            raise ValueError("Invalid strategy. Must be 'rotate_then_drive' or 'tentacles' or 'smooth_pursuit'.")

        self.current_linear_velocity = v
        self.current_angular_velocity = w
        return {
            'linear_velocity': v,
            'angular_velocity': w,
            'current_x': x,
            'current_y': y,
            'current_theta': th,
            'goal_x': goal_x,
            'goal_y': goal_y,
            'goal_theta': goal_th
        }


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

                if _ == self.steps // 2:
                    dx, dy = x_new - x_traj[-2], y_new - y_traj[-2]
                    arrow = Arrow(x_traj[-2], y_traj[-2], dx, dy, width=0.2, color=color)
                    plt.gca().add_patch(arrow)

            plt.plot(x_traj, y_traj, '-', color=color, linewidth=2, label=f'v={v:.2f}, w={w:.2f}')

        if len(self.obstacles) > 0:
            plt.scatter(self.obstacles[:, 0], self.obstacles[:, 1], color='red', s=50, label='Obstacles')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Tentacle Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def visualize_plan(self, goal_x, goal_y, goal_th, x, y, th, use_straight_line=False, extrapolation_steps=100):
        v, w = self.plan(goal_x, goal_y, goal_th, x, y, th, use_straight_line)

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

        if len(self.obstacles) > 0:
            plt.scatter(self.obstacles[:, 0], self.obstacles[:, 1], color='red', s=50, label='Obstacles')
        plt.legend()
        plt.title(f'Planned Path (v={v:.2f}, w={w:.2f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()




# # Example usage

# # Create an instance of the planner
# obstacles = np.array([[1, 1], [2, 2], [3, 1]])  # Example obstacles
# planner = TentaclePlanner(obstacles, max_linear_velocity=1.0, max_angular_velocity=0.5)

# # Get control inputs for tentacle-based planning
# inputs = planner.get_control_inputs(goal_x=5, goal_y=5, goal_th=0, x=0, y=0, th=0)
# print("Tentacle-based planning:", inputs)

# # Get control inputs for straight-line planning
# inputs_straight = planner.get_control_inputs(goal_x=5, goal_y=5, goal_th=0, x=0, y=0, th=0, use_straight_line=True)
# print("Straight-line planning:", inputs_straight)

# # Visualize the plan
# planner.visualize_plan(goal_x=5, goal_y=5, goal_th=0, x=0, y=0, th=0)
# planner.visualize_plan(goal_x=5, goal_y=5, goal_th=0, x=0, y=0, th=0, use_straight_line=True)