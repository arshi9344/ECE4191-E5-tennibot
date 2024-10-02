import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle



class ScanPoint:
    def __init__(self, x, y, rotation_limited: bool = False, rotation_bounds: tuple = (0, 0)):
        self.x = x
        self.y = y
        self.rotation_limited = rotation_limited
        self.rotation_bounds = rotation_bounds

    @property
    def coords(self):
        return self.x, self.y

    def __repr__(self):
        return f"Scan Point: ({self.x}, {self.y}). Limited?: {self.rotation_limited}, Rotation: {self.rotation_bounds}"


def _filter_array_inplace(arr, min_value, max_value):
    indices_to_remove = np.where((arr <= min_value) | (arr >= max_value))[0]
    return np.delete(arr, indices_to_remove)


class ScanPointGenerator:
    def __init__(self, x_lim=4.12, y_lim=5.48, scan_radius=2, flip_x: bool = False, flip_y: bool = False,):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.scan_radius = scan_radius
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.points = self.generate_scan_points(x_lim, y_lim, scan_radius, flip_x, flip_y)


    def generate_scan_points(self, x_lim, y_lim, scan_radius, flip_x=False, flip_y=False, remove_points_on_boundary=True):
        x_coords = np.arange(0, x_lim + scan_radius, scan_radius)
        y_coords = np.arange(0, y_lim + scan_radius, scan_radius)
        # Clip the last points to ensure they don't exceed the rectangle dimensions
        # scan_lines = np.clip(scan_lines, 0, x_lim)
        # scan_points = np.clip(scan_points, 0, y_lim)

        if remove_points_on_boundary:
            x_coords = _filter_array_inplace(x_coords, 0, x_lim)
            y_coords = _filter_array_inplace(y_coords, 0, y_lim)

        if flip_x:
            x_coords = -x_coords

        if flip_y:
            y_coords = -y_coords

        scans = []
        for i, x in enumerate(x_coords):
            if i % 2 == 0:
                for y in y_coords:
                    scans.append(ScanPoint(x, y))
            else:
                for y in y_coords[::-1]:
                    scans.append(ScanPoint(x, y))

        return scans

    def plot_scan_points(self, ax=None):
        width = self.x_lim if not self.flip_x else -self.x_lim
        depth = self.y_lim if not self.flip_y else -self.y_lim

        if not ax:
            fig, ax = plt.subplots(figsize=(12, 10))

        # Plot rectangle
        rect = Rectangle((0, 0), width, depth, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

        # Plot scan points and circles
        for point in self.points:
            ax.scatter(*point.coords, color='blue', s=30)
            circle = Circle(point.coords, self.scan_radius, fill=False, edgecolor='purple', alpha=0.5)
            ax.add_patch(circle)

        # Set plot limits and labels
        # ax.set_xlim(0, width)
        # ax.set_ylim(0, depth)
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Scan Points and Lines with Coverage Areas')

        # Add grid and legend
        ax.grid(True, linestyle=':')
        ax.legend([rect, plt.Line2D([0], [0], color='red', linestyle='--'),
                   plt.Line2D([0], [0], marker='o', color='blue', linestyle='None'),
                   Circle((0, 0), 1, fill=False, edgecolor='purple')],
                  ['Area', 'Scan Lines', 'Scan Points', 'Scan Coverage'],
                  loc='upper right')

        plt.tight_layout()
        plt.show()


# # Parameters
if __name__ == '__main__':
    width = 4.12
    depth = 5.48
    max_scan_distance = 2
    flip_x = True
    flip_y = False
    # Generate scan points and lines
    scan_gen = ScanPointGenerator(width, depth, max_scan_distance, flip_x, flip_y)
    print(scan_gen.points)
    # Plot the points
    scan_gen.plot_scan_points()