import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from .fringe import Fringe


class FringeSet:
    def __init__(self, fringes=None):
        """
        Initialize a FringeSet with an optional list of Fringe objects.
        """
        self.fringes = fringes if fringes else []

    @classmethod
    def create_set_from_base(cls, base_fringe, step, count):
        """
        Create a FringeSet by generating variations from a base fringe.
        :param base_fringe: A Fringe object to use as a base.
        :param step: A step value to modify the phase (Fi_func) of the base fringe.
        :param count: The number of fringes to generate.
        :return: A new FringeSet object.
        """
        def format_step_description(step):
            if step == 0.0:
                return '0'
            fraction = Fraction(step / np.pi).limit_denominator()
            return f"{fraction.numerator}/{fraction.denominator} * π"

        fringes = []
        for i in range(count):
            new_fringe = Fringe(
                size=base_fringe.size,
                A_func=base_fringe.A_func,
                B_func=base_fringe.B_func,
                Fi_func=lambda x, y, phase_shift=i * step: base_fringe.Fi_func(x, y) + phase_shift,
                description=format_step_description(i * step)
            )
            new_fringe.generate()
            fringes.append(new_fringe)
        return cls(fringes)

    @classmethod
    def create_set_from_fringes(cls, *fringes):
        """
        Create a FringeSet from multiple Fringe objects.
        :param fringes: Variable number of Fringe objects.
        :return: A new FringeSet object.
        """
        return cls(list(fringes))

    def get_trajectory(self):
        """
        Generate a trajectory from the FringeSet using up to 3 fringes.
        For 2 fringes, returns a 2D trajectory.
        For 3 fringes, returns a 3D trajectory.
        :return: A Trajectory object.
        """
        if len(self.fringes) < 2:
            raise ValueError("At least two fringes are required to compute a trajectory.")
        elif len(self.fringes) > 3:
            raise ValueError("Trajectory generation supports up to 3 fringes.")

        # Flatten and stack intensities based on number of fringes
        intensity_matrices = [fringe.intensity.values.flatten() for fringe in self.fringes[:3]]
        trajectory_points = np.vstack(intensity_matrices).T  # Stack columns

        return Trajectory(trajectory_points)

    def plot(self, columns=None):
        num_images = len(self.fringes)
        if not columns:
            columns = 2 if num_images%2 else 3
        num_cols = min(num_images, columns)
        num_rows = (num_images + num_cols - 1) // num_cols  # Fix row calculation

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))  # Define vertical size

        #axs = np.atleast_2d(axs)  # Ensure axs is always a 2D array

        for i, (image, desc) in enumerate([(f.intensity, f.description) for f in self.fringes]):
            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col] if num_rows > 1 else axs[col]

            ax.imshow(image, cmap='Spectral')
            ax.set_title(desc)
            fig.colorbar(ax.get_images()[0], ax=ax)

        return fig.tight_layout()

    def __len__(self):
        """Return the number of fringes in the set."""
        return len(self.fringes)


class Trajectory:
    def __init__(self, points):
        """
        Initialize a Trajectory.
        :param points: An array of trajectory points.
        :param dimensions: Number of dimensions (2 or 3).
        """
        self.points = points
        self.dimensions = points.shape[1]

    def get_points(self):
        """
        Return the trajectory points.
        """
        return self.points

    def plot(self):
        """
        Visualize the trajectory:
        - 2D trajectory as a scatter plot using Matplotlib.
        - 3D trajectory using Plotly's Scatter3d.
        """
        if self.dimensions == 2:
            x, y = self.points[:, 0], self.points[:, 1]
            plt.scatter(x, y, c='blue', s=1, alpha=0.7)
            plt.title("2D Trajectory")
            plt.xlabel("I1")
            plt.ylabel("I2")
            plt.grid(True)
        elif self.dimensions == 3:
            from mpl_toolkits.mplot3d import Axes3D  # Import here to avoid dependency if not needed
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
            ax.scatter(x, y, z, c='blue', s=1, alpha=0.7)
            ax.set_title("3D Trajectory")
            ax.set_xlabel("I1")
            ax.set_ylabel("I2")
            ax.set_zlabel("I3")
            return fig.tight_layout()
        else:
            raise ValueError("Trajectory visualization supports only 2D or 3D data.")
        return plt
