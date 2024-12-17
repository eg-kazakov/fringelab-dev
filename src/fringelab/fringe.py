import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class Fringe:
    def __init__(self, size, A_func, B_func, Fi_func):
        self.size = size
        self.A_func = A_func
        self.B_func = B_func
        self.Fi_func = Fi_func

        # Initialize empty matrices for intensity and intermediate values
        self.intensity = None
        self.A = None
        self.B = None
        self.Fi = None

        self.generate()

    def generate(self):
        """Generates the intensity and intermediate matrices."""
        size = self.size
        A = np.empty([size, size])
        B = np.empty([size, size])
        Fi = np.empty([size, size])

        for i in range(size):
            for j in range(size):
                x = i / (size / 2) - 1
                y = j / (size / 2) - 1
                Fi[i][j] = self.Fi_func(x, y)
                A[i][j] = self.A_func(x, y)
                B[i][j] = self.B_func(x, y)

        I = A + B * np.cos(Fi)

        # Store matrices
        self.A = pd.DataFrame(A)
        self.B = pd.DataFrame(B)
        self.Fi = pd.DataFrame(Fi)
        self.intensity = pd.DataFrame(I)

    def save_full(self, filename):
        """Save the Fringe object including its metadata and functions."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_full(filename):
        """Load a Fringe object from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save_png(self, filename):
        """Save the intensity matrix as a PNG image."""
        plt.imshow(self.intensity, cmap='gray', norm=Normalize(vmin=self.intensity.min().min(), vmax=self.intensity.max().max()))
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def load_png(filename):
        """Load an intensity matrix from a PNG file."""
        image = plt.imread(filename)
        return pd.DataFrame(image)

    def plot(self):
        """Visualize the intensity matrix."""
        plt.imshow(self.intensity, cmap='gray', norm=Normalize(vmin=self.intensity.min().min(), vmax=self.intensity.max().max()))
        plt.axis('off')
        return plt
