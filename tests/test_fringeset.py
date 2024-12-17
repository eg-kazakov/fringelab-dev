import unittest
import numpy as np

from src.fringelab import FringeSet, Fringe


class TestFringeSet(unittest.TestCase):
    def setUp(self):
        size = 50
        A_func = lambda x, y: 1
        B_func = lambda x, y: 1
        Fi_func = lambda x, y: np.pi * (x + y)

        self.base_fringe = Fringe(size, A_func, B_func, Fi_func)
        self.base_fringe.generate()

    def test_get_trajectory_2d(self):
        step = np.pi / 4
        fringe_set = FringeSet.create_set_from_base(self.base_fringe, step, 2)
        trajectory = fringe_set.get_trajectory()
        self.assertEqual(trajectory.dimensions, 2)
        self.assertEqual(trajectory.points.shape[1], 2)

    def test_get_trajectory_3d(self):
        step = np.pi / 4
        fringe_set = FringeSet.create_set_from_base(self.base_fringe, step, 3)
        trajectory = fringe_set.get_trajectory()
        self.assertEqual(trajectory.dimensions, 3)
        self.assertEqual(trajectory.points.shape[1], 3)

    def test_trajectory_visualization_2d(self):
        step = np.pi / 4
        fringe_set = FringeSet.create_set_from_base(self.base_fringe, step, 2)
        trajectory = fringe_set.get_trajectory()
        trajectory.plot()  # Ensure it does not raise errors

    def test_trajectory_visualization_3d(self):
        step = np.pi / 4
        fringe_set = FringeSet.create_set_from_base(self.base_fringe, step, 3)
        fringe_set.plot()
        trajectory = fringe_set.get_trajectory()
        trajectory.plot()  # Ensure it does not raise errors
