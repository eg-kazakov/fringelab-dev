import unittest
import numpy as np
from src.fringelab import Fringe


class TestFringe(unittest.TestCase):
    def setUp(self):
        self.size = 400

        self.A_func = lambda c: lambda x,y: c
        self.B_func = lambda c: lambda x,y: c
        self.Fi_func = lambda c, step: lambda x,y: c * np.pi * (x**2 + y**2) + step

        self.fringe = Fringe(self.size, self.A_func(1), self.B_func(0.9), self.Fi_func(1,0))

    def test_generate(self):
        self.assertIsNotNone(self.fringe.intensity)
        self.assertEqual(self.fringe.intensity.shape, (self.size, self.size))

    def test_save_and_load_full(self):
        self.fringe.save_full('test_fringe.pkl')
        loaded_fringe = Fringe.load_full('test_fringe.pkl')
        self.assertEqual(loaded_fringe.size, self.size)

    def test_save_png(self):
        self.fringe.save_png('test_fringe.png')

if __name__ == '__main__':
    unittest.main()
