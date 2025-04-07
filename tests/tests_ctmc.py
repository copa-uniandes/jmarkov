import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.ctmc import ctmc


class TestSteadyState(unittest.TestCase):
    def test_ctmc_steady_state(self):
        Q = np.array([[-4, 1, 3], [2, -5, 3], [1, 2, -3]])
        mc = ctmc(Q)
        assert_allclose(mc.steady_state(), [0.25, 0.25, 0.5],err_msg="should be [0.25, 0.25, 0.5]")

class TestIsErgodic(unittest.TestCase):
    def test_ctmc_isergodic(self):
        Q = np.array([[-4, 1, 3], [2, -5, 3], [1, 2, -3]])
        mc = ctmc(Q)
        self.assertTrue(mc.is_ergodic())

class TestFirstPassageTime(unittest.TestCase):
    def test_ctmc_first_passage_time(self):
        Q = np.array([[-3, 2, 1], [2, -5, 3], [1, 1, -2]])
        mc = ctmc(Q)
        assert_allclose(mc.first_passage_time(0), [[0.71428571], [0.85714286]], err_msg="should be [0.71428571, 0.85714286]")

class TestAbsorbtionTimes(unittest.TestCase):
    def test_absorbtion_times(self):
        Q = np.array([[-4, 1, 3], [2, -5, 3], [0, 0, 0]])
        states = np.array([0,1,2])
        mc = ctmc(Q, states)
        self.assertAlmostEqual(mc.absorbtion_times(target=0,start=1)[0][0],0.1111111)

if __name__ == '__main__':
    unittest.main()