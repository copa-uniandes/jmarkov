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

class TestTransientProbabilities(unittest.TestCase):
    def test_ctmc_transient_probabilities(self):
        Q = np.array([[-4, 1, 3], [2, -5, 3], [1, 2, -3]])
        mc = ctmc(Q)
        assert_allclose(mc.transient_probabilities(3,np.array([1,0,0])),np.array([0.25000003, 0.24999997, 0.49999999]))    

class TestOccupationTime(unittest.TestCase):
    def test_ctmc_occupation_time(self):
        Q = np.array([[-4,1,3],[2,-5,3],[1,2,-3]])
        mc = ctmc(Q)
        assert_allclose(mc.occupation_time(4),np.array([[1.13888654, 0.9444421 , 1.91666197],
                                                        [0.97221987, 1.11110876, 1.91666197],
                                                        [0.9444421 , 0.97221987, 2.08332863]]))
class TestIsIrreducible(unittest.TestCase):
    def test_ctmc_is_irreducible(self):
        Q = np.array([[-4,1,3],[2,-5,3],[1,2,-3]])
        mc = ctmc(Q)
        self.assertTrue(mc.is_irreducible())

if __name__ == '__main__':
    unittest.main()