import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.ctbd import ctbd


class TestSteadyState(unittest.TestCase):
    def test_ctdb_steady_state(self):
        birth = np.array([1])
        death = np.array([2])
        bd = ctbd(birth, death)
        assert_allclose(bd.steady_state(), [0.5, 0.25], 
                        err_msg="should be [0.5, 0.25]")

    def test_ctdb_steady_state5(self):
        birth = np.array([1])
        death = np.array([2])
        n=5
        bd = ctbd(birth, death)
        assert_allclose(bd.steady_state(5), [0.5, 0.25, 0.125, 0.0625, 0.03125], 
                        err_msg="should be [0.5, 0.25, 0.125, 0.0625, 0.03125]")

class TestIsErgodic(unittest.TestCase):
    def test_ctdb_isergodic(self):
        birth = np.array([1])
        death = np.array([2])
        bd = ctbd(birth, death)
        self.assertTrue(bd.is_ergodic())

if __name__ == '__main__':
    unittest.main()