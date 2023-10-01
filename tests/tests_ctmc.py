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


if __name__ == '__main__':
    unittest.main()