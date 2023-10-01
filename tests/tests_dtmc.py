import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.dtmc import dtmc


class TestSteadyState(unittest.TestCase):
    def test_dtmc_steady_state(self):
        P = np.array([[0.2, 0.8, 0], [0.3, 0.4, 0.3], [0.4, 0, 0.6]])
        mc = dtmc(P)
        assert_allclose(mc.steady_state(), [0.3, 0.4, 0.3],err_msg="should be [0.3, 0.4, 0.3]")


if __name__ == '__main__':
    unittest.main()
