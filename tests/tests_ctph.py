import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.phase.ctph import ctph


class TestSteadyState(unittest.TestCase):
    def test_ctph_pdf(self):
        alpha = np.array([0.5, 0.5])
        T = np.array([[-2, 0], [0, -5]])
        ph = ctph(alpha, T)
        assert_allclose(ph.pdf(0.2), 1.590018648964245,err_msg="should be 1.590018648964245")

class TestIsErgodic(unittest.TestCase):
    def test_ctph_cdf(self):
        alpha = np.array([0.5, 0.5])
        T = np.array([[-2, 0], [0, -5]])
        ph = ctph(alpha, T)
        assert_allclose(ph.cdf(0.2), 0.480900256396459,err_msg="should be 0.480900256396459")

if __name__ == '__main__':
    unittest.main()