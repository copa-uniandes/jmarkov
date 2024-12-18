import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.mmk import mmk


class TestSteadyState(unittest.TestCase):
    def test_mmk_mean_number_system(self):
        lda = 1 
        mu = 2
        k =3
        q = mmk(k,lda,mu) 
        assert_almost_equal(q.mean_number_entities(), 0.5030303030, 
                        err_msg="should be 0.5030303030]")

    def test_mmk_mean_number_queue(self):
        lda = 1 
        mu = 2
        k =3
        q = mmk(k,lda,mu) 
        assert_almost_equal(q.mean_number_entities_queue(), 0.0030303030, 
                        err_msg="should be 0.0030303030]")


if __name__ == '__main__':
    unittest.main()