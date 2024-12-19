import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.mmkn import mmkn


class TestSteadyState(unittest.TestCase):
    # tests for M/M/1/5
    def test_mm15_mean_number_system(self):
        lda = 1 
        mu = 2
        k =1
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_number_entities(), 0.904761905, 
                        err_msg="should be 0.904761905]")

    def test_mm15_mean_number_queue(self):
        lda = 1 
        mu = 2
        k = 1
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_number_entities_queue(), 0.412698413, 
                        err_msg="should be 0.412698413]")
    
    def test_mm15_mean_number_service(self):
        lda = 1 
        mu = 2
        k = 1
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_number_entities_service(), 0.492063492, 
                        err_msg="should be 0.492063492]")

    def test_mm15_mean_time_system(self):
        lda = 1 
        mu = 2
        k = 1
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_time_system(), 0.919354839, 
                        err_msg="should be 0.919354839]")

    def test_mm15_mean_time_queue(self):
        lda = 1 
        mu = 2
        k = 1
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_time_queue(), 0.419354839, 
                        err_msg="should be 0.419354839]")
    
    def test_mm15_mean_time_service(self):
        lda = 1 
        mu = 2
        k = 1
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_time_service(), 0.5, 
                        err_msg="should be 0.5]")

    # tests for M/M/2/5
    def test_mm25_mean_number_system(self):
        lda = 1 
        mu = 2
        k = 2
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_number_entities(), 0.531066822978, 
                        err_msg="should be 0.531066822978]")

    def test_mm25_mean_number_queue(self):
        lda = 1 
        mu = 2
        k = 2
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_number_entities_queue(), 0.031652989449, 
                        err_msg="should be 0.031652989449]")
    
    def test_mm25_mean_number_service(self):
        lda = 1 
        mu = 2
        k = 2
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_number_entities_service(), 0.499413833529, 
                        err_msg="should be 0.499413833529]")

    def test_mm25_mean_time_system(self):
        lda = 1 
        mu = 2
        k = 2
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_time_system(), 0.531690140845, 
                        err_msg="should be 0.531690140845]")

    def test_mm25_mean_time_queue(self):
        lda = 1 
        mu = 2
        k = 2
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_time_queue(), 0.031690140845, 
                        err_msg="should be 0.031690140845]")
    
    def test_mm25_mean_time_service(self):
        lda = 1 
        mu = 2
        k = 2
        n = 5
        q = mmkn(k,lda,mu,n) 
        assert_almost_equal(q.mean_time_service(), 0.5, 
                        err_msg="should be 0.5]")

if __name__ == '__main__':
    unittest.main()