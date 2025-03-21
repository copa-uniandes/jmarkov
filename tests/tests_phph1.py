import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.phph1 import phph1
from jmarkov.phase.ctph import ctph


class TestSteadyState(unittest.TestCase):
    def test_phph1_queue_length_dist(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST) 
        assert_allclose(q.number_entities_dist()[0:24], 
                         [0.581578947368421, 0.260580854966927,     0.0985624593192181,     0.0370701967531774,
                          0.0138991231047589, 0.00520237612537792,   0.00194536082714717,   0.000727054345714727, 
                          0.000271646647013180, 0.000101477485516757, 3.79048496301054e-05, 1.41578521476521e-05,
                          5.28795084061249e-06, 1.97501515154022e-06, 7.37648624451807e-07,	2.75503077188071e-07,
                          1.02896866379604e-07, 3.84305996751784e-08, 1.43533009965034e-08, 5.36075809927903e-09,
                          2.00216797059640e-09, 7.47781544129671e-10, 2.79285853219323e-10, 1.04309319610366e-10], 
                        err_msg="""should be [0.581578947368421, 0.260580854966927,     0.0985624593192181,     0.0370701967531774,
                          0.0138991231047589, 0.00520237612537792,   0.00194536082714717,   0.000727054345714727, 
                          0.000271646647013180, 0.000101477485516757, 3.79048496301054e-05, 1.41578521476521e-05,
                          5.28795084061249e-06, 1.97501515154022e-06, 7.37648624451807e-07,	2.75503077188071e-07,
                          1.02896866379604e-07, 3.84305996751784e-08, 1.43533009965034e-08, 5.36075809927903e-09,
                          2.00216797059640e-09, 7.47781544129671e-10, 2.79285853219323e-10, 1.04309319610366e-10]""")
    
    def test_phph1_mean_number_system(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST) 
        assert_almost_equal(q.mean_number_entities(), 0.671013852117399, 
                        err_msg="should be 0.671013852117399")
        

    def test_phph1_mean_number_queue(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST) 
        assert_almost_equal(q.mean_number_entities_queue(), 0.252592799548006, 
                        err_msg="should be 0.252592799548006")
    
    def test_phph1_mean_number_service(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST) 
        assert_almost_equal(q.mean_number_entities_service(), 0.418421052631579, 
                        err_msg="should be 0.418421052631579")

    def test_phph1_mean_time_system(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST) 
        assert_almost_equal(q.mean_time_system(), 0.424975439674353, 
                        err_msg="should be 0.424975439674353")

    def test_phph1_mean_time_queue(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST) 
        assert_almost_equal(q.mean_time_queue(),0.159975439713737, 
                        err_msg="should be 0.159975439713737")
    
    def test_phph1_mean_time_service(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST) 
        assert_almost_equal(q.mean_time_service(), 0.265, 
                        err_msg="should be 0.265]")
    
    def test_phph1_wait_time_dist(self):
        alpha = np.array([0.9, 0.1])
        T = np.array([[-2, 1], [0, -3]])
        IAT = ctph(alpha, T)
        beta = np.array([0.3, 0.7])
        S = np.array([[-5, 2], [0, -4]])
        ST = ctph(beta, S)
        q = phph1(IAT,ST)
        WT = q.wait_time_dist() 
        assert_allclose(WT.alpha, 
                         [0.087261802731903, 0.292692473447050], 
                        err_msg="should be [0.087261802731903, 0.292692473447050]")
        assert_allclose(WT.T, 
                         [[-4.563690986340487, 1.463462367235248],
                          [0.883332984938204, -3.000562285790562]], 
                        err_msg="""should be [[-4.563690986340487, 1.463462367235248],
                                             [0.883332984938204, -3.000562285790562]]""")
    

if __name__ == '__main__':
    unittest.main()