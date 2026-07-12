import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.phase.ctph import ctph


class Test_ctph_pdf(unittest.TestCase):
    def test_ctph_pdf(self):
        alpha = np.array([0.5, 0.5])
        T = np.array([[-2, 0], [0, -5]])
        ph = ctph(alpha, T)
        assert_allclose(ph.pdf([0.2],unif=False), 1.590018648964245,err_msg="should be 1.590018648964245")

class Test_ctph_cdf(unittest.TestCase):
    def test_ctph_cdf(self):
        alpha = np.array([0.5, 0.5])
        T = np.array([[-2, 0], [0, -5]])
        ph = ctph(alpha, T)
        assert_allclose(ph.cdf(0.2), 0.480900256396459,err_msg="should be 0.480900256396459")

class Test_pdf_uniformization(unittest.TestCase):
    def test_ctph_cdf(self):
        alpha = np.array([0.5, 0.5])
        T = np.array([[-2, 0], [0, -5]])
        ph = ctph(alpha, T)
        res = ph.pdf([0.2], unif=False)
        print(f"Direct method result: {res}")
        res_unif = ph.pdf([0.2], unif=True)
        print(f"Uniformization method result: {res_unif}")
        assert_allclose(res_unif, res, err_msg="Uniformization and direct methods should give similar results")

class Test_pdf_uniformization_large(unittest.TestCase):
    def test_ctph_cdf(self):
        lda = 1
        n = 10
        alpha = np.zeros(n)
        alpha[0] = 1
        T = np.array([[-2, 0], [0, -5]])
        T = np.zeros((n,n))
        for i in range(n-1):
            T[i,i] = -lda
            T[i,i+1] = lda
        T[n-1,n-1] = -lda
        ph = ctph(alpha, T)
        res = ph.pdf([0.2], unif=False)
        print(f"Direct method result: {res}")
        res_unif = ph.pdf([0.2], unif=True)
        print(f"Uniformization method result: {res_unif}")
        assert_allclose(res_unif, res, err_msg="Uniformization and direct methods should give similar results")
if __name__ == '__main__':
    unittest.main()