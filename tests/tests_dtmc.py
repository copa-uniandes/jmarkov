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

class TestPeriod(unittest.TestCase):
    def test_period_1(self):
        P = np.array([[0.1, 0.4, 0.5, 0.0, 0.0, 0.0],
                      [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.1, 0.4, 0.5, 0.0],
                      [0.0, 0.0, 0.0, 0.2, 0.3, 0.5],
                      [0.5, 0.0, 0.0, 0.0, 0.2, 0.3],
                      [0.3, 0.5, 0.0, 0.0, 0.0, 0.2]])
        mc = dtmc(P)
        self.assertEqual(mc.period(), 1)
        
    def test_period_2(self):
        P = np.array([[0, 0, 0.8, 0.2],
                      [0, 0, 0.4, 0.6],
                      [0.7, 0.3, 0, 0],
                      [0.2, 0.8, 0, 0]])
        mc = dtmc(P)
        self.assertEqual(mc.period(), 2)
    
    def test_period_3(self):
        P = np.array([[0, 0, 0.5, 0.25, 0.25, 0, 0],
                      [0, 0, 1/3, 0, 2/3, 0, 0],
                      [0, 0, 0, 0, 0, 1/3, 2/3],
                      [0, 0, 0, 0, 0, 1/2, 1/2],
                      [0, 0, 0, 0, 0, 3/4, 1/4],
                      [0.5, 0.5, 0, 0, 0, 0, 0],
                      [0.25, 0.75, 0, 0, 0, 0, 0]])
        mc = dtmc(P)
        self.assertEqual(mc.period(), 3)
    
    def test_period_4(self):
        P = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0]])
        mc = dtmc(P)
        self.assertEqual(mc.period(), 4)

class TestIsIrreducible(unittest.TestCase):
    def test_dtmc_isirreducible(self):
        P = np.array([[0, 1/3, 2/3],
                      [0,0,1],
                      [1,0,0]])
        mc = dtmc(P)
        self.assertTrue(mc.is_irreducible())

class TestIsErgodic(unittest.TestCase):
    def test_dtmc_isergodic_true(self):
        P = np.array([[0, 1/3, 2/3],
                      [0,0,1],
                      [1,0,0]])
        mc = dtmc(P)
        self.assertTrue(mc.is_ergodic())

    def test_dtmc_isergodic_false(self):
        P = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0]])
        mc = dtmc(P)
        self.assertFalse(mc.is_ergodic())

class TestFirstPassageTime(unittest.TestCase):
    def test_dtmc_first_passage_time(self):
        P = np.array([[0.8,0.2,0],[0,0.2,0.8],[0.8,0.2,0]])
        mc = dtmc(P)
        assert_allclose(mc.first_passage_time(0), [[2.8125], [1.5625]], err_msg="should be [[2.8125], [1.5625]]")

class TestAbsorbtionTimes(unittest.TestCase):
    def test_absorbtion_times(self):
        P = np.array([[0.2, 0.3, 0.1, 0.4], [0.1, 0.2, 0.5, 0.2], [0,0,1,0], [0,0,0,1]])
        states = np.array([0,1,2])
        mc = dtmc(P, states)
        self.assertAlmostEqual(mc.absorbtion_times(target=0,start=0),1.31147540983607)

class TestAbsorbtionProbabilities(unittest.TestCase):
    def test_absortion_probabilities(self):
        P = np.array([[0.2, 0.3, 0.1, 0.4], [0.1, 0.2, 0.5, 0.2], [0,0,1,0], [0,0,0,1]])
        states = np.array([0,1,2])
        mc = dtmc(P, states)
        self.assertAlmostEqual(mc.absorbtion_probabilities(target=3,start=0)[0][0], 0.622950819672131)


if __name__ == '__main__':
    unittest.main()
