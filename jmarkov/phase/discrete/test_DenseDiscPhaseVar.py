"""
test_DenseDiscPhaseVar.py

Tests for DenseDiscPhaseVar.py.

This concrete class only supplies copy() and new_var() on top of the
logic in AbstractDiscretePhaseType (already covered in
test_AbstractDiscPhaseVar.py), so the tests concentrate on those two operations
and on the general contract promised by the class docstring (its own two
doctests: Geometric and Negative Binomial).

Run with:
    python3 -m unittest test_DenseDiscPhaseVar -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import doctest
import unittest

import numpy as np
from numpy.testing import assert_allclose

import DenseDiscPhaseVar as dense_module
from DenseDiscPhaseVar import DenseDiscretePhaseType as D


class TestCopy(unittest.TestCase):
    def setUp(self):
        self.alpha = np.array([0.4, 0.3])
        self.A = np.array([[0.5, 0.2], [0.1, 0.6]])
        self.X = D(self.alpha, self.A)

    def test_copy_is_equal_value(self):
        Y = self.X.copy()
        self.assertEqual(self.X, Y)

    def test_copy_is_independent_object(self):
        Y = self.X.copy()
        self.assertIsNot(self.X, Y)
        self.assertIsNot(self.X.alpha, Y.alpha)
        self.assertIsNot(self.X.A, Y.A)

    def test_mutating_copy_does_not_affect_original(self):
        Y = self.X.copy()
        Y._alpha[0] = 0.0
        Y._A[0, 0] = 0.0
        assert_allclose(self.X.alpha, self.alpha)
        assert_allclose(self.X.A, self.A)

    def test_copy_returns_same_concrete_type(self):
        Y = self.X.copy()
        self.assertIsInstance(Y, D)


class TestNewVar(unittest.TestCase):
    def setUp(self):
        self.X = D(np.array([0.4, 0.3]), np.array([[0.5, 0.2], [0.1, 0.6]]))

    def test_new_var_has_requested_size(self):
        Y = self.X.new_var(5)
        self.assertEqual(Y.n_phases, 5)
        self.assertEqual(Y.alpha.shape, (5,))
        self.assertEqual(Y.A.shape, (5, 5))

    def test_new_var_is_all_zeros(self):
        Y = self.X.new_var(3)
        assert_allclose(Y.alpha, np.zeros(3))
        assert_allclose(Y.A, np.zeros((3, 3)))

    def test_new_var_has_all_mass_at_absorption(self):
        # alpha = 0 implies alpha_0 = 1 - sum(alpha) = 1, that is P(X=0)=1.
        Y = self.X.new_var(3)
        self.assertAlmostEqual(Y.get_vec0(), 1.0, places=10)
        self.assertAlmostEqual(Y.pmf(0), 1.0, places=10)
        self.assertAlmostEqual(Y.mean(), 0.0, places=10)

    def test_new_var_rejects_n_less_than_one(self):
        with self.assertRaises(ValueError):
            self.X.new_var(0)
        with self.assertRaises(ValueError):
            self.X.new_var(-2)

    def test_new_var_returns_same_concrete_type(self):
        Y = self.X.new_var(4)
        self.assertIsInstance(Y, D)


class TestDocstringExamples(unittest.TestCase):
    """
    Explicitly verifies, with assertAlmostEqual, the two examples that appear in
    the class docstring (over and above the doctest runner picking them up in
    test_all_doctests_pass).
    """

    def test_geometric_example_from_docstring(self):
        X = D(np.array([1.0]), np.array([[0.5]]))
        self.assertAlmostEqual(X.mean(), 2.0, places=6)
        self.assertAlmostEqual(X.var(), 2.0, places=6)
        expected_pmf = [0.0, 0.5, 0.25, 0.125]
        for k, value in enumerate(expected_pmf):
            self.assertAlmostEqual(round(X.pmf(k), 4), value, places=4)

    def test_negative_binomial_example_from_docstring(self):
        alpha = np.array([1.0, 0.0])
        A = np.array([[0.6, 0.4], [0.0, 0.6]])
        Y = D(alpha, A)
        self.assertAlmostEqual(Y.mean(), 5.0, places=6)
        self.assertAlmostEqual(Y.var(), 7.5, places=6)


class TestDoctests(unittest.TestCase):
    def test_all_doctests_pass(self):
        results = doctest.testmod(dense_module, verbose=False)
        self.assertEqual(results.failed, 0,
                         f"{results.failed} doctest(s) failed in "
                         f"DenseDiscPhaseVar.py")


if __name__ == "__main__":
    unittest.main()
