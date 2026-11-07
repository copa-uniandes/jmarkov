"""
test_AbstractDiscPhaseVar.py

Tests for AbstractDiscPhaseVar.py, using DenseDiscretePhaseType as the concrete
implementation (the abstract class cannot be instantiated on its own).

Where this file fits
--------------------
This file covers closed-form distributions,
internal consistency between methods, constructor validation, equality and
hashing. It rests on:

  1. Cases with a known closed-form solution in the Phase-Type literature
     (Neuts 1981; Latouche & Ramaswami 1999):
       - Geometric(p)          as a 1-phase DPH.
       - Negative Binomial(r,p) as r geometric phases in series.
  2. Brute force: summing pmf(k) over many k and comparing against 1, or
     building the cdf by accumulating the pmf term by term and comparing it
     against cdf() directly.
    3. Direct-search references for discrete quantiles.

Run with:
    python3 -m unittest test_AbstractDiscPhaseVar -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest
from math import comb

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from DenseDiscPhaseVar import DenseDiscretePhaseType as D


# =============================================================================
# Fixtures: distributions with a known closed form
# =============================================================================

def geometric(p: float) -> D:
    """
    Geometric on {1, 2, 3, ...} with success parameter p.

    DPH(alpha=[1], A=[[1-p]]). E[X] = 1/p, Var(X) = (1-p)/p^2.
    """
    return D(np.array([1.0]), np.array([[1.0 - p]]))


def negative_binomial(r: int, p: float) -> D:
    """
    Negative Binomial(r, p): r geometric(p) phases in series (discrete Erlang).

    E[X] = r/p, Var(X) = r(1-p)/p^2.
    """
    q = 1.0 - p
    A = np.diag([q] * r) + np.diag([p] * (r - 1), k=1)
    alpha = np.zeros(r)
    alpha[0] = 1.0
    return D(alpha, A)


class TestGeometricAnalytic(unittest.TestCase):
    """Geometric(p=0.5): the simplest case, with a closed form for everything."""

    def setUp(self):
        self.p = 0.5
        self.X = geometric(self.p)

    def test_mean(self):
        self.assertAlmostEqual(self.X.mean(), 1 / self.p, places=10)

    def test_var(self):
        expected = (1 - self.p) / self.p ** 2
        self.assertAlmostEqual(self.X.var(), expected, places=10)

    def test_pmf_matches_closed_form(self):
        for k in range(1, 10):
            expected = (1 - self.p) ** (k - 1) * self.p
            self.assertAlmostEqual(self.X.pmf(k), expected, places=10)

    def test_pmf_at_zero_is_zero(self):
        # Classic geometric: support on {1, 2, 3, ...}, so P(X=0) = 0.
        self.assertAlmostEqual(self.X.pmf(0), 0.0, places=10)

    def test_cdf_matches_closed_form(self):
        for k in range(1, 10):
            expected = 1 - (1 - self.p) ** k
            self.assertAlmostEqual(self.X.cdf(k), expected, places=10)

    def test_survival_is_1_minus_cdf(self):
        for k in range(1, 10):
            self.assertAlmostEqual(self.X.survival(k), 1 - self.X.cdf(k),
                                   places=10)

    def test_pmf_sums_to_one(self):
        total = sum(self.X.pmf(k) for k in range(200))
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_quantile_matches_direct_search(self):
        """
        Independent reference for the quantile: search directly for the
        smallest k with cdf(k) >= p, without going through the internal
        accumulation method.
        """
        for p_order in (0.5, 0.75, 0.9, 0.95, 0.99):
            computed = self.X.quantile(p_order)
            reference = next(k for k in range(1, 1000)
                             if self.X.cdf(k) >= p_order - 1e-12)
            self.assertEqual(computed, reference,
                             f"quantile({p_order}) disagrees with direct search")

    def test_median_equals_quantile_one_half(self):
        self.assertEqual(self.X.median(), self.X.quantile(0.5))

class TestNegativeBinomialAnalytic(unittest.TestCase):
    """Negative Binomial(r=2, p=0.4): two phases in series."""

    def setUp(self):
        self.r, self.p = 2, 0.4
        self.X = negative_binomial(self.r, self.p)

    def test_mean(self):
        self.assertAlmostEqual(self.X.mean(), self.r / self.p, places=8)

    def test_var(self):
        expected = self.r * (1 - self.p) / self.p ** 2
        self.assertAlmostEqual(self.X.var(), expected, places=8)

    def test_pmf_matches_negative_binomial_formula(self):
        q = 1 - self.p
        for k in range(self.r, self.r + 10):
            expected = comb(k - 1, self.r - 1) * (self.p ** self.r) * (q ** (k - self.r))
            self.assertAlmostEqual(self.X.pmf(k), expected, places=8)

    def test_pmf_zero_below_support(self):
        # The support starts at k=r (at least two phases to reach absorption).
        self.assertAlmostEqual(self.X.pmf(0), 0.0, places=10)
        self.assertAlmostEqual(self.X.pmf(1), 0.0, places=10)

    def test_pmf_sums_to_one(self):
        total = sum(self.X.pmf(k) for k in range(500))
        self.assertAlmostEqual(total, 1.0, places=6)


class TestPmfCdfConsistency(unittest.TestCase):
    """The hand-accumulated cdf must match cdf() for any DPH."""

    def setUp(self):
        alpha = np.array([0.4, 0.3, 0.3])
        A = np.array([[0.5, 0.2, 0.0],
                      [0.1, 0.6, 0.1],
                      [0.0, 0.0, 0.4]])
        self.X = D(alpha, A)

    def test_cdf_equals_cumulative_pmf(self):
        running = 0.0
        for k in range(50):
            running += self.X.pmf(k)
            self.assertAlmostEqual(self.X.cdf(k), running, places=8)

    def test_pmf_range_matches_individual_calls(self):
        block = self.X.pmf_range(20)
        one_by_one = np.array([self.X.pmf(k) for k in range(21)])
        assert_allclose(block, one_by_one, atol=1e-10)

    def test_cdf_range_matches_individual_calls(self):
        block = self.X.cdf_range(20)
        one_by_one = np.array([self.X.cdf(k) for k in range(21)])
        assert_allclose(block, one_by_one, atol=1e-10)

    def test_prob_between_a_and_b(self):
        # Convention of this implementation (see the prob() docstring):
        # P(a < X <= b) = F(b) - F(a), not P(a <= X <= b).
        a, b = 2, 5
        expected = self.X.cdf(b) - self.X.cdf(a)
        self.assertAlmostEqual(self.X.prob(a, b), expected, places=10)

    def test_prob_returns_zero_if_b_not_greater_than_a(self):
        self.assertEqual(self.X.prob(5, 5), 0.0)
        self.assertEqual(self.X.prob(5, 2), 0.0)

    def test_moment_1_equals_mean(self):
        self.assertAlmostEqual(self.X.moment(1), self.X.mean(), places=8)

    def test_std_squared_equals_var(self):
        self.assertAlmostEqual(self.X.std() ** 2, self.X.var(), places=8)

    def test_scv_equals_var_over_mean_squared(self):
        expected = self.X.var() / self.X.mean() ** 2
        self.assertAlmostEqual(self.X.scv(), expected, places=10)

    def test_cv_is_sqrt_scv(self):
        self.assertAlmostEqual(self.X.cv(), self.X.scv() ** 0.5, places=10)


class TestVec0Mat0Consistency(unittest.TestCase):
    """get_vec0/get_mat0 must be consistent with pmf(0) and with the cdf."""

    def setUp(self):
        alpha = np.array([0.4, 0.3])
        A = np.array([[0.5, 0.2], [0.1, 0.6]])
        self.X = D(alpha, A)

    def test_get_vec0_equals_pmf_zero(self):
        self.assertAlmostEqual(self.X.get_vec0(), self.X.pmf(0), places=10)

    def test_get_mat0_is_absorption_vector(self):
        A = self.X.A
        expected = np.ones(2) - A @ np.ones(2)
        assert_allclose(self.X.get_mat0(), expected, atol=1e-10)


class TestConstructorValidation(unittest.TestCase):
    def test_rejects_non_square_matrix(self):
        with self.assertRaises(ValueError):
            D(np.array([1.0, 0.0]), np.zeros((2, 3)))

    def test_rejects_recurrent_matrix(self):
        # A row summing to exactly 1: absorption is impossible.
        with self.assertRaises(ValueError):
            D(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.5, 0.4]]))

    def test_rejects_negative_alpha(self):
        with self.assertRaises(ValueError):
            D(np.array([-0.1, 1.1]), np.array([[0.5, 0.0], [0.0, 0.5]]))

    def test_accepts_defective_alpha(self):
        # alpha_0 > 0 (mass on immediate absorption) is valid: P(X=0) > 0.
        X = D(np.array([0.5, 0.3]), np.array([[0.5, 0.2], [0.1, 0.6]]))
        self.assertGreater(X.get_vec0(), 0.0)

    def test_constructor_copies_its_inputs(self):
        # Java copies (new DenseMatrix(matrix)); modifying the caller's arrays
        # afterwards must not change the variable.
        alpha = np.array([1.0, 0.0])
        A = np.array([[0.5, 0.2], [0.0, 0.5]])
        X = D(alpha, A)
        mean = X.mean()
        alpha[0], A[0, 0] = 0.1, 0.9
        self.assertEqual(X.mean(), mean)

        A_sparse = sp.csr_array(np.array([[0.5, 0.2], [0.0, 0.5]]))
        Y = D(np.array([1.0, 0.0]), A_sparse)
        mean = Y.mean()
        A_sparse.data[0] = 0.9
        self.assertEqual(Y.mean(), mean)

    def test_rejects_recurrent_sparse_matrix_with_value_error(self):
        # The error message used to call np.linalg.eigvals on the sparse
        # matrix and crash with LinAlgError instead.
        with self.assertRaises(ValueError):
            D(np.array([1.0, 0.0]), sp.csr_array(np.array([[1.0, 0.0], [0.5, 0.4]])))


class TestEqualityAndHash(unittest.TestCase):
    def test_copy_is_equal_but_independent(self):
        X = D(np.array([0.4, 0.3]), np.array([[0.5, 0.2], [0.1, 0.6]]))
        Y = X.copy()
        self.assertEqual(X, Y)
        Y._alpha[0] = 0.0
        self.assertNotEqual(X._alpha[0], Y._alpha[0])

    def test_equal_objects_have_equal_hash(self):
        X = D(np.array([0.4, 0.3]), np.array([[0.5, 0.2], [0.1, 0.6]]))
        Y = X.copy()
        self.assertEqual(hash(X), hash(Y))

    def test_dense_and_sparse_equal_if_same_values(self):
        alpha = np.array([0.4, 0.3])
        A = np.array([[0.5, 0.2], [0.1, 0.6]])
        X_dense = D(alpha, A)
        X_sparse = D(alpha, sp.csr_array(A))
        self.assertEqual(X_dense, X_sparse)


class TestDenseVsSparseEndToEnd(unittest.TestCase):
    """The whole class, end to end, with a sparse matrix."""

    def setUp(self):
        alpha = np.array([0.4, 0.3])
        A = np.array([[0.5, 0.2], [0.1, 0.6]])
        self.dense = D(alpha, A)

    def test_all_public_methods_match(self):
        for api in (sp.csr_matrix, sp.csr_array):
            sparse_var = D(self.dense.alpha.copy(), api(self.dense.A))
            for method, arg in [
                ("mean", None), ("var", None), ("std", None), ("cv", None),
                ("scv", None), ("get_vec0", None),
                ("moment", 3), ("factorial_moment", 2),
                ("pmf", 4), ("cdf", 4), ("survival", 4),
            ]:
                a = getattr(self.dense, method)() if arg is None \
                    else getattr(self.dense, method)(arg)
                b = getattr(sparse_var, method)() if arg is None \
                    else getattr(sparse_var, method)(arg)
                self.assertAlmostEqual(float(np.asarray(a)), float(np.asarray(b)),
                                       places=10,
                                       msg=f"{method} differs with {api.__name__}")

    def test_get_mat0_matches(self):
        for api in (sp.csr_matrix, sp.csr_array):
            sparse_var = D(self.dense.alpha.copy(), api(self.dense.A))
            assert_allclose(self.dense.get_mat0(), sparse_var.get_mat0(), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
