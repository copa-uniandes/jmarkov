"""
test_DenseDiscClosure.py

Tests for the closure methods of DenseDiscretePhaseType: sum, min, max, mix,
sum_geom and sum_ph.

Two layers of checks:

1. DenseDiscClosureTest - the (alpha, A) produced by each operation must
    equal the hand-built representation (reference_values.CLOSURE, exact
    arithmetic) and the operands must not change.
2. DenseDiscClosureDistributionTest - the resulting DISTRIBUTION must
   satisfy identities that hold for any independent random variables,
   phase-type or not: convolution for the sum, 1-(1-F)(1-G) and F*G for min
   and max, the p-weighted pmf for the mixture, and brute-force compound sums
   for sum_geom and sum_ph. Several fixtures have alpha_0 > 0 (mass at 0),
   which is where the geometric and PH-counted sums are delicate.

Run with:
    python3 -m unittest test_DenseDiscClosure -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

import discrete_fixtures as fx
from DenseDiscPhaseVar import DenseDiscretePhaseType
from matrix_utils import to_dense
from reference_values import CLOSURE

TOL = 1e-12
K = 60


def compound_pmf(pmf_x, pmf_n, k_max=K):
    """pmf of X_1 + ... + X_N (X_i iid with pmf_x, N with pmf_n), by brute force."""
    out = np.zeros(k_max + 1)
    conv_n = np.zeros(k_max + 1)
    conv_n[0] = 1.0                       # X^{*0}: point mass at 0
    for w in pmf_n:
        out += w * conv_n
        conv_n = np.convolve(conv_n, pmf_x)[:k_max + 1]
    return out


class ClosureFixtures(unittest.TestCase):
    """setUp shared by the three test classes below."""

    @staticmethod
    def make(vector, matrix):
        return DenseDiscretePhaseType(vector, matrix)

    def setUp(self):
        self.vector1, self.matrix1 = fx.fixture("closure1")
        self.vector2, self.matrix2 = fx.fixture("closure2")
        self.vectorD, self.matrixD = fx.fixture("closureD")
        self.var1 = self.make(self.vector1, self.matrix1)
        self.var2 = self.make(self.vector2, self.matrix2)
        self.varD = self.make(self.vectorD, self.matrixD)

    def tearDown(self):
        for var, vector, matrix in ((self.var1, self.vector1, self.matrix1),
                                    (self.var2, self.vector2, self.matrix2),
                                    (self.varD, self.vectorD, self.matrixD)):
            self.assertLess(np.max(np.abs(to_dense(var.A) - matrix)), 1.0e-12,
                            "Matrix changed")
            self.assertLess(np.max(np.abs(var.alpha - vector)), 1.0e-12,
                            "Vector changed")


class DenseDiscClosureTest(ClosureFixtures):
    """Check the representations produced by the closure operations."""

    def check(self, calc, name, label):
        real = CLOSURE[name]
        assert_allclose(to_dense(calc.A), real["A"], atol=TOL, rtol=0,
                        err_msg=f"{label} non equal (Matrix)")
        assert_allclose(calc.alpha, real["alpha"], atol=TOL, rtol=0,
                        err_msg=f"{label} non equal (Vector)")
        self.assertIsInstance(calc, type(self.var1))

    def test_sum(self):
        self.check(self.var1.sum(self.var2), "sum", "Sum of Variables")

    def test_min(self):
        self.check(self.var1.min(self.var2), "min", "Min of Variables")

    def test_max(self):
        self.check(self.var1.max(self.var2), "max", "Max of Variables")

    def test_mix(self):
        self.check(self.var1.mix(fx.MIX_P, self.var2), "mix", "Mix of Variables")

    def test_sum_geom(self):
        self.check(self.var2.sum_geom(fx.SUM_GEOM_P), "sum_geom",
                   "Geometric Sum of Variables")

    def test_sum_ph(self):
        self.check(self.var2.sum_ph(self.varD), "sum_ph",
                   "Sum of a discrete phase number")


class DenseDiscClosureDistributionTest(ClosureFixtures):
    """The resulting distributions satisfy the probability identities."""

    def test_sum_is_convolution(self):
        for X, Y in ((self.var1, self.var2), (self.var2, self.varD),
                     (self.varD, self.var1)):
            conv = np.convolve(X.pmf_range(K), Y.pmf_range(K))[:K + 1]
            assert_allclose(X.sum(Y).pmf_range(K), conv, atol=TOL)
            self.assertAlmostEqual(X.sum(Y).mean(), X.mean() + Y.mean(), delta=1e-10)

    def test_min_and_max_cdf_identities(self):
        for X, Y in ((self.var1, self.var2), (self.var2, self.varD)):
            F, G = X.cdf_range(K), Y.cdf_range(K)
            assert_allclose(X.min(Y).cdf_range(K), 1 - (1 - F) * (1 - G), atol=TOL)
            assert_allclose(X.max(Y).cdf_range(K), F * G, atol=TOL)

    def test_mix_is_weighted_pmf(self):
        p = fx.MIX_P
        mixed = self.var1.mix(p, self.var2)
        assert_allclose(mixed.pmf_range(K),
                        p * self.var1.pmf_range(K) + (1 - p) * self.var2.pmf_range(K),
                        atol=TOL)

    def test_sum_geom_is_compound_geometric_sum(self):
        p = fx.SUM_GEOM_P
        pmf_n = [0.0] + [p * (1 - p) ** (n - 1) for n in range(1, 400)]
        for X in (self.var1, self.var2, self.varD):
            assert_allclose(X.sum_geom(p).pmf_range(K),
                            compound_pmf(X.pmf_range(K), pmf_n), atol=1e-10)
            # Wald's identity: E[X_1 + ... + X_N] = E[N] E[X] = E[X] / p.
            self.assertAlmostEqual(X.sum_geom(p).mean(), X.mean() / p, delta=1e-9)

    def test_sum_ph_is_compound_sum(self):
        pmf_n = self.varD.pmf_range(400)
        for X in (self.var1, self.var2):
            assert_allclose(X.sum_ph(self.varD).pmf_range(K),
                            compound_pmf(X.pmf_range(K), pmf_n), atol=1e-10)
            self.assertAlmostEqual(X.sum_ph(self.varD).mean(),
                                   X.mean() * self.varD.mean(), delta=1e-9)

    def test_sum_ph_with_geometric_counter_is_sum_geom(self):
        p = fx.SUM_GEOM_P
        counter = self.make([1.0], [[1 - p]])
        assert_allclose(self.var2.sum_ph(counter).pmf_range(K),
                        self.var2.sum_geom(p).pmf_range(K), atol=TOL)

    def test_sum_ph_with_counter_one_is_identity(self):
        one = self.make([1.0], [[0.0]])
        assert_allclose(self.var2.sum_ph(one).pmf_range(K),
                        self.var2.pmf_range(K), atol=TOL)

    def test_phase_counts(self):
        n1, n2, nD = self.var1.n_phases, self.var2.n_phases, self.varD.n_phases
        self.assertEqual(self.var1.sum(self.var2).n_phases, n1 + n2)
        self.assertEqual(self.var1.mix(0.5, self.var2).n_phases, n1 + n2)
        self.assertEqual(self.var1.min(self.var2).n_phases, n1 * n2)
        self.assertEqual(self.var1.max(self.var2).n_phases, n1 * n2 + n1 + n2)
        self.assertEqual(self.var2.sum_geom(0.5).n_phases, n2)
        self.assertEqual(self.var2.sum_ph(self.varD).n_phases, n2 * nD)

    def test_sum_with_identically_zero_variable(self):
        zero = self.make([0.0, 0.0], self.matrix1)
        self.assertIs(zero.sum(self.var2), self.var2)
        self.assertIs(self.var2.sum(zero), self.var2)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            self.var1.mix(1.5, self.var2)
        with self.assertRaises(ValueError):
            self.var1.sum_geom(0.0)


if __name__ == "__main__":
    unittest.main()
