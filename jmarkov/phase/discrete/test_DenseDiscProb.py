"""
test_DenseDiscProb.py

Tests for the probability methods of DenseDiscretePhaseType.

The expected PMF, CDF and quantile values are computed independently in exact
rational arithmetic by stepping the absorbing chain
(``generate_reference_values.py``).

Run with:
    python3 -m unittest test_DenseDiscProb -v

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
from reference_values import PROB

EXACT_TOL = 1e-12
N_MAX = 3000


class DenseDiscProbTest(unittest.TestCase):
    """Tests for the discrete probability functions."""

    @staticmethod
    def make(vector, matrix):
        return DenseDiscretePhaseType(vector, matrix)

    def setUp(self):
        self.vector1, self.matrix1 = fx.fixture("prob1")
        self.vector2, self.matrix2 = fx.fixture("prob2")
        self.vector3, self.matrix3 = fx.fixture("prob3")
        self.var1 = self.make(self.vector1, self.matrix1)
        self.var2 = self.make(self.vector2, self.matrix2)
        self.var3 = self.make(self.vector3, self.matrix3)

    def assert_unchanged(self, var, vector, matrix):
        self.assertLess(np.max(np.abs(to_dense(var.A) - matrix)), 1.0e-12,
                        "Matrix changed")
        self.assertLess(np.max(np.abs(var.alpha - vector)), 1.0e-12,
                        "Vector changed")

    def check_prob(self, var, name):
        ref = PROB[name]
        for k, (real_pmf, real_cdf) in enumerate(zip(ref["pmf"], ref["cdf"])):
            self.assertAlmostEqual(var.pmf(k), real_pmf, delta=EXACT_TOL,
                                   msg=f"Probability Mass Function non equal (k={k})")
            self.assertAlmostEqual(var.cdf(k), real_cdf, delta=EXACT_TOL,
                                   msg=f"Cumulative Probability Function non equal (k={k})")
            self.assertAlmostEqual(var.survival(k), 1 - real_cdf, delta=EXACT_TOL,
                                   msg=f"Survival Function non equal (k={k})")
        k_max = len(ref["pmf"]) - 1
        assert_allclose(var.pmf_range(k_max), ref["pmf"], atol=EXACT_TOL,
                        err_msg="pmf_range non equal")
        assert_allclose(var.cdf_range(k_max), ref["cdf"], atol=EXACT_TOL,
                        err_msg="cdf_range non equal")

    def check_quantile(self, var, name):
        for p, k in PROB[name]["quantile"].items():
            self.assertEqual(var.quantile(p), k, f"Quantile non equal (p={p})")

    def test_prob1(self):
        self.check_prob(self.var1, "prob1")
        self.assert_unchanged(self.var1, self.vector1, self.matrix1)

    def test_prob2(self):
        self.check_prob(self.var2, "prob2")
        self.assert_unchanged(self.var2, self.vector2, self.matrix2)

    def test_prob3(self):
        self.check_prob(self.var3, "prob3")
        self.assert_unchanged(self.var3, self.vector3, self.matrix3)

    def test_quantile1(self):
        self.check_quantile(self.var1, "prob1")
        self.assertEqual(self.var1.median(), PROB["prob1"]["quantile"][0.5])
        self.assert_unchanged(self.var1, self.vector1, self.matrix1)

    def test_quantile2(self):
        self.check_quantile(self.var2, "prob2")
        self.assert_unchanged(self.var2, self.vector2, self.matrix2)

    def test_quantile3(self):
        self.check_quantile(self.var3, "prob3")
        self.assert_unchanged(self.var3, self.vector3, self.matrix3)

    def test_pmf_sums_to_one(self):
        for var in (self.var1, self.var2, self.var3):
            self.assertAlmostEqual(var.pmf_range(N_MAX).sum(), 1.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
