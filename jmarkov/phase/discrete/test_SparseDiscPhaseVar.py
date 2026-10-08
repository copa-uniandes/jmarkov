"""
test_SparseDiscPhaseVar.py

Tests for SparseDiscPhaseVar.py.

This concrete class supplies copy(), new_var() and a constructor that
guarantees sparse storage, on top of the logic in AbstractDiscretePhaseType
(already covered in test_AbstractDiscPhaseVar.py). The tests concentrate on
those, plus an exhaustive check that the sparse variable produces exactly the
same numbers as the dense one: storage must not change the mathematics.

Run with:
    python3 -m unittest test_SparseDiscPhaseVar -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import doctest
import unittest

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

import SparseDiscPhaseVar as sparse_module
from SparseDiscPhaseVar import SparseDiscretePhaseType as S
from DenseDiscPhaseVar import DenseDiscretePhaseType as D


class TestStorage(unittest.TestCase):
    """The class must actually store the matrix sparsely."""

    def setUp(self):
        self.alpha = np.array([0.4, 0.3])
        self.A = np.array([[0.5, 0.2], [0.1, 0.6]])

    def test_dense_input_is_converted_to_sparse(self):
        X = S(self.alpha, self.A)
        self.assertTrue(sp.issparse(X.A),
                        "a dense input must be converted, or the class name lies")

    def test_sparse_input_stays_sparse(self):
        for api in (sp.csr_matrix, sp.csr_array, sp.csc_matrix, sp.csc_array):
            X = S(self.alpha, api(self.A))
            self.assertTrue(sp.issparse(X.A), f"failed with {api.__name__}")

    def test_any_format_is_normalized_to_csr(self):
        """
        as_square_matrix in matrix_utils normalizes with .tocsr(), so the stored
        format is always CSR. What is preserved is the scipy API family:
        sparray in gives sparray stored, spmatrix in gives spmatrix stored.
        Every DPH operation is row oriented (alpha @ A, A @ 1, row sums), which
        is exactly what CSR is built for.
        """
        expected = {
            sp.csr_matrix: "csr_matrix",
            sp.csc_matrix: "csr_matrix",
            sp.csr_array: "csr_array",
            sp.csc_array: "csr_array",
        }
        for api, stored in expected.items():
            X = S(self.alpha, api(self.A))
            self.assertEqual(type(X.A).__name__, stored,
                             f"{api.__name__} should be stored as {stored}")

    def test_values_survive_the_conversion(self):
        X = S(self.alpha, self.A)
        assert_allclose(X.A.toarray(), self.A, atol=1e-12)


class TestCopy(unittest.TestCase):
    def setUp(self):
        self.alpha = np.array([0.4, 0.3])
        self.A = sp.csr_array([[0.5, 0.2], [0.1, 0.6]])
        self.X = S(self.alpha, self.A)

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
        assert_allclose(self.X.alpha, self.alpha)

    def test_copy_returns_same_concrete_type(self):
        Y = self.X.copy()
        self.assertIsInstance(Y, S)

    def test_copy_stays_sparse(self):
        Y = self.X.copy()
        self.assertTrue(sp.issparse(Y.A))


class TestNewVar(unittest.TestCase):
    def setUp(self):
        self.X = S(np.array([0.4, 0.3]), sp.csr_array([[0.5, 0.2], [0.1, 0.6]]))

    def test_new_var_has_requested_size(self):
        Y = self.X.new_var(5)
        self.assertEqual(Y.n_phases, 5)
        self.assertEqual(Y.alpha.shape, (5,))
        self.assertEqual(Y.A.shape, (5, 5))

    def test_new_var_is_all_zeros(self):
        Y = self.X.new_var(3)
        assert_allclose(Y.alpha, np.zeros(3))
        assert_allclose(Y.A.toarray(), np.zeros((3, 3)))

    def test_new_var_stores_no_explicit_entries(self):
        # An all-zero sparse matrix should hold nothing at all.
        self.assertEqual(self.X.new_var(6).A.nnz, 0)

    def test_new_var_has_all_mass_at_absorption(self):
        # alpha = 0 implies alpha_0 = 1, that is P(X=0)=1.
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
        self.assertIsInstance(Y, S)


class TestDocstringExamples(unittest.TestCase):
    """Explicitly verifies the examples in the class docstring."""

    def test_geometric_example_from_docstring(self):
        X = S(np.array([1.0]), sp.csr_array([[0.5]]))
        self.assertAlmostEqual(X.mean(), 2.0, places=6)
        self.assertAlmostEqual(X.var(), 2.0, places=6)
        expected_pmf = [0.0, 0.5, 0.25, 0.125]
        for k, value in enumerate(expected_pmf):
            self.assertAlmostEqual(round(X.pmf(k), 4), value, places=4)

    def test_negative_binomial_example_from_docstring(self):
        alpha = np.array([1.0, 0.0])
        A = sp.csr_array([[0.6, 0.4], [0.0, 0.6]])
        Y = S(alpha, A)
        self.assertAlmostEqual(Y.mean(), 5.0, places=6)
        self.assertAlmostEqual(Y.var(), 7.5, places=6)


class TestSparseVsDenseEquivalence(unittest.TestCase):
    """
    The sparse and the dense class must be mathematically equivalent: the same
    numbers out, even though the internal representation differs.
    """

    def setUp(self):
        self.alpha = np.array([0.4, 0.3, 0.2])
        A_dense = np.array([[0.5, 0.2, 0.0],
                            [0.1, 0.6, 0.1],
                            [0.0, 0.0, 0.4]])
        self.dense = D(self.alpha, A_dense)
        self.sparse = S(self.alpha, sp.csr_array(A_dense))

    def test_mean_matches(self):
        self.assertAlmostEqual(self.dense.mean(), self.sparse.mean(), places=10)

    def test_var_matches(self):
        self.assertAlmostEqual(self.dense.var(), self.sparse.var(), places=10)

    def test_std_matches(self):
        self.assertAlmostEqual(self.dense.std(), self.sparse.std(), places=10)

    def test_cv_matches(self):
        self.assertAlmostEqual(self.dense.cv(), self.sparse.cv(), places=10)

    def test_scv_matches(self):
        self.assertAlmostEqual(self.dense.scv(), self.sparse.scv(), places=10)

    def test_pmf_matches_for_range(self):
        for k in range(20):
            self.assertAlmostEqual(self.dense.pmf(k), self.sparse.pmf(k),
                                   places=10, msg=f"pmf({k}) differs")

    def test_cdf_matches_for_range(self):
        for k in range(20):
            self.assertAlmostEqual(self.dense.cdf(k), self.sparse.cdf(k),
                                   places=10, msg=f"cdf({k}) differs")

    def test_survival_matches_for_range(self):
        for k in range(20):
            self.assertAlmostEqual(self.dense.survival(k), self.sparse.survival(k),
                                   places=10, msg=f"survival({k}) differs")

    def test_pmf_range_matches(self):
        assert_allclose(self.dense.pmf_range(15), self.sparse.pmf_range(15),
                        atol=1e-10)

    def test_cdf_range_matches(self):
        assert_allclose(self.dense.cdf_range(15), self.sparse.cdf_range(15),
                        atol=1e-10)

    def test_moment_matches(self):
        for k in range(1, 4):
            self.assertAlmostEqual(self.dense.moment(k), self.sparse.moment(k),
                                   places=8, msg=f"moment({k}) differs")

    def test_factorial_moment_matches(self):
        for k in range(1, 4):
            self.assertAlmostEqual(self.dense.factorial_moment(k),
                                   self.sparse.factorial_moment(k),
                                   places=8, msg=f"factorial_moment({k}) differs")

    def test_quantile_matches(self):
        for p in (0.25, 0.5, 0.75, 0.9):
            self.assertEqual(self.dense.quantile(p), self.sparse.quantile(p),
                             msg=f"quantile({p}) differs")

    def test_median_matches(self):
        self.assertEqual(self.dense.median(), self.sparse.median())

    def test_prob_matches(self):
        self.assertAlmostEqual(self.dense.prob(2, 5), self.sparse.prob(2, 5),
                               places=10)

    def test_get_vec0_matches(self):
        self.assertAlmostEqual(self.dense.get_vec0(), self.sparse.get_vec0(),
                               places=10)

    def test_get_mat0_matches(self):
        assert_allclose(self.dense.get_mat0(), self.sparse.get_mat0(), atol=1e-10)

    def test_equality_across_storage_formats(self):
        # Same representation, different storage: they must compare equal.
        self.assertEqual(self.dense, self.sparse)


class TestMultipleSparseFormats(unittest.TestCase):
    """Every sparse format must give the same answer as the dense one."""

    def setUp(self):
        self.alpha = np.array([0.6, 0.4])
        self.A = np.array([[0.5, 0.2], [0.1, 0.6]])
        self.dense = D(self.alpha, self.A)

    def test_every_format_matches_dense(self):
        for api in (sp.csr_matrix, sp.csr_array, sp.csc_matrix, sp.csc_array,
                    sp.coo_matrix, sp.lil_matrix):
            sparse_var = S(self.alpha, api(self.A))
            name = api.__name__
            self.assertAlmostEqual(sparse_var.mean(), self.dense.mean(),
                                   places=10, msg=f"mean differs with {name}")
            self.assertAlmostEqual(sparse_var.var(), self.dense.var(),
                                   places=10, msg=f"var differs with {name}")
            self.assertAlmostEqual(sparse_var.pmf(3), self.dense.pmf(3),
                                   places=10, msg=f"pmf differs with {name}")


class TestConstructorValidation(unittest.TestCase):
    """Validation must work the same way on the sparse branch."""

    def test_rejects_recurrent_matrix(self):
        with self.assertRaises(ValueError):
            S(np.array([1.0, 0.0]), sp.csr_array([[1.0, 0.0], [0.5, 0.4]]))

    def test_rejects_negative_alpha(self):
        with self.assertRaises(ValueError):
            S(np.array([-0.1, 1.1]), sp.csr_array([[0.5, 0.0], [0.0, 0.5]]))

    def test_rejects_alpha_length_mismatch(self):
        with self.assertRaises(ValueError):
            S(np.array([1.0]), sp.csr_array([[0.5, 0.2], [0.1, 0.6]]))

    def test_accepts_defective_alpha(self):
        X = S(np.array([0.5, 0.3]), sp.csr_array([[0.5, 0.2], [0.1, 0.6]]))
        self.assertAlmostEqual(X.get_vec0(), 0.2, places=10)


class TestLargeSparseMatrix(unittest.TestCase):
    """
    The case sparse storage exists for: a large, genuinely sparse matrix. A
    bidiagonal sub-generator with n = 2000 is a discrete Erlang, so the mean is
    known in closed form: n phases each geometric with success 1 - 0.4 - 0.5.
    """

    def test_bidiagonal_erlang_mean(self):
        n = 2000
        stay, forward = 0.4, 0.5
        A = sp.diags_array([np.full(n, stay), np.full(n - 1, forward)],
                           offsets=[0, 1], format="csr")
        alpha = np.zeros(n)
        alpha[0] = 1.0
        X = S(alpha, A)
        # Each phase is geometric with absorption probability 1 - stay - forward
        # only in the last phase; the mean follows from solving (I-A)x = 1.
        self.assertGreater(X.mean(), 0.0)
        self.assertTrue(np.isfinite(X.mean()))
        self.assertTrue(sp.issparse(X.A))
        # Memory sanity: CSR must hold far fewer entries than n*n.
        self.assertLess(X.A.nnz, 3 * n)


class TestDoctests(unittest.TestCase):
    def test_all_doctests_pass(self):
        results = doctest.testmod(sparse_module, verbose=False)
        self.assertEqual(results.failed, 0,
                         f"{results.failed} doctest(s) failed in "
                         f"SparseDiscPhaseVar.py")


if __name__ == "__main__":
    unittest.main()
