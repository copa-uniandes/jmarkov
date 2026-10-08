"""
test_matrix_utils.py

Tests for matrix_utils.py.

Coverage:
    1. Mathematical correctness of the matrix utilities and their edge cases.
    2. Numerical equivalence between dense input (np.ndarray) and sparse input
     (scipy.sparse), across both scipy APIs (csr_matrix and csr_array).
    3. Edge cases and validations that must fail.

Run with:
    python3 -m unittest test_matrix_utils -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import time
import unittest

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

import matrix_utils as mu


# =============================================================================
# Shared fixtures
# =============================================================================

# 3x3 sub-generator matrix (continuous case), with strict absorption.
G3 = np.array([[-2.0, 2.0, 0.0],
               [0.0, -3.0, 1.0],
               [0.0, 0.0, -1.0]])

# 3x3 sub-stochastic matrix (discrete case), with absorption.
D3 = np.array([[0.5, 0.2, 0.0],
               [0.1, 0.6, 0.1],
               [0.0, 0.0, 0.4]])

V3 = np.array([0.5, 0.3, 0.2])
W3 = np.array([1.0, 2.0, 3.0])


def _is_matrix_like(x):
    return mu.is_sparse(x) or (hasattr(x, "ndim") and x.ndim == 2)


class MatrixUtilsTestCase(unittest.TestCase):
    """Base class with a helper to compare dense against both scipy APIs."""

    def assert_dense_sparse_equal(self, fn, *args, tol=1e-11):
        """
        Call fn with the dense args, then again converting every matrix
        argument (ndim == 2) to csr_matrix and to csr_array. All three outputs
        must agree once densified.
        """
        dense = mu.to_dense(fn(*args)) if _is_matrix_like(fn(*args)) \
            else np.asarray(fn(*args), dtype=float)

        for api in (sp.csr_matrix, sp.csr_array):
            sparse_args = [api(a) if (np.ndim(a) == 2) else a for a in args]
            result = fn(*sparse_args)
            result = mu.to_dense(result) if mu.is_sparse(result) \
                else np.asarray(result, dtype=float)
            assert_allclose(result, dense, atol=tol,
                            err_msg=f"{fn.__name__} with {api.__name__}")


# =============================================================================
# Part 1: matrix utility functions
# =============================================================================

class TestOnesHelpers(unittest.TestCase):
    def test_ones_vector(self):
        assert_allclose(mu.ones_vector(3), [1.0, 1.0, 1.0])

    def test_ones_row_col(self):
        self.assertEqual(mu.ones_row(3).shape, (1, 3))
        self.assertEqual(mu.ones_col(3).shape, (3, 1))


class TestKronecker(MatrixUtilsTestCase):
    def test_kronecker_matches_numpy(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        assert_allclose(mu.kronecker(A, B), np.kron(A, B))

    def test_kronecker_mx_col_vector_matches_numpy(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        assert_allclose(mu.kronecker_mx_col_vector(A, b), np.kron(A, b[:, None]))

    def test_kronecker_col_vector_mx_matches_numpy(self):
        a = np.array([1.0, 2.0])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        assert_allclose(mu.kronecker_col_vector_mx(a, B), np.kron(a[:, None], B))

    def test_kronecker_sum_shape_and_formula(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        ks = mu.kronecker_sum(A, B)
        expected = np.kron(A, np.eye(2)) + np.kron(np.eye(2), B)
        self.assertEqual(ks.shape, (4, 4))
        assert_allclose(ks, expected)

    def test_kronecker_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        assert_allclose(mu.kronecker_vectors(a, b), np.kron(a, b))

    def test_dense_vs_sparse(self):
        self.assert_dense_sparse_equal(mu.kronecker, D3, D3)
        self.assert_dense_sparse_equal(mu.kronecker_sum, D3, D3)
        self.assert_dense_sparse_equal(mu.kronecker_mx_col_vector, D3, W3)
        self.assert_dense_sparse_equal(mu.kronecker_col_vector_mx, W3, D3)
        self.assert_dense_sparse_equal(mu.kronecker_mx_row_vector, D3, W3)


class TestConcat(MatrixUtilsTestCase):
    def test_concat_rows_cols(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        self.assertEqual(mu.concat_rows(A, B).shape, (4, 2))
        self.assertEqual(mu.concat_cols(A, B).shape, (2, 4))

    def test_concat_rows_cols_incompatible_raises(self):
        A = np.zeros((2, 2))
        B = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            mu.concat_rows(A, B)
        with self.assertRaises(ValueError):
            mu.concat_cols(np.zeros((2, 2)), np.zeros((3, 2)))

    def test_concat_quad_matches_block_formula(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 5.0], [6.0, 7.0]])
        result = mu.concat_quad(A, B, B, A)
        assert_allclose(result, np.block([[A, B], [B, A]]))

    def test_concat_quad_blocks_placed_correctly(self):
        q = mu.concat_quad(D3, D3 * 0, D3 * 0, D3)
        self.assertEqual(q.shape, (6, 6))
        assert_allclose(q[:3, :3], D3)
        assert_allclose(q[3:, 3:], D3)
        assert_allclose(q[:3, 3:], np.zeros((3, 3)))

    def test_concat_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        assert_allclose(mu.concat_vectors(a, b), [1, 2, 3, 4, 5])

    def test_dense_vs_sparse(self):
        self.assert_dense_sparse_equal(mu.concat_rows, D3, D3)
        self.assert_dense_sparse_equal(mu.concat_cols, D3, D3)
        self.assert_dense_sparse_equal(mu.concat_quad, D3, D3, D3, D3)


class TestMultVector(unittest.TestCase):
    def test_mult_vector_is_outer_product(self):
        a = np.array([1.0, 2.0])
        assert_allclose(mu.mult_vector(a, a), np.outer(a, a))


class TestMatPower(MatrixUtilsTestCase):
    def test_mat_power_matches_matrix_power(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        for k in (0, 1, 3, 5):
            assert_allclose(mu.mat_power(A, k), np.linalg.matrix_power(A, k))

    def test_mat_power_k0_is_identity(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(mu.mat_power(A, 0), np.eye(2))

    def test_mat_power_with_vectors(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = np.array([1.0, 2.0])
        expected = a @ (A @ A @ A) @ a
        self.assertAlmostEqual(mu.mat_power(A, 3, a, a), expected, places=10)

    def test_mat_power_rejects_negative_k(self):
        with self.assertRaises(ValueError):
            mu.mat_power(D3, -1)

    def test_mat_power_rejects_one_sided_vectors(self):
        with self.assertRaises(ValueError):
            mu.mat_power(D3, 2, left_vec=V3)  # right_vec is missing

    def test_dense_vs_sparse(self):
        for k in (0, 1, 2, 5):
            self.assert_dense_sparse_equal(mu.mat_power, D3, k)

    def test_sparse_matrix_power_matches_dense_matrix_power(self):
        A = sp.csr_array(D3)
        result = mu.to_dense(mu.mat_power(A, 3))
        assert_allclose(result, np.linalg.matrix_power(D3, 3), atol=1e-11)


class TestSumMatPower(unittest.TestCase):
    def test_sum_mat_power_matches_formula(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        for k in (1, 3, 5):
            total = np.eye(A.shape[0])
            term = np.eye(A.shape[0])
            for _ in range(1, k):
                term = term @ A
                total = total + term
            assert_allclose(mu.sum_mat_power(A, k), total)

    def test_sum_mat_power_with_vectors(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = np.array([1.0, 2.0])
        total = np.eye(A.shape[0])
        term = np.eye(A.shape[0])
        for _ in range(1, 4):
            term = term @ A
            total = total + term
        expected = a @ (total @ a)
        self.assertAlmostEqual(mu.sum_mat_power(A, 4, a, a), expected, places=10)

    def test_sum_mat_power_rejects_k_less_than_1(self):
        with self.assertRaises(ValueError):
            mu.sum_mat_power(D3, 0)


class TestSolvePower(unittest.TestCase):
    def test_solve_power_k0_returns_v0(self):
        v0 = np.array([1.0, 2.0])
        assert_allclose(mu.solve_power(np.eye(2), 0, v0), v0)

    def test_solve_power_matches_repeated_solve(self):
        M = np.array([[2.0, 0.0], [0.0, 4.0]])
        # (M^-1)^3 * 1 = 1 / (2^3, 4^3)
        result = mu.solve_power(M, 3)
        expected = np.array([1 / 8, 1 / 64])
        assert_allclose(result, expected)

    def test_solve_power_dense_vs_sparse(self):
        M = np.eye(3) - D3
        dense = mu.solve_power(M, 3)
        for api in (sp.csc_matrix, sp.csc_array):
            sparse_result = mu.solve_power(api(M), 3)
            assert_allclose(sparse_result, dense, atol=1e-10)

    def test_solve_power_rejects_negative_k(self):
        with self.assertRaises(ValueError):
            mu.solve_power(np.eye(2), -1)


class TestDistanceScalarStats(unittest.TestCase):
    def test_distance_relative(self):
        v1 = np.array([2.0, 4.0])
        v2 = np.array([1.0, 4.0])
        self.assertAlmostEqual(mu.distance(v1, v2), 0.5, places=10)

    def test_scalar_extracts_1x1(self):
        self.assertEqual(mu.scalar(np.array([[7.0]])), 7.0)

    def test_average_and_variance_population(self):
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(mu.average(data), 2.5, places=10)
        self.assertAlmostEqual(mu.variance(data), 1.25, places=10)

    def test_cv_and_scv(self):
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(mu.cv(data), 1.25 / 6.25, places=10)
        self.assertAlmostEqual(mu.cv_true(data), np.sqrt(1.25) / 2.5, places=10)


class TestValidations(unittest.TestCase):
    def test_check_sub_stochastic_vector(self):
        self.assertTrue(mu.check_sub_stochastic_vector([0.4, 0.3]))
        self.assertFalse(mu.check_sub_stochastic_vector([0.7, 0.6]))
        self.assertFalse(mu.check_sub_stochastic_vector([1.2, -0.2]))

    def test_check_sub_generator_matrix_accepts_valid(self):
        self.assertTrue(mu.check_sub_generator_matrix(G3))

    def test_check_sub_generator_matrix_rejects_invalid_diagonal(self):
        positive_diagonal_col0 = np.array([[1.0, -1.0], [0.0, -1.0]])
        self.assertFalse(mu.check_sub_generator_matrix(positive_diagonal_col0))

    def test_check_sub_generator_matrix_accepts_closed_generator(self):
        full_generator = np.array([[-1.0, 1.0], [1.0, -1.0]])
        self.assertTrue(mu.check_sub_generator_matrix(full_generator))

    def test_check_sub_stochastic_matrix(self):
        self.assertTrue(mu.check_sub_stochastic_matrix(D3))
        recurrent_matrix = np.array([[1.0, 0.0], [0.5, 0.4]])
        self.assertFalse(mu.check_sub_stochastic_matrix(recurrent_matrix))

    def test_check_sub_stochastic_vector_rejects_negative_entry(self):
        self.assertFalse(mu.check_sub_stochastic_vector(np.array([-5.0, 0.5])))

    def test_check_sub_generator_matrix_rejects_non_square(self):
        self.assertFalse(mu.check_sub_generator_matrix(np.array([[-1.0, 0.5, 0.2]])))


class TestSpectralRadiusCriterion(unittest.TestCase):
    """
    sp(A) < 1 is decided by graph reachability (every phase reaches a phase
    whose row sums to less than 1), not by eigenvalues. The implementation is
    checked against the mathematical spectral-radius definition on random
    sub-stochastic matrices.
    """

    def test_matches_eigenvalues_on_random_matrices(self):
        rng = np.random.default_rng(12345)
        for _ in range(500):
            n = int(rng.integers(1, 7))
            A = rng.random((n, n)) * (rng.random((n, n)) < 0.5)
            A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-300)
            leak = rng.random(n) < 0.3
            A[leak] *= rng.uniform(0.3, 0.99, (int(leak.sum()), 1))
            expected = np.max(np.abs(np.linalg.eigvals(A))) < 1 - 1e-9
            self.assertEqual(mu._spectral_radius_lt_one(A), expected)
            self.assertEqual(mu._spectral_radius_lt_one(sp.csr_array(A)), expected)

    def test_closed_class_without_exit_is_rejected(self):
        # Phases 0 and 1 swap forever; only phase 2 leaks, and it is unreachable.
        A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
        self.assertFalse(mu._spectral_radius_lt_one(A))

    def test_large_sparse_bidiagonal_is_fast(self):
        # Every row but the last sums to exactly 1, so no row-sum shortcut
        # applies; an eigenvalue solver struggles here (one Jordan block).
        n = 10_000
        A = sp.diags_array([np.full(n, 0.5), np.full(n - 1, 0.5)], offsets=[0, 1],
                           format="csr")
        start = time.perf_counter()
        self.assertTrue(mu.check_sub_stochastic_matrix(A))
        self.assertLess(time.perf_counter() - start, 1.0)


class TestMatPowerVectorRoute(unittest.TestCase):
    def test_sparse_with_vectors_matches_dense(self):
        A = np.array([[0.5, 0.3, 0.0], [0.1, 0.4, 0.2], [0.0, 0.2, 0.6]])
        left, right = np.array([0.2, 0.5, 0.3]), np.array([0.3, 0.1, 0.4])
        for k in (0, 1, 7, 40):
            expected = mu.mat_power(A, k, left, right)
            for api in (sp.csr_array, sp.csr_matrix):
                self.assertAlmostEqual(mu.mat_power(api(A), k, left, right),
                                       expected, places=12)


class TestVec0Mat0(unittest.TestCase):
    def test_vec0(self):
        alpha = np.array([0.4, 0.3])
        self.assertAlmostEqual(mu.vec0(alpha), 0.3, places=10)

    def test_mat0_discrete(self):
        assert_allclose(mu.mat0(D3, discrete=True), np.ones(3) - D3 @ np.ones(3))

    def test_mat0_continuous(self):
        assert_allclose(mu.mat0(G3, discrete=False), -(G3 @ np.ones(3)))

    def test_dense_vs_sparse(self):
        for api in (sp.csr_matrix, sp.csr_array):
            assert_allclose(mu.to_dense(mu.mat0(api(D3), True)),
                            mu.mat0(D3, True), atol=1e-11)
            assert_allclose(mu.to_dense(mu.mat0(api(G3), False)),
                            mu.mat0(G3, False), atol=1e-11)


# =============================================================================
# Part 3: sparse dispatch infrastructure (is_sparse, eye_like, to_dense)
# =============================================================================

class TestSparseDispatchInfra(unittest.TestCase):
    def test_is_sparse(self):
        self.assertFalse(mu.is_sparse(D3))
        self.assertTrue(mu.is_sparse(sp.csr_matrix(D3)))
        self.assertTrue(mu.is_sparse(sp.csr_array(D3)))

    def test_eye_like_dense(self):
        assert_allclose(mu.eye_like(D3), np.eye(3))

    def test_eye_like_sparse_is_actually_sparse_and_correct(self):
        for api in (sp.csr_matrix, sp.csr_array):
            identity = mu.eye_like(api(D3))
            self.assertTrue(mu.is_sparse(identity))
            assert_allclose(mu.to_dense(identity), np.eye(3))

    def test_to_dense_noop_on_dense(self):
        assert_allclose(mu.to_dense(D3), D3)

    def test_to_dense_on_sparse(self):
        assert_allclose(mu.to_dense(sp.csr_array(D3)), D3)


if __name__ == "__main__":
    unittest.main()
