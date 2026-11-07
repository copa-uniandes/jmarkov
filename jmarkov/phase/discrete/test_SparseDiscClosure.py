"""
test_SparseDiscClosure.py

Tests for the closure methods of SparseDiscretePhaseType.

The sparse closure tests re-run the dense closure representation and
distribution checks with matrices stored sparsely, once per SciPy sparse API,
against the same expected values.
SparseDiscClosureStorageTest adds what only matters for sparse storage:
results stay sparse, and combining dense and sparse variables follows the
storage type of the variable on which the operation is called.

Run with:
    python3 -m unittest test_SparseDiscClosure -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import scipy.sparse as sp
from numpy.testing import assert_allclose

import discrete_fixtures as fx
import test_DenseDiscClosure as dense
from DenseDiscPhaseVar import DenseDiscretePhaseType
from SparseDiscPhaseVar import SparseDiscretePhaseType


def make_csr_array(vector, matrix):
    return SparseDiscretePhaseType(vector, sp.csr_array(matrix))


def make_csr_matrix(vector, matrix):
    return SparseDiscretePhaseType(vector, sp.csr_matrix(matrix))


class SparseDiscClosureTest(dense.DenseDiscClosureTest):
    """Closure representation tests using csr_array storage."""
    make = staticmethod(make_csr_array)


class SparseDiscClosureTestSpmatrix(dense.DenseDiscClosureTest):
    make = staticmethod(make_csr_matrix)


class SparseDiscClosureDistributionTest(dense.DenseDiscClosureDistributionTest):
    make = staticmethod(make_csr_array)


class SparseDiscClosureDistributionTestSpmatrix(dense.DenseDiscClosureDistributionTest):
    make = staticmethod(make_csr_matrix)


class SparseDiscClosureStorageTest(unittest.TestCase):

    def setUp(self):
        self.dense1 = DenseDiscretePhaseType(*fx.fixture("closure1"))
        self.dense2 = DenseDiscretePhaseType(*fx.fixture("closure2"))
        self.sparse1 = make_csr_array(*fx.fixture("closure1"))
        self.sparse2 = make_csr_array(*fx.fixture("closure2"))
        self.sparseD = make_csr_array(*fx.fixture("closureD"))

    def operations(self, X, Y, D):
        return {
            "sum": X.sum(Y), "min": X.min(Y), "max": X.max(Y),
            "mix": X.mix(fx.MIX_P, Y), "sum_geom": X.sum_geom(fx.SUM_GEOM_P),
            "sum_ph": X.sum_ph(D),
        }

    def test_results_stay_sparse(self):
        for name, res in self.operations(self.sparse1, self.sparse2, self.sparseD).items():
            self.assertIsInstance(res, SparseDiscretePhaseType, name)
            self.assertTrue(sp.issparse(res.A), name)

    def test_storage_follows_the_calling_variable(self):
        for name, res in self.operations(self.dense1, self.sparse2, self.sparseD).items():
            self.assertIsInstance(res, DenseDiscretePhaseType, name)
            self.assertFalse(sp.issparse(res.A), name)
        for name, res in self.operations(self.sparse1, self.dense2, self.sparseD).items():
            self.assertIsInstance(res, SparseDiscretePhaseType, name)
            self.assertTrue(sp.issparse(res.A), name)

    def test_mixed_storage_gives_same_distribution(self):
        all_dense = self.operations(self.dense1, self.dense2,
                                    DenseDiscretePhaseType(*fx.fixture("closureD")))
        mixed = self.operations(self.dense1, self.sparse2, self.sparseD)
        for name in all_dense:
            assert_allclose(mixed[name].pmf_range(30), all_dense[name].pmf_range(30),
                            atol=1e-12, err_msg=name)


if __name__ == "__main__":
    unittest.main()
