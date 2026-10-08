"""
test_SparseDiscMoment.py

Tests for the moment methods of SparseDiscretePhaseType.

Every test of test_DenseDiscMoment is re-run with the matrix stored sparsely,
once per SciPy sparse API, against the same expected values.

Run with:
    python3 -m unittest test_SparseDiscMoment -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import scipy.sparse as sp

import test_DenseDiscMoment as dense
from SparseDiscPhaseVar import SparseDiscretePhaseType


class SparseDiscMomentTest(dense.DenseDiscMomentTest):
    """Mirror of SparseContMomentTest.java (csr_array storage)."""

    @staticmethod
    def make(vector, matrix):
        return SparseDiscretePhaseType(vector, sp.csr_array(matrix))

    def test_storage_is_sparse(self):
        self.assertTrue(sp.issparse(self.var.A))


class SparseDiscMomentTestSpmatrix(SparseDiscMomentTest):
    """Same tests with the legacy csr_matrix API."""

    @staticmethod
    def make(vector, matrix):
        return SparseDiscretePhaseType(vector, sp.csr_matrix(matrix))


if __name__ == "__main__":
    unittest.main()
