"""
test_SparseDiscProb.py

Tests for the probability methods of SparseDiscretePhaseType.

Discrete counterpart of jMarkov's test/jphase/SparseContProbTest.java. As
in Java, where the Sparse test repeats the Dense one with a
FlexCompRowMatrix, every test of test_DenseDiscProb is re-run here
with the matrix stored sparsely - once per SciPy sparse API (csr_array
and the older csr_matrix) - against the same expected values.

Run with:
    python3 -m unittest test_SparseDiscProb -v

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import unittest

import scipy.sparse as sp

import test_DenseDiscProb as dense
from SparseDiscPhaseVar import SparseDiscretePhaseType


class SparseDiscProbTest(dense.DenseDiscProbTest):
    """Mirror of SparseContProbTest.java (csr_array storage)."""

    @staticmethod
    def make(vector, matrix):
        return SparseDiscretePhaseType(vector, sp.csr_array(matrix))

    def test_storage_is_sparse(self):
        for var in (self.var1, self.var2, self.var3):
            self.assertTrue(sp.issparse(var.A))


class SparseDiscProbTestSpmatrix(SparseDiscProbTest):
    """Same tests with the legacy csr_matrix API."""

    @staticmethod
    def make(vector, matrix):
        return SparseDiscretePhaseType(vector, sp.csr_matrix(matrix))


if __name__ == "__main__":
    unittest.main()
