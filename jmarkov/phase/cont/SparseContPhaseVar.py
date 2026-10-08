"""
jmarkov/phase/sparse_cont_phase_var.py

Continuous phase-type distribution with sparse storage.

Java: jphase.SparseContPhaseVar. All PH mathematics lives in
AbstractContPhaseVar; this class only fixes the storage policy
(canonical scipy.sparse.csr_array) and the concrete type of closure results
and copies. Java's SparseContPhaseVar has no factory methods.

Requires SciPy >= 1.12: matrix_utils uses ``eye_array`` / ``diags_array`` for
sparse arrays.
"""

from __future__ import annotations

from typing import Any

from scipy import sparse

from AbstractContPhaseVar import AbstractContPhaseVar, ArrayLike

__all__ = ["SparseContPhaseVar"]

if not hasattr(sparse, "eye_array"):  # pragma: no cover
    raise ImportError("SparseContPhaseVar requires SciPy >= 1.12.")


class SparseContPhaseVar(AbstractContPhaseVar):
    """
    Continuous PH distribution PH(alpha, T) with T stored as a canonical CSR
    sparse array (float64, sorted indices, no duplicates, no explicit zeros).

    ``SparseContPhaseVar(alpha, T)`` accepts lists, NumPy arrays or any
    scipy.sparse matrix/array. Parameters and methods are those of
    AbstractContPhaseVar.
    """

    # =========================================================================
    # ABSTRACT CONTRACT
    # =========================================================================

    def _to_storage(self, T: Any) -> sparse.csr_array:
        """Return T as a new canonical float64 csr_array; never shares buffers."""
        # csr_array(T) without copy=True shares data/indices/indptr with a CSR
        # input, and in closures T is another variable's internal storage.
        M = sparse.csr_array(T, dtype=float, copy=True)
        M.sum_duplicates()
        M.eliminate_zeros()
        return M

    def new_var(self, alpha: ArrayLike, T: Any) -> "SparseContPhaseVar":
        """Return a SparseContPhaseVar for a closure result."""
        # Same reasoning as Dense: results are valid by construction and Java
        # never validates them; re-validation only adds absolute-tolerance
        # artifacts (e.g. times(1e12)).
        return SparseContPhaseVar(alpha, T, validate=False, tol=self.tol)

    def copy(self) -> "SparseContPhaseVar":
        """Return an independent copy with no shared sparse buffers."""
        return type(self)(self._alpha, self._T, validate=False, tol=self.tol)
