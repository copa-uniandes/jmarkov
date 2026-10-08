"""
jmarkov/phase/dense_cont_phase_var.py

Continuous phase-type distribution with dense storage.

Java: jphase.DenseContPhaseVar. All PH mathematics lives in
AbstractContPhaseVar; this class fixes the storage policy (numpy.ndarray),
the concrete type of closure results and copies, and provides the Java
factory methods for common PH distributions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg as sla

from AbstractContPhaseVar import AbstractContPhaseVar, ArrayLike
from matrix_utils import to_dense

__all__ = ["DenseContPhaseVar"]


class DenseContPhaseVar(AbstractContPhaseVar):
    """
    Continuous PH distribution PH(alpha, T) with T stored as a dense ndarray.

    ``DenseContPhaseVar(alpha, T)`` accepts lists, NumPy arrays or
    scipy.sparse matrices (densified). Parameters and methods are those of
    AbstractContPhaseVar.
    """

    # =========================================================================
    # ABSTRACT CONTRACT
    # =========================================================================

    def _to_storage(self, T: Any) -> np.ndarray:
        """Return T as a dense float64 ndarray (no copy if it already is one)."""
        return to_dense(T)

    def new_var(self, alpha: ArrayLike, T: Any) -> "DenseContPhaseVar":
        """Return a DenseContPhaseVar for a closure result."""
        # Closure results are valid by construction, and Java's newVar + setMatrix
        # never validates. Re-validating would only add absolute-tolerance
        # artifacts, e.g. times(1e12) makes T_ii = -1e-12 >= -tol.
        return DenseContPhaseVar(alpha, T, validate=False, tol=self.tol)

    def copy(self) -> "DenseContPhaseVar":
        """Return an independent copy; the constructor makes the private copies."""
        return type(self)(self.alpha, self.T, validate=False, tol=self.tol)

    # =========================================================================
    # FACTORY METHODS (Java static constructors)
    # =========================================================================

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value}.")
        return int(value)

    @staticmethod
    def _coxian_matrix(rates: ArrayLike, cont_probs: ArrayLike) -> np.ndarray:
        """Bidiagonal sub-generator: T_ii = -rates[i], T_i,i+1 = cont_probs[i] * rates[i]."""
        r = np.asarray(rates, dtype=float).ravel()
        p = np.asarray(cont_probs, dtype=float).ravel()
        if r.size == 0 or p.size != r.size - 1:
            raise ValueError(f"Expected n >= 1 rates and n - 1 probabilities; "
                             f"got {r.size} and {p.size}.")
        return np.diag(-r) + np.diag(p * r[:-1], 1)

    @classmethod
    def expo(cls, lam: float) -> "DenseContPhaseVar":
        """Exponential distribution with rate lam. Java: expo(double)."""
        return cls.erlang(lam, 1)

    @classmethod
    def erlang(cls, lam: float, n: int) -> "DenseContPhaseVar":
        """Erlang distribution: n phases with rate lam. Java: Erlang(double, int)."""
        return cls.hypo_exponential(np.full(cls._positive_int(n, "n"), float(lam)))

    @classmethod
    def hypo_exponential(cls, lambdas: ArrayLike) -> "DenseContPhaseVar":
        """Phases in series with rates lambdas. Java: HipoExponential(double[])."""
        lambdas = np.asarray(lambdas, dtype=float).ravel()
        return cls.coxian(lambdas, np.ones(max(lambdas.size - 1, 0)))

    @classmethod
    def coxian(cls, lambdas: ArrayLike, probs: ArrayLike) -> "DenseContPhaseVar":
        """
        Coxian distribution; probs[i] is the probability of moving from phase i
        to phase i + 1. Java: Coxian(int, double[], double[]); n = len(lambdas).
        """
        T = cls._coxian_matrix(lambdas, probs)
        alpha = np.zeros(T.shape[0])
        alpha[0] = 1.0
        return cls(alpha, T)

    @classmethod
    def hyper_expo(cls, lambdas: ArrayLike, probs: ArrayLike) -> "DenseContPhaseVar":
        """Mixture of exponentials. Java: HyperExpo(double[], double[])."""
        lambdas = np.asarray(lambdas, dtype=float).ravel()
        return cls.hyper_erlang(lambdas, [1] * lambdas.size, probs)

    @classmethod
    def hyper_erlang(cls, lambdas: ArrayLike, phases: ArrayLike,
                     probs: ArrayLike) -> "DenseContPhaseVar":
        """
        Mixture of Erlang branches: branch i has phases[i] phases with rate
        lambdas[i] and weight probs[i].
        Java: HyperErlang(int k, double[], int[], double[]); k = len(lambdas).
        """
        lambdas = np.asarray(lambdas, dtype=float).ravel()
        probs = np.asarray(probs, dtype=float).ravel()
        # Java does not check phases[i] >= 1; an empty branch shifts the weights.
        phases = [cls._positive_int(r, "phases[i]") for r in np.asarray(phases).ravel()]
        if lambdas.size == 0 or not lambdas.size == len(phases) == probs.size:
            raise ValueError("hyper_erlang: lambdas, phases and probs must have the "
                             "same nonzero length.")
        T = sla.block_diag(*(cls._coxian_matrix(np.full(r, lam), np.ones(r - 1))
                             for lam, r in zip(lambdas, phases)))
        alpha = np.zeros(T.shape[0])
        alpha[np.cumsum([0] + phases[:-1])] = probs
        return cls(alpha, T)

    @classmethod
    def erlang_coxian(cls, n: int, p: float, lambda_y: float, lambda_x1: float,
                      lambda_x2: float, px: float) -> "DenseContPhaseVar":
        """
        Erlang-Coxian distribution (Osogami & Harchol-Balter): with probability
        p, Erlang(n - 2, lambda_y) followed by a two-phase Coxian (lambda_x1,
        px, lambda_x2); otherwise 0. n = 1 is an exponential with rate lambda_x1.
        Java: ErlangCoxian(int, double, double, double, double, double).
        """
        n = cls._positive_int(n, "n")
        # Corrects Java: the atom 1 - p is kept for every n (Java drops it for n <= 2).
        if n == 1:
            return cls([p], [[-lambda_x1]])
        T = cls._coxian_matrix([lambda_y] * (n - 2) + [lambda_x1, lambda_x2],
                               [1.0] * (n - 2) + [px])
        alpha = np.zeros(n)
        alpha[0] = p
        return cls(alpha, T)

    @classmethod
    def from_var(cls, var: AbstractContPhaseVar) -> "DenseContPhaseVar":
        """
        Dense copy of any continuous PH variable (e.g. sparse or parametric).
        Java: HyperErlang(HyperErlangVar).
        """
        if not isinstance(var, AbstractContPhaseVar):
            raise TypeError(f"from_var: expected a continuous PH; "
                            f"got {type(var).__name__}.")
        return cls(var._alpha, var._T, validate=False, tol=var.tol)
