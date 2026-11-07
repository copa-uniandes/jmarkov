"""
jmarkov/phase/abstract_cont_phase_var.py

Abstract continuous phase-type distribution X ~ PH(alpha, T).

Consolidates the Java hierarchy
JMarkovElement -> PhaseVar -> ContPhaseVar -> AbstractContPhaseVar.
Concrete classes implement the storage policy (``_to_storage``),
``new_var`` and ``copy``.

Conventions
-----------
    alpha_0 = 1 - alpha @ 1          atom at x = 0
    t       = -T @ 1                 exit rates
    S(x) = alpha expm(T x) 1,   F(x) = 1 - S(x),   f(x) = alpha expm(T x) t
    E[X^k] = k! alpha (-T)^{-k} 1
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy import optimize, sparse

from matrix_utils import (
    EPS,
    check_sub_generator_matrix,
    check_sub_stochastic_matrix,
    check_sub_stochastic_vector,
    coerce_representation,
    concat_cols,
    concat_quad,
    concat_vectors,
    expm_left_multiply,
    eye_like,
    format_representation,
    is_sparse,
    kronecker,
    kronecker_col_vector_mx,
    kronecker_mx_col_vector,
    kronecker_sum,
    kronecker_vectors,
    mat0,
    mult_vector_like,
    non_transient_phases,
    solve_power,
    to_dense,
    vec0,
    zeros_like,
)

ArrayLike = npt.ArrayLike
FloatOrArray = Union[float, np.ndarray]

__all__ = ["AbstractContPhaseVar", "DiscPhaseVarLike"]


@runtime_checkable
class DiscPhaseVarLike(Protocol):
    """Minimal discrete PH interface required by ``sum_ph``."""

    @property
    def alpha(self) -> np.ndarray: ...

    @property
    def T(self) -> Any: ...


class AbstractContPhaseVar(ABC):
    """
    Abstract continuous phase-type distribution PH(alpha, T).

    Parameters
    ----------
    alpha : array_like, shape (n,)
        Sub-stochastic initial vector; alpha_0 = 1 - sum(alpha) is an atom at 0.
    T : array_like or scipy.sparse matrix, shape (n, n)
        Sub-generator; every phase must reach absorption.
    validate : bool
        Validate the representation (default True).
    tol : float
        Validation tolerance.
    """

    __hash__ = None  # type: ignore[assignment]  # mutable, value-based equality

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================

    def __init__(self, alpha: ArrayLike, T: Any, *, validate: bool = True,
                 tol: float = EPS) -> None:
        self._tol = float(tol)
        self._set_representation(alpha, T, validate)

    def _set_representation(self, alpha: ArrayLike, T: Any, validate: bool) -> None:
        a, M, n = coerce_representation(alpha, T)
        if n == 0:
            raise ValueError("A phase-type distribution needs at least one phase.")
        # Private float64 copies: no aliasing with caller arrays (Java stores references).
        a = np.array(a, dtype=float)
        M = M.astype(float) if is_sparse(M) else np.array(M, dtype=float)
        M = self._to_storage(M)
        if validate:
            self._validate(a, M, self._tol)
        a.setflags(write=False)
        if not is_sparse(M):
            M.setflags(write=False)
        self._alpha, self._T = a, M

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def alpha(self) -> np.ndarray:
        """Initial vector (read-only view). Assignment is validated."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: ArrayLike) -> None:
        self._set_representation(value, self._T, True)

    @property
    def T(self):
        """Sub-generator: read-only if dense, a copy if sparse. Assignment is validated."""
        return self._T.copy() if is_sparse(self._T) else self._T

    @T.setter
    def T(self, value: Any) -> None:
        self._set_representation(self._alpha, value, True)

    def set_representation(self, alpha: ArrayLike, T: Any, *,
                           validate: bool = True) -> None:
        """Replace (alpha, T) at once; needed when the number of phases changes."""
        self._set_representation(alpha, T, validate)

    @property
    def n_phases(self) -> int:
        """Number of transient phases."""
        return int(self._alpha.shape[0])

    @property
    def vec0(self) -> float:
        """Probability mass at zero, alpha_0 = 1 - alpha @ 1."""
        return vec0(self._alpha)

    @property
    def mat0(self) -> np.ndarray:
        """Exit vector t = -T @ 1."""
        return mat0(self._T, discrete=False)

    @property
    def tol(self) -> float:
        """Validation tolerance."""
        return self._tol

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate(alpha: np.ndarray, T: Any, tol: float) -> None:
        """Raise ValueError if (alpha, T) is not a valid continuous PH."""
        if not check_sub_stochastic_vector(alpha, tol):
            raise ValueError(
                f"alpha must satisfy alpha_i >= 0 and sum(alpha) <= 1 "
                f"(min = {alpha.min():.6g}, sum = {alpha.sum():.6g})."
            )
        if not check_sub_generator_matrix(T, tol):
            raise ValueError(
                "T must be a sub-generator: T_ii < 0, T_ij >= 0 (i != j), "
                "row sums <= 0 and at least one row sum < 0."
            )
        bad = non_transient_phases(T, discrete=False, tol=tol)
        if bad.size:
            raise ValueError(
                f"Absorption is unreachable from phases {bad.tolist()}; T is singular."
            )

    # =========================================================================
    # DISTRIBUTION FUNCTIONS
    # =========================================================================

    @staticmethod
    def _points(x: ArrayLike, name: str = "x") -> Tuple[np.ndarray, Tuple[int, ...]]:
        arr = np.asarray(x, dtype=float)
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN.")
        return arr.ravel(), arr.shape

    @staticmethod
    def _restore(values: np.ndarray, shape: Tuple[int, ...]) -> FloatOrArray:
        return float(values[0]) if shape == () else values.reshape(shape)

    def _alpha_exp(self, xs: np.ndarray) -> np.ndarray:
        """Rows alpha expm(T max(x, 0)); infinite x gives a zero row."""
        rows = np.zeros((xs.size, self.n_phases))
        finite = np.isfinite(xs)
        rows[finite] = expm_left_multiply(self._T, np.maximum(xs[finite], 0.0),
                                          self._alpha)
        return rows

    def survival(self, x: ArrayLike) -> FloatOrArray:
        """Survival function S(x) = P(X > x); S(x) = 1 for x < 0."""
        xs, shape = self._points(x)
        res = self._alpha_exp(xs).sum(axis=1)
        res[xs < 0.0] = 1.0
        return self._restore(np.clip(res, 0.0, 1.0), shape)

    def cdf(self, x: ArrayLike) -> FloatOrArray:
        """Cumulative distribution F(x) = 1 - S(x); F(0) = alpha_0."""
        xs, shape = self._points(x)
        return self._restore(1.0 - np.asarray(self.survival(xs)), shape)

    def pdf(self, x: ArrayLike) -> FloatOrArray:
        """Density of the continuous part, f(x) = alpha expm(T x) t."""
        # f(0) = alpha @ t is the right limit; the atom alpha_0 has no density.
        xs, shape = self._points(x)
        res = self._alpha_exp(xs) @ self.mat0
        res[xs < 0.0] = 0.0
        return self._restore(np.maximum(res, 0.0), shape)

    def prob(self, a: ArrayLike, b: ArrayLike) -> FloatOrArray:
        """P(a < X <= b); zero when b <= a."""
        a_arr, b_arr = np.broadcast_arrays(np.asarray(a, dtype=float),
                                           np.asarray(b, dtype=float))
        diff = np.asarray(self.cdf(b_arr)) - np.asarray(self.cdf(a_arr))
        res = np.where(b_arr > a_arr, np.maximum(diff, 0.0), 0.0)
        return float(res) if res.ndim == 0 else res

    def quantile(self, p: ArrayLike) -> FloatOrArray:
        """Generalized inverse Q(p) = inf{x >= 0 : F(x) >= p}."""
        ps, shape = self._points(p, "p")
        return self._restore(np.array([self._quantile_scalar(float(q)) for q in ps]),
                             shape)

    def _quantile_scalar(self, p: float) -> float:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        if p <= self.vec0:
            return 0.0
        if p >= 1.0:
            return math.inf
        # Java uses unsafeguarded Newton; bracketing + Brent always converges
        # because F is continuous and strictly increasing on (0, inf).
        mean = self.expected_value()
        lo, hi = 0.0, max(mean, np.finfo(float).tiny)
        for _ in range(2000):
            if self.cdf(hi) >= p:
                break
            lo, hi = hi, 2.0 * hi
        else:  # pragma: no cover
            raise RuntimeError(f"Could not bracket the quantile for p = {p}.")
        return float(optimize.brentq(lambda x: self.cdf(x) - p, lo, hi,
                                     xtol=1e-14 * max(mean, 1e-300), rtol=1e-13,
                                     maxiter=500))

    def median(self) -> float:
        """Median, Q(0.5)."""
        return self.quantile(0.5)

    # =========================================================================
    # MOMENTS
    # =========================================================================

    def moment(self, k: int) -> float:
        """Return the k-th raw moment E[X^k] = k! alpha (-T)^{-k} 1."""
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError(f"k must be an integer; got {type(k).__name__}.")
        if k < 0:
            raise ValueError(f"k must be >= 0; got {k}.")
        if k == 0:
            return 1.0  # Java returns alpha @ 1.
        # Java uses CG on normal equations (cond(T)^(2k)); repeated solves instead.
        v = solve_power(-self._T, int(k))
        return float(math.factorial(k) * (self._alpha @ v))

    def expected_value(self) -> float:
        """Return E[X]."""
        return self.moment(1)

    def variance(self) -> float:
        """Return Var[X] = E[X^2] - E[X]^2."""
        m1 = self.moment(1)
        return self.moment(2) - m1 * m1

    def std_deviation(self) -> float:
        """Return the standard deviation."""
        return math.sqrt(max(self.variance(), 0.0))

    def cv(self) -> float:
        """Return the SQUARED coefficient of variation, Var[X] / E[X]^2."""
        # Same value as Java CV() (and matrix_utils.cv); the Java name is misleading.
        m1 = self.moment(1)
        if m1 == 0.0:  # only when alpha = 0 (X = 0 a.s.); Java returns NaN.
            raise ValueError("cv (SCV) is undefined when E[X] = 0.")
        return self.moment(2) / (m1 * m1) - 1.0

    def cv_true(self) -> float:
        """Return the coefficient of variation, std / E[X]."""
        return math.sqrt(max(self.cv(), 0.0))

    # =========================================================================
    # LOSS FUNCTIONS
    # =========================================================================

    def loss_function_1(self, x: ArrayLike) -> FloatOrArray:
        """First-order loss E[(X - x)^+]."""
        xs, shape = self._points(x)
        w1 = solve_power(-self._T, 1)
        res = self._alpha_exp(xs) @ w1
        neg = xs < 0.0
        if neg.any():
            res[neg] = float(self._alpha @ w1) - xs[neg]
        return self._restore(res, shape)

    def loss_function_2(self, x: ArrayLike) -> FloatOrArray:
        """Second-order loss int_x^inf L1(u) du = E[((X - x)^+)^2] / 2."""
        xs, shape = self._points(x)
        w1 = solve_power(-self._T, 1)
        w2 = solve_power(-self._T, 1, w1)
        res = self._alpha_exp(xs) @ w2
        neg = xs < 0.0
        if neg.any():
            m1, half_m2 = float(self._alpha @ w1), float(self._alpha @ w2)
            xn = xs[neg]
            res[neg] = half_m2 - xn * m1 + 0.5 * xn * xn
        return self._restore(res, shape)

    # =========================================================================
    # CLOSURE OPERATIONS
    # =========================================================================

    @staticmethod
    def _require_cont(other: Any, op: str) -> "AbstractContPhaseVar":
        if not isinstance(other, AbstractContPhaseVar):
            raise TypeError(f"{op}: expected a continuous PH; "
                            f"got {type(other).__name__}.")
        return other

    def sum(self, other: "AbstractContPhaseVar") -> "AbstractContPhaseVar":
        """Convolution X + Y of independent PH variables."""
        other = self._require_cont(other, "sum")
        # Java returns the operand itself here (aliasing).
        if not np.any(self._alpha):
            return self.new_var(other._alpha, other._T)
        if not np.any(other._alpha):
            return self.new_var(self._alpha, self._T)
        S = self._to_storage(other._T)
        n1, n2 = self.n_phases, other.n_phases
        new_alpha = concat_vectors(self._alpha, self.vec0 * other._alpha)
        new_T = concat_quad(self._T, mult_vector_like(self.mat0, other._alpha, self._T),
                            zeros_like(self._T, (n2, n1)), S)
        return self.new_var(new_alpha, new_T)

    def mix(self, p: float, other: "AbstractContPhaseVar") -> "AbstractContPhaseVar":
        """Convex mixture: X with probability p, Y with probability 1 - p."""
        other = self._require_cont(other, "mix")
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"mix: p must be in [0, 1]; got {p}.")
        S = self._to_storage(other._T)
        n1, n2 = self.n_phases, other.n_phases
        new_alpha = concat_vectors(p * self._alpha, (1.0 - p) * other._alpha)
        new_T = concat_quad(self._T, zeros_like(self._T, (n1, n2)),
                            zeros_like(self._T, (n2, n1)), S)
        return self.new_var(new_alpha, new_T)

    def min(self, other: "AbstractContPhaseVar") -> "AbstractContPhaseVar":
        """Minimum of independent PH variables: (alpha kron beta, T kronsum S)."""
        other = self._require_cont(other, "min")
        S = self._to_storage(other._T)
        return self.new_var(kronecker_vectors(self._alpha, other._alpha),
                            kronecker_sum(self._T, S))

    def max(self, other: "AbstractContPhaseVar") -> "AbstractContPhaseVar":
        """Maximum of independent PH variables (n1*n2 + n1 + n2 phases)."""
        other = self._require_cont(other, "max")
        S = self._to_storage(other._T)
        n1, n2 = self.n_phases, other.n_phases
        new_alpha = concat_vectors(
            kronecker_vectors(self._alpha, other._alpha),
            concat_vectors(other.vec0 * self._alpha, self.vec0 * other._alpha),
        )
        # Both alive -> Y absorbs (rate s): only X left; X absorbs (rate t): only Y left.
        right_up = concat_cols(kronecker_mx_col_vector(eye_like(self._T), other.mat0),
                               kronecker_col_vector_mx(self.mat0, eye_like(S)))
        bottom_right = concat_quad(self._T, zeros_like(self._T, (n1, n2)),
                                   zeros_like(self._T, (n2, n1)), S)
        new_T = concat_quad(kronecker_sum(self._T, S), right_up,
                            zeros_like(self._T, (n1 + n2, n1 * n2)), bottom_right)
        return self.new_var(new_alpha, new_T)

    def times(self, c: float) -> "AbstractContPhaseVar":
        """Scaled variable cX = PH(alpha, T / c), c > 0."""
        c = float(c)
        if not math.isfinite(c) or c <= 0.0:
            raise ValueError(f"times: c must be finite and > 0; got {c}.")
        return self.new_var(self._alpha, self._T / c)

    def sum_geom(self, p: float) -> "AbstractContPhaseVar":
        """Sum of N i.i.d. copies, N ~ Geometric(p) on {1, 2, ...}."""
        p = float(p)
        if not 0.0 < p <= 1.0:
            raise ValueError(f"sum_geom: p must be in (0, 1]; got {p}.")
        # Corrects Java when alpha_0 > 0: zero-length copies must not stop the sum.
        c = 1.0 / (1.0 - (1.0 - p) * self.vec0)
        restart = mult_vector_like(self.mat0, self._alpha, self._T)
        return self.new_var(c * self._alpha, self._T + ((1.0 - p) * c) * restart)

    def sum_ph(self, B: Union[DiscPhaseVarLike, Tuple[ArrayLike, ArrayLike]]
               ) -> "AbstractContPhaseVar":
        """
        Sum of N i.i.d. copies, N ~ DPH(beta, S).

        ``B`` is a discrete PH exposing ``alpha`` and ``T`` or a tuple (beta, S).
        Phases are ordered (phase of X) x (phase of N).
        """
        beta, S = self._disc_representation(B)
        m = beta.shape[0]
        Sd = to_dense(S)  # (I - alpha_0 S)^{-1} is dense in general.
        M = np.eye(m) - self.vec0 * Sd
        # Corrects Java: no extra (1 - alpha_0) factor on the restart term.
        restart = np.linalg.solve(M, Sd)       # (I - alpha_0 S)^{-1} S
        start = np.linalg.solve(M.T, beta)     # beta (I - alpha_0 S)^{-1}
        new_T = (kronecker(self._T, eye_like(self._T, m))
                 + kronecker(mult_vector_like(self.mat0, self._alpha, self._T), restart))
        return self.new_var(kronecker_vectors(self._alpha, start), new_T)

    def _disc_representation(self, B: Any) -> Tuple[np.ndarray, Any]:
        if isinstance(B, AbstractContPhaseVar):
            raise TypeError("sum_ph: B must be a discrete PH.")
        if isinstance(B, tuple) and len(B) == 2:
            beta, S = B
        elif isinstance(B, DiscPhaseVarLike):
            beta, S = B.alpha, B.T
        else:
            raise TypeError("sum_ph: B must expose alpha and T, or be a tuple (beta, S).")
        beta, S, _ = coerce_representation(beta, S)
        if not check_sub_stochastic_vector(beta, self._tol):
            raise ValueError("sum_ph: beta must be sub-stochastic.")
        if not check_sub_stochastic_matrix(S, self._tol):
            raise ValueError("sum_ph: S must be sub-stochastic with sp(S) < 1.")
        return np.asarray(beta, dtype=float), S

    def residual_time(self, x: float) -> "AbstractContPhaseVar":
        """Residual time X - x given X > x: PH(alpha expm(T x) / S(x), T)."""
        x = float(x)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError(f"residual_time: x must be finite and >= 0; got {x}.")
        v = self._alpha_exp(np.array([x]))[0]
        surv = float(v.sum())
        if not surv > 0.0:
            raise ValueError(f"residual_time: P(X > {x}) is numerically zero.")
        return self.new_var(v / surv, self._T)

    def _eq_residual_vector(self) -> np.ndarray:
        # Normalizing by the same solve makes the result sum to 1 exactly.
        y = solve_power(-self._T.T, 1, self._alpha)   # alpha (-T)^{-1}
        mean = float(y.sum())
        if not mean > 0.0:
            raise ValueError("E[X] = 0: the equilibrium residual is undefined.")
        return y / mean

    def eq_residual_time(self) -> "AbstractContPhaseVar":
        """Equilibrium residual time, density S(x) / E[X]."""
        return self.new_var(self._eq_residual_vector(), self._T)

    def waiting_q(self, rho: float) -> "AbstractContPhaseVar":
        """Waiting time in queue of an M/PH/1 queue with utilization rho."""
        rho = float(rho)
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"waiting_q: rho must be in [0, 1); got {rho}.")
        new_alpha = rho * self._eq_residual_vector()
        new_T = self._T + mult_vector_like(self.mat0, new_alpha, self._T)
        return self.new_var(new_alpha, new_T)

    def residual_var(self, a: float) -> "AbstractContPhaseVar":
        """Loss variable (X - a)^+ = PH(alpha expm(T a), T), a >= 0."""
        a = float(a)
        if not math.isfinite(a) or a < 0.0:
            raise ValueError(f"residual_var: a must be finite and >= 0; got {a}.")
        return self.new_var(self._alpha_exp(np.array([a]))[0], self._T)

    # =========================================================================
    # DESCRIPTION AND COMPARISON (JMarkovElement)
    # =========================================================================

    def label(self) -> str:
        """Short label."""
        return f"CPH - {self.n_phases} Phases"

    def description(self, include_stats: bool = False) -> str:
        """Multi-line description of the representation."""
        stats = None
        if include_stats:
            stats = {"E[X]": self.expected_value(), "Var[X]": self.variance(),
                     "SCV": self.cv()}
        return format_representation("Continuous Phase-Type Distribution",
                                     self._alpha, self._T, "T", stats)

    def __str__(self) -> str:
        return self.description()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_phases={self.n_phases})"

    def __eq__(self, other: object) -> bool:
        """Exact equality of the representation, regardless of storage format."""
        if not isinstance(other, AbstractContPhaseVar):
            return NotImplemented
        if self.n_phases != other.n_phases or not np.array_equal(self._alpha,
                                                                 other._alpha):
            return False
        A, B = self._T, other._T
        if is_sparse(A) or is_sparse(B):
            return (sparse.csr_matrix(A) != sparse.csr_matrix(B)).nnz == 0
        return bool(np.array_equal(A, B))

    def allclose(self, other: "AbstractContPhaseVar", rtol: float = 1e-9,
                 atol: float = 1e-12) -> bool:
        """Equality of the representation within tolerances (densifies T)."""
        other = self._require_cont(other, "allclose")
        return (self.n_phases == other.n_phases
                and bool(np.allclose(self._alpha, other._alpha, rtol=rtol, atol=atol))
                and bool(np.allclose(to_dense(self._T), to_dense(other._T),
                                     rtol=rtol, atol=atol)))

    # =========================================================================
    # ABSTRACT CONTRACT
    # =========================================================================

    @abstractmethod
    def _to_storage(self, T: Any) -> Any:
        """
        Return T in this class's storage format (e.g. ndarray or CSR).

        Receives a private float64 copy. This is the only representation-
        dependent step (Java: the DenseMatrix / FlexCompRowMatrix cast).
        """

    @abstractmethod
    def new_var(self, alpha: ArrayLike, T: Any) -> "AbstractContPhaseVar":
        """
        Return a new variable of the appropriate concrete type for (alpha, T).

        Used by closure operations with representations that are valid by
        construction, so implementations may pass ``validate=False``.
        Parametric subclasses should return a generic (e.g. dense) variable.
        """

    @abstractmethod
    def copy(self) -> "AbstractContPhaseVar":
        """Return an independent deep copy of the same concrete type."""
