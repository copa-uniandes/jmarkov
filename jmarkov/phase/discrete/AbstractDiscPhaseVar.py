"""
AbstractDiscPhaseVar.py

Abstract class for discrete Phase-Type distributions.

Migration of jphase.AbstractDiscPhaseVar from Java to Python.

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

from abc import ABC, abstractmethod
from math import factorial, sqrt

import numpy as np

try:
    from .matrix_utils import (
        EPS,
        check_sub_stochastic_matrix,
        check_sub_stochastic_vector,
        coerce_representation,
        concat_cols,
        concat_quad,
        concat_vectors,
        factorial_to_raw,
        format_representation,
        eye_like,
        is_sparse,
        kronecker,
        kronecker_col_vector_mx,
        kronecker_mx_col_vector,
        kronecker_vectors,
        mat0,
        mat_power,
        mult_vector,
        ones_vector,
        solve_power,
        to_dense,
        vec0,
    )
except ImportError:
    from jmarkov.jmarkov.phase.discrete.matrix_utils import (
        EPS,
        check_sub_stochastic_matrix,
        check_sub_stochastic_vector,
        coerce_representation,
        concat_cols,
        concat_quad,
        concat_vectors,
        factorial_to_raw,
        format_representation,
        eye_like,
        is_sparse,
        kronecker,
        kronecker_col_vector_mx,
        kronecker_mx_col_vector,
        kronecker_vectors,
        mat0,
        mat_power,
        mult_vector,
        ones_vector,
        solve_power,
        to_dense,
        vec0,
    )


class AbstractDiscretePhaseType(ABC):
    """
    Discrete Phase-Type distribution DPH(alpha, A).

    Parameters
    ----------
    alpha : array_like
        1-D vector of initial probabilities over the transient phases. It must
        satisfy alpha >= 0 and sum(alpha) <= 1.
    A : array_like
        Sub-stochastic n x n matrix of transitions between transient phases. It
        must satisfy A >= 0, row sums <= 1, and sp(A) < 1.

    Attributes
    ----------
    n_phases : int
        Number of transient phases.

    """

    def __init__(self, alpha, A):
        self._alpha, self._A, self.n_phases = coerce_representation(alpha, A)

        if not check_sub_stochastic_vector(self._alpha):
            raise ValueError(
                "'alpha' is not a valid probability vector: "
                "alpha_i >= 0 and sum(alpha) <= 1 are required. "
                f"Got sum(alpha) = {self._alpha.sum():.6g}."
            )

        if not check_sub_stochastic_matrix(self._A):
            detail = ""
            if not is_sparse(self._A) or self.n_phases <= 1000:
                radius = np.max(np.abs(np.linalg.eigvals(to_dense(self._A))))
                detail = f" sp(A) = {radius:.6g}."
            raise ValueError(
                "'A' is not a valid transient sub-stochastic matrix. "
                "Required: A_ij >= 0, row sums <= 1, and sp(A) < 1 "
                "(this last one guarantees that absorption happens with "
                "probability 1)." + detail
            )



    @property
    def alpha(self) -> np.ndarray:
        """Vector of initial probabilities."""
        return self._alpha.copy()

    @property
    def A(self) -> np.ndarray:
        """Sub-stochastic matrix."""
        return self._A.copy()

    def get_mat0(self) -> np.ndarray:
        """
        Absorption vector a = 1 - A*1.

        a_i is the probability of being absorbed in one step from phase i.
        Java: getMat0().
        """
        return mat0(self._A, discrete=True)

    def get_vec0(self) -> float:
        """
        P(X = 0) = alpha_0 = 1 - alpha*1.

        Java: getVec0().
        """
        return vec0(self._alpha)


    def pmf(self, k: int) -> float:
        """
        Probability mass function.

            P(X = 0) = alpha_0
            P(X = k) = alpha A^{k-1} a, k >= 1

        Parameters
        ----------
        k : int
            Evaluation point, k >= 0.

        """
        k = self._check_index(k)
        if k == 0:
            return self.get_vec0()
        return mat_power(self._A, k - 1, self._alpha, self.get_mat0())

    def pmf_range(self, k_max: int) -> np.ndarray:
        """
        PMF evaluated at k = 0, 1, ..., k_max.

        Parameters
        ----------
        k_max : int
            Maximum value of k to evaluate.

        Returns
        -------
        np.ndarray
            Array of length k_max + 1.
        """
        k_max = self._check_index(k_max, name="k_max")
        a = self.get_mat0()
        res = np.zeros(k_max + 1)
        res[0] = self.get_vec0()
        v = self._alpha.copy()          # v = alpha A^{k-1}
        for k in range(1, k_max + 1):
            res[k] = v @ a
            v = v @ self._A
        return res

    def cdf(self, k: int) -> float:
        """
        Cumulative distribution function.

            F(k) = P(X <= k) = 1 - alpha A^k 1

        For k = 0 it reduces to alpha_0, consistent with pmf(0).
        """
        k = self._check_index(k)
        return 1.0 - mat_power(self._A, k, self._alpha, ones_vector(self.n_phases))

    def cdf_range(self, k_max: int) -> np.ndarray:
        """
        CDF evaluated at k = 0, 1, ..., k_max (incremental propagation).
        """
        return np.cumsum(self.pmf_range(k_max))

    def survival(self, k: int) -> float:
        """
        Survival function.

            S(k) = P(X > k) = alpha A^k 1
        """
        k = self._check_index(k)
        return mat_power(self._A, k, self._alpha, ones_vector(self.n_phases))

    def prob(self, a: int, b: int) -> float:
        """
        P(a < X <= b) = F(b) - F(a). Returns 0.0 if b <= a.
        """
        if b <= a:
            return 0.0
        return self.cdf(b) - self.cdf(a)

    # -------------------------------------------------------------------
    # Moments
    # -------------------------------------------------------------------

    def factorial_moment(self, k: int) -> float:
        """
        k-th descending factorial moment.

            E[X(X-1)...(X-k+1)] = k! alpha (I-A)^{-k} A^{k-1} 1

        This is the moment that comes out in closed form for a DPH; the raw
        moments E[X^k] are derived from these (see ``moment``).

        Java: AbstractDiscPhaseVar.moment(k)
        """
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}.")
        # A^{k-1} 1, then (I - A)^{-k} applied k times. (I-A)^{-1} and A commute, so the order does not matter.
        v = mat_power(self._A, k - 1) @ ones_vector(self.n_phases)
        I_minus_A = eye_like(self._A) - self._A
        v = solve_power(I_minus_A, k, v)
        return float(factorial(k) * (self._alpha @ v))

    def moment(self, k: int = 1) -> float:
        """
        k-th raw moment E[X^k].

        It is obtained from the factorial moments through Stirling numbers of
        the second kind:

            E[X^k] = sum_{j=1}^{k} S(k, j) E[X(X-1)...(X-j+1)]

        Note **TO BE APPROVED BY JUAN FERNANDO**: 
        AbstractDiscPhaseVar.moment(k) in Java returns the FACTORIAL moment,
        not the raw one, even though variance() and CV() there use it as
        if it were raw. For k = 1 they coincide; for k >= 2 they do not. Example
        with Geometric(0.5) - alpha = [1], A = [[0.5]]:

            E[X]        = 2      (Java and Python agree)
            E[X(X-1)]   = 4      (Java moment(2) = our factorial_moment(2))
            E[X^2]      = 6      (our moment(2))
            True Var    = 2
            Java var()  = 4 - 2^2 = 0      <-- incorrect

        """
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}.")
        fmoments = [self.factorial_moment(j) for j in range(1, k + 1)]
        return factorial_to_raw(fmoments)

    def mean(self) -> float:
        """
        Expected value E[X] = alpha (I-A)^{-1} 1.

        """
        I_minus_A = eye_like(self._A) - self._A
        v = solve_power(I_minus_A, 1)
        return float(self._alpha @ v)

    def var(self) -> float:
        """
        Variance Var(X) = E[X^2] - E[X]^2.

        Equivalently, in terms of factorial moments:
        Var(X) = E[X(X-1)] + E[X] - E[X]^2.
        """
        m1 = self.mean()
        return self.factorial_moment(2) + m1 - m1 * m1

    def std(self) -> float:
        """Standard deviation."""
        return sqrt(self.var())

    def cv(self) -> float:
        """
        Coefficient of variation CV = sd(X) / E[X].
        """
        return self.std() / self.mean()

    def scv(self) -> float:
        """
        Squared coefficient of variation, SCV = Var(X) / E[X]^2.

        Note: AbstractDiscPhaseVar.CV() in Java computes moment(2)/m^2 - 1,
        which is the form of the SCV and with
        the factorial moment instead of the raw one.
        """
        m = self.mean()
        return self.var() / (m * m)

    def quantile(self, p: float, k_max: int = 10 ** 6) -> int:
        """
        Quantile of order p: the smallest k with F(k) >= p.

        Java: quantil(double p).

        Parameters
        ----------
        p : float
            Order of the quantile, in (0, 1].
        k_max : int
            Search cap. It is only reached if p is extremely close to 1 or if
            sp(A) is almost 1.

        Returns
        -------
        int

        Changes from Java ** TO BE APPROVED BY JUAN FERNANDO **: 
        quantil in Java applies Newton-Raphson using pmf as if it were
        the derivative of cdf, that is, it treats a step function as if it
        were differentiable. After 100 iterations,it does not converge 
        and returns 0.0 instead of reporting a failure. 
        Example: With a Geometric(0.5) that happens for p = 0.9, 0.95 and 0.99, where the
        correct answers are 4, 5 and 7.

        For a discrete variable the quantile follows from the definition by
        accumulating the pmf. 
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"'p' must be in (0, 1]; got {p}.")
        accumulated = self.get_vec0()
        if accumulated >= p - EPS:
            return 0
        a = self.get_mat0()
        v = self._alpha.copy()
        for k in range(1, k_max + 1):
            accumulated += float(v @ a)
            if accumulated >= p - EPS:
                return k
            v = v @ self._A
        raise RuntimeError(
            f"quantile: p={p} was not reached within {k_max} steps. Raise "
            "'k_max' or check whether sp(A) is too close to 1."
        )

    def median(self) -> int:
        """Median, that is the quantile of order 0.5. Java: ``median()``."""
        return self.quantile(0.5)

    # -------------------------------------------------------------------
    # Closure operations
    #
    # Each of these builds a new DPH variable representing some operation on
    # independent random variables (a sum, a mixture, a minimum, ...). 
    # In Java the matrices and vectors are mutable containers so they need to 
    # be preallocated and filled in place. In Python the matrices and vectors are immutable values, 
    # so they are built in one go.
    # Only the single-result form is implemented here; 
    # _build_result plays the role of
    # newVar + setVector + setMatrix combined.
    # -------------------------------------------------------------------

    def _build_result(self, alpha, A) -> "AbstractDiscretePhaseType":
        """
        Build a variable of the same concrete class as self from a
        computed (alpha, A) pair.

        Arithmetic that mixes a dense and a sparse operand (e.g. a Dense
        variable combined with a Sparse one in min/max/sum) can
        land on either storage depending on the matrices involved. This
        forces the result back to dense when self is a Dense variable;
        the Sparse constructor already sparsifies whatever comes in, so
        nothing extra is needed on that side.
        """
        if not is_sparse(self._A):
            A = to_dense(A)
        return type(self)(alpha, A)

    def sum(self, other: "AbstractDiscretePhaseType") -> "AbstractDiscretePhaseType":
        """
        Sum of two independent variables: self + other.

        Java: sum(DiscPhaseVar B).

            vec1_0    = P(self = 0)
            beta      = other._alpha
            alpha_res = concat(alpha, beta * vec1_0)
            A_res     = [ A                  mat0(self) (x) beta ]
                        [ 0                  B                   ]

        Runs self's phase process first; once it absorbs (from phase i,
        with probability mat0(self)_i) it jumps straight into other's
        initial distribution to continue running other. If self was
        already absorbed at time 0 (vec1_0 > 0), the sum starts running
        other right away instead, weighted by that probability.

        If either variable is identically 0 (alpha entirely zero), the
        sum is just the other variable unchanged - matching Java's
        degenerate-case shortcut, which returns the same object rather than a
        copy.
        """
        if not np.any(self._alpha):
            return other
        if not np.any(other._alpha):
            return self

        n1, n2 = self.n_phases, other.n_phases
        a0 = self.get_mat0()
        vec1_0 = self.get_vec0()

        alpha_res = concat_vectors(self._alpha, other._alpha * vec1_0)
        right_up = mult_vector(a0, other._alpha)
        A_res = concat_quad(self._A, right_up, np.zeros((n2, n1)), other._A)
        return self._build_result(alpha_res, A_res)

    def sum_geom(self, p: float) -> "AbstractDiscretePhaseType":
        """
        Sum of a Geometric(p)-distributed number of iid copies of this
        variable (the geometric variable is supported on {1, 2, 3, ...}).

        Java: sumGeom(double p).

        With N ~ Geometric(p), P(N = n) = p (1-p)^{n-1}, and c = 1 / (1 -
        (1-p) alpha_0):

            alpha_res = c * alpha
            A_res     = A + (1-p) * c * mat0(self) (x) alpha

        Each time a copy absorbs, another copy starts with probability (1-p).
        A new copy may itself be 0 (probability alpha_0), in which case it
        is over immediately and the (1-p) decision is taken again. The factor
        c = sum_{m>=0} ((1-p) alpha_0)^m adds up those chains of zero-length
        copies. It is the 1-phase case of sum_ph: a Geometric(p) counter
        is DPH(beta=[1], S=[[1-p]]), and then M = (1 - alpha_0 (1-p))^{-1} = c.

        Change from Java: 

        Java returns alpha_res = alpha and A_res = A + (1-p) a alpha,
        without the factor c. That is only exact when alpha_0 = 0. With
        alpha_0 > 0 it assigns no probability to the chains of zero-length
        copies. Example: X with alpha = [0.5, 0.2] (alpha_0 = 0.3) and p =
        0.3 differs from the brute-force compound sum by 0.19 in the pmf. 
        
        The continuous AbstractContPhaseVar.sumGeom uses the same formula, and
        its reference test (DenseContClosureTest.testSumGeom, with alpha_0
        = 0.5) was computed with that formula too.
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"'p' must be in (0, 1]; got {p}.")
        a = self.get_mat0()
        c = 1.0 / (1.0 - (1.0 - p) * self.get_vec0())
        A_res = self._A + (1.0 - p) * c * mult_vector(a, self._alpha)
        return self._build_result(c * self._alpha, A_res)

    def sum_ph(self, counter: "AbstractDiscretePhaseType") -> "AbstractDiscretePhaseType":
        """
        Sum of a Phase-type-distributed number of iid copies of this
        variable.

        Java: sumPH(DiscPhaseVar B).

        counter is itself a DPH variable (beta, S) whose VALUE is the
        number of copies of this to add: Y = X_1 + ... + X_N, with the
        X_i iid copies of this variable and N ~ counter. This generalizes
        sum_geom, which is the special case where counter is a
        1-phase geometric count.

        Construction 
        n1 = self.n_phases, 
        n2 = counter.n_phases
        a0 = P(self = 0)
        a = self's absorption vector get_mat0()
        beta
        S = counter's vector/matrix):

            M         = (I_n2 - a0*S)^{-1}
            alpha_res = alpha (x) (M^T beta)
            A_res     = (A (x) I_n2) + (a (x) alpha) (x) (M S)

        The state is (phase of the current copy, phase of the counter). While
        a copy runs, only its phase moves (A (x) I). When it absorbs (a), the
        counter takes one step (S). The next copy either starts in a phase
        (alpha) or is 0 (a0); a 0 copy makes the counter step again at once.
        M = sum_{m>=0} (a0 S)^m collects those chains of zero-length copies,
        both at the start (alpha_res) and after each absorption (M S = S M).

        Changes from Java
        ---------------------
        1. Java computes M with
           ISInv.solve(Matrices.identity(this.getNumPhases()), ...). It
           solves against I_n1 even though ISInv is n2 x n2, so the
           call only works when n1 == n2 and raises an exception otherwise.
           The continuous version, AbstractContPhaseVar.sumPH, uses
           Matrices.identity(n2), as it should.
        2. Java multiplies the second term by (1 - a0). The formula above
           has no such factor: Java's would be correct if alpha were the
           conditional vector alpha / (1 - a0) (the start of a copy GIVEN that
           it is not 0), but getVector() is the unconditional alpha, so
           the factor counts (1 - a0) twice. 
           With a0 = 0 the two versions coincide. With a0 > 0, 
           the factor is applied twice, leading to a loss of probability mass.
           For example, alpha = [0.5, 0.2] (a0 = 0.3) against a 2-phase counter
           differs from the brute-force compound sum by 0.016 in the pmf.

        No discrete test exists on the Java side, so neither problem ever
        showed up there.

        """
        n2 = counter.n_phases
        a0 = self.get_vec0()
        a = self.get_mat0()
        beta = counter._alpha
        S = to_dense(counter._A)

        M = np.linalg.inv(np.eye(n2) - a0 * S)

        alpha_res = kronecker_vectors(self._alpha, M.T @ beta)
        L1 = kronecker(self._A, eye_like(self._A, n2))
        L2 = kronecker(mult_vector(a, self._alpha), M @ S)
        A_res = L1 + L2
        return self._build_result(alpha_res, A_res)

    def mix(self, p: float, other: "AbstractDiscretePhaseType") -> "AbstractDiscretePhaseType":
        """
        Mixture: with probability p realize self, otherwise other.

        Java: mix(double p, DiscPhaseVar B).

            alpha_res = concat(p*alpha, (1-p)*beta)
            A_res     = block_diag(A, B)
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"'p' must be in [0, 1]; got {p}.")
        n1, n2 = self.n_phases, other.n_phases
        alpha_res = concat_vectors(p * self._alpha, (1.0 - p) * other._alpha)
        A_res = concat_quad(self._A, np.zeros((n1, n2)), np.zeros((n2, n1)), other._A)
        return self._build_result(alpha_res, A_res)

    def min(self, other: "AbstractDiscretePhaseType") -> "AbstractDiscretePhaseType":
        """
        Minimum of two independent variables: min(self, other).

        Java: min(DiscPhaseVar B).

            alpha_res = alpha (x) beta
            A_res     = A (x) B

        Runs both phase processes in parallel; the combined process absorbs
        as soon as either one does
        """
        alpha_res = kronecker_vectors(self._alpha, other._alpha)
        A_res = kronecker(self._A, other._A)
        return self._build_result(alpha_res, A_res)

    def max(self, other: "AbstractDiscretePhaseType") -> "AbstractDiscretePhaseType":
        """
        Maximum of two independent variables: max(self, other).

        Java: max(DiscPhaseVar B).

        Layout of the result (n1*n2 + n1 + n2 phases, n1 = self.n_phases,
        n2 = other.n_phases):

          - the first n1*n2 phases track (self, other) running in parallel,
            while both are still active;
          - the next n1 phases track self alone, once other has
            already absorbed;
          - the last n2 phases track other alone, once self has
            already absorbed.

        The combined process only absorbs once BOTH have absorbed

        Changes from Java ** TO BE APPROVED BY JUAN FERNANDO **:
        --------------------
        Java's discrete max is a line-for-line copy of
        AbstractContPhaseVar.max, including the joint "both still
        running" block, built with kroneckerSum(A, B) = A(x)I + I(x)B,
        and the two boundary-crossing blocks, built with
        kronecker(I_n1, mat0(B)) / kronecker(mat0(A), I_n2).

        That is the right construction in CONTINUOUS time: with probability
        one, at most one of the two subprocesses moves at any given instant,
        so rates simply add (hence the Kronecker SUM) and, when one
        subprocess crosses into absorption, the other's phase is unchanged
        during that same instantaneous event (hence the IDENTITY next to
        each ``mat0``).

        In DISCRETE time neither assumption holds: every phase transition
        step advances both subprocesses simultaneously, and both may even
        absorb in the very same step. Reusing the continuous formula
        produces invalid rows (row sums greater than 1) as soon as both
        variables have a positive chance of continuing
        confirmed numerically for two Geometric(0.5)/Geometric(0.7) variables, where
        the joint self-loop alone came out to 0.8 while transitions to the
        two boundary blocks added another 0.7 + 0.5, all from the same
        state. 
        The correct discrete-time construction replaces the Kronecker sum with the Kronecker 
        PRODUCT for the joint block and replaces each boundary identity with the surviving variable's
        own sub-stochastic matrix, since it also takes a real transition in
        the step where the other one absorbs:

            joint block    = A (x) B                  (was A(+)B)
            self-only edge = A (x) mat0(other)         (was I_n1 (x) mat0(other))
            other-only edge= mat0(self) (x) B          (was mat0(self) (x) I_n2)

        With this fix the row sums work out to
        1 - mat0(self)_i * mat0(other)_j for every joint row - i.e. the
        only probability mass missing from a joint row is exactly the
        probability that both variables absorb in that same step, which is
        precisely when max itself absorbs. 
        No test exercises this method on the Java side either, so the bug is latent there too.
        """
        n1, n2 = self.n_phases, other.n_phases
        a1, a2 = self._alpha, other._alpha
        a1_0, a2_0 = self.get_vec0(), other.get_vec0()
        m1, m2 = self.get_mat0(), other.get_mat0()

        alpha_both = kronecker_vectors(a1, a2)
        alpha_self_only = a1 * a2_0
        alpha_other_only = a2 * a1_0
        alpha_res = concat_vectors(
            alpha_both, concat_vectors(alpha_self_only, alpha_other_only)
        )

        joint_block = kronecker(self._A, other._A)
        boundary = concat_quad(
            self._A, np.zeros((n1, n2)), np.zeros((n2, n1)), other._A
        )
        transition_to_boundary = concat_cols(
            kronecker_mx_col_vector(self._A, m2),
            kronecker_col_vector_mx(m1, other._A),
        )
        A_res = concat_quad(
            joint_block,
            transition_to_boundary,
            np.zeros((n1 + n2, n1 * n2)),
            boundary,
        )
        return self._build_result(alpha_res, A_res)

    # -------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------

    def label(self) -> str:
        """Short label."""
        return f"DPH - {self.n_phases} phases"

    def description(self) -> str:
        """Full description of the representation and its statistics."""
        return format_representation(
            title="Discrete Phase-Type distribution",
            alpha=self._alpha,
            A=self._A,
            matrix_name="A",
            stats={"Mean": self.mean(), "Variance": self.var(), "CV": self.cv()},
        )

    def __str__(self) -> str:
        return self.label()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_phases={self.n_phases})"

    def __eq__(self, other) -> bool:
        """
        Equality of representations (not of distributions).

        Two different representations can generate the same distribution; this
        method compares the parameters, not the probability law.
        """
        if not isinstance(other, AbstractDiscretePhaseType):
            return NotImplemented
        if self.n_phases != other.n_phases:
            return False
        if not np.allclose(self._alpha, other._alpha, atol=EPS):
            return False
        if is_sparse(self._A) or is_sparse(other._A):
            # Sparse structures are not compared entry by entry: two
            # representations with a different format or a different pattern of
            # explicit zeros can still be the same matrix. What is compared is
            # the difference, which stays sparse.
            diff = self._A - other._A
            largest = abs(diff).max() if getattr(diff, "nnz", 1) else 0.0
            return bool(largest <= EPS)
        return bool(np.allclose(self._A, other._A, atol=EPS))

    def __hash__(self):
        A = self._A
        fingerprint = (A.data.tobytes(), A.indices.tobytes()) if is_sparse(A) \
            else (A.tobytes(),)
        return hash((type(self).__name__, self.n_phases,
                     self._alpha.tobytes()) + fingerprint)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _check_index(k, name: str = "k") -> int:
        """Validate that k is a non-negative integer."""
        if isinstance(k, (bool, np.bool_)):
            raise TypeError(f"'{name}' must be an integer, not a bool.")
        if isinstance(k, (float, np.floating)):
            if abs(k - round(k)) > EPS:
                raise ValueError(
                    f"'{name}' must be an integer; got {k}. The DPH is only "
                    "defined over the non-negative integers."
                )
            k = int(round(k))
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"'{name}' must be an integer; got {type(k).__name__}.")
        if k < 0:
            raise ValueError(f"'{name}' must be >= 0; got {k}.")
        return int(k)

    # -------------------------------------------------------------------
    # Abstract methods
    # -------------------------------------------------------------------

    @abstractmethod
    def copy(self) -> "AbstractDiscretePhaseType":
        """Return an independent copy of this variable."""

    @abstractmethod
    def new_var(self, n: int) -> "AbstractDiscretePhaseType":
        """
        Build an empty variable of the same concrete type with n phases.

        It is used by the closure operations (sum, mixture, minimum, maximum)
        to create the container for the result without knowing the subclass.
        """
