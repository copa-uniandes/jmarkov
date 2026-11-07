"""
DenseDiscPhaseVar.py

Discrete Phase-Type distribution with a dense matrix representation.

Migration of jphase.DenseDiscPhaseVar Java to Python.

Every numerical method (pmf, cdf, moments, quantiles) is in
AbstractDiscretePhaseType. This class only supplies the two operations that
depend on how the matrix is stored: copy and new_var. That is the same
split Java draws between AbstractDiscPhaseVar and DenseDiscPhaseVar.

Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

import numpy as np

# Written twice on purpose: the relative form resolves inside the jmarkov
# package, the absolute form resolves in a flat folder with no __init__.py.
try:
    from .discrete.AbstractDiscPhaseVar import AbstractDiscretePhaseType
except ImportError:  # pragma: no cover - depends on the layout, not on logic
    from jmarkov.jmarkov.phase.discrete.AbstractDiscPhaseVar import AbstractDiscretePhaseType


class DenseDiscretePhaseType(AbstractDiscretePhaseType):
    """
    DPH(alpha, A) with A stored as a dense np.ndarray.

    Parameters
    ----------
    alpha : array_like
        1-D vector of initial probabilities over the transient phases. It must
        satisfy alpha >= 0 and sum(alpha) <= 1.
    A : array_like
        Sub-stochastic n x n matrix. It must satisfy A >= 0, row sums <= 1 and
        sp(A) < 1.

    """

    def copy(self) -> "DenseDiscretePhaseType":
        """
        Independent copy of this variable. Java: copy().

        The arrays are duplicated, so mutating the copy does not affect the
        original.
        """
        return DenseDiscretePhaseType(self._alpha.copy(), self._A.copy())

    def new_var(self, n: int) -> "DenseDiscretePhaseType":
        """
        Empty variable of the same type with n phases. Java: newVar(int n).

        It returns alpha and A filled with zeros, which is a valid if degenerate
        DPH: with alpha = 0 we get alpha_0 = 1, that is P(X = 0) = 1.

        The closure operations (sum, mixture, minimum, maximum) use it to create
        the container for the result without knowing the concrete subclass, and
        then fill it with the corresponding blocks.

        Raises
        ------
        ValueError
            If n < 1.
        """
        if n < 1:
            raise ValueError(f"'n' must be >= 1; got {n}.")
        return DenseDiscretePhaseType(np.zeros(n), np.zeros((n, n)))
