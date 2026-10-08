"""
jmarkov/phase/matrix_utils.py

Matrix utilities for the Phase-Type package, shared by the continuous (ctph)
and discrete (dtph) cases.

--------------------------------

Dense vs. sparse

So that the functions that process matrices can be used in the dense and sparse cases, 
we decided to implement a is_sparse, so that the functions can use the appropiate procedure based on the type of the matrix. 
The functions that take matrices accept both dense np.ndarray and scipy.sparse matrices, and dispatch internally. 

----------------
Notes:

There were a few functions with the same name, but different purposes. In Python, with NumPy and SciPy, 
we can unify them into a single function with optional arguments. 

VECTORS (alpha, the output vector) are always densified.


Operation correspondence
------------------------
    dense                            sparse
    ------------------------------   ----------------------------------
    A @ np.ones(n)                   A @ np.ones(n)          (same)
    A.sum(axis=1)                    A.sum(axis=1)           (same)
    np.eye(n)                        eye_like(A)   <- NOT A ** 0
    np.linalg.matrix_power(A, k)     explicit multiplication
    np.linalg.solve(M, v)            splu(M).solve(v), factorizing once
    alpha @ A^k @ v (matrix power)   k vector-matrix products (no fill-in)
    sp(A) < 1 check                  graph reachability, same for both
    np.kron(A, B)                    scipy.sparse.kron(A, B)
    np.block([[...]])                scipy.sparse.bmat([[...]])
    scipy.linalg.expm(A)             scipy.sparse.linalg.expm(A)


Changes done from the original Java version **TO BE APPROVED BY JUAN FERNANDO** :
There were some functions that we thought needed some changes because they had some errors:
    1. The function "sumMatPower" was not saving the correct result of the intermediate matrix multiplication, 
    so it was returning k * I instead of the sum of powers. We wrote the correct version below. 
    2. The function "kroneckerMxRowVector" was calculating the output column using the number of columns of A 
    instead of the size of b, so it was leaving some columns unfilled. We wrote the correct version below.
    3. The function "checkSubGeneratorMatrix" is supposed to validate whether a matrix is a sub-generator, 
    but the functions starts checking at j=1 so the first column is not checked. We wrote the correct version below.
    4. The function "checkSubStochasticVector" was not checking whether the first entry of the vector was non-negative.
    We wrote the correct version below.


Changes done on purpose ** TO BE APPROVED BY JUAN FERNANDO** : 
    1. In the function "sumMatPower", when k < 1, the Java version prints a message and returns 0, 
    while the Python version raises a ValueError.
    2. In the function "CV(double[])", the Java version returns Var/mean^2, 
    which is the SCV and not the CV. We checked in which parts in Java we used this function and we found that
    it is used in the fitting algorithms EMHyperErlangFit.doFitHyperErlang(), where it is compared against PhaseVar.CV().
    This method returns moment(2)/mean^2 - 1, which is the same SCV and the printing lines found say SCV(data) and SCV(variable)
    Meaning that maybe we only need to change the name to SCV (We kept it the same as the Java version) and we added a CV_true just in case
    we do want to use and calculate the CV.


Authors: Juanita Carrascal Mendez, Luciana Bacci Tarazona
Advisor: Juan Fernando Perez Bernal
Version: 1.0
"""

from math import comb, factorial, sqrt
from typing import Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla

# Numerical tolerance (Epsilon in Java)
EPS = 1.0e-10


# =============================================================================
# PART 1 - Functions directly from matrix_utils.java
# =============================================================================

def ones_vector(m: int) -> np.ndarray:
    """
    One-dimensional array of length m filled with ones - Java: OnesVector(int m).

    In Java there were 2 functions with the same name, 
    we decided to only write the first one since in Python, 
    vectors are always dense ndarrays.
    """
    return np.ones(m, dtype=float)


def ones_row(m: int) -> np.ndarray:
    """
    Row matrix (1, m) of ones - Java: OnesRow(int m).
    """
    return np.ones((1, m), dtype=float)


def ones_col(m: int) -> np.ndarray:
    """
    Column matrix (m, 1) of ones - Java: OnesCol(int m)
    """
    return np.ones((m, 1), dtype=float)

"""
Java has four kronecker functions because it distinguishes the Matrix
and Vector types, and because a Vector can enter either as a column or as a
row. In NumPy everything is an ndarray, so we decided to divide it with the name of the functions:

    kronecker(Matrix A, Matrix B, res)       -> kronecker(A, B)
    kronecker(Matrix A, Vector B, res)       -> kronecker_mx_col_vector(A, b)
    kronecker(Vector A, Matrix B, res)       -> kronecker_col_vector_mx(a, B)
    kroneckerMxRowVector(Matrix A, Vector B) -> kronecker_mx_row_vector(A, b)
"""
def kronecker(A, B) -> np.ndarray:
    """
    Kronecker product A (x) B - Java: kronecker(Matrix A, Matrix B).

    Used by the PH closure operations (min, max)
    """
    if is_sparse(A) or is_sparse(B):
        return sparse.kron(A, B, format="csr")
    return np.kron(np.asarray(A, dtype=float), np.asarray(B, dtype=float))


def kronecker_mx_col_vector(A, b) -> np.ndarray:
    """
    Kronecker product A (x) b, with b treated as a COLUMN vector.

    Java: kronecker(Matrix A, Vector B, Matrix res), whose output matrix has
    size (A.numRows * b.size, A.numColumns).
    """
    b = as_vector(b, "b").reshape(-1, 1)
    if is_sparse(A):
        return sparse.kron(A, b, format="csr")
    return np.kron(np.asarray(A, dtype=float), b)


def kronecker_col_vector_mx(a, B) -> np.ndarray:
    """
    a (x) B, with a treated as a COLUMN vector.

    Java: kronecker(Vector A, Matrix B, Matrix res), whose output matrix has
    size (a.size * B.numRows, B.numColumns).
    """
    a = as_vector(a, "a").reshape(-1, 1)
    if is_sparse(B):
        return sparse.kron(a, B, format="csr")
    return np.kron(a, np.asarray(B, dtype=float))


def kronecker_mx_row_vector(A, b) -> np.ndarray:
    """
    A (x) b, with b treated as a ROW vector.

    Java: kroneckerMxRowVector(Matrix A, Vector B, Matrix res), with output
    of size (A.numRows, A.numColumns * b.size).

    Changes from the Java version
    --------------------
    The Java version writes into column j1 * c1 + i2, where c1 is the
    number of columns of A. 
    The correct stride is j1 * n2 + i2, with n2 = b.size. The two agree only when A has as many columns as b has
    entries. With A of size 2x3 and b of size 2, for example, Java writes into
    columns {0,1,3,4,6,7} of a 2x6 matrix: it runs out of range and leaves
    columns 2 and 5 unfilled.

    Nothing else in jphase calls this method, so the bug never showed up.
    """
    b = as_vector(b, "b").reshape(1, -1)
    if is_sparse(A):
        return sparse.kron(A, b, format="csr")
    return np.kron(np.asarray(A, dtype=float), b)


def kronecker_sum(A, B) -> np.ndarray:
    """
    Kronecker sum A (+) B = A (x) I_nb + I_na (x) B.

    Java: kroneckerSum(Matrix A, Matrix B).

    It returns the kronecker sum of two matrices, which is used in the closure 
    operations (min, max) to build the generator of the resulting variable.
    """
    return (kronecker(A, eye_like(B))
            + kronecker(eye_like(A), B))


def kronecker_vectors(a, b) -> np.ndarray:
    """
    Kronecker product of two vectors - Java: kroneckerVectors
    """
    return np.kron(as_vector(a, "a"), as_vector(b, "b"))


def concat_rows(A, B) -> np.ndarray:
    """
    Concatenates A and B vertically keeping the same number of columns - Java: concatRows.
    """
    if is_sparse(A) or is_sparse(B):
        return sparse.vstack([A, B], format="csr")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"concat_rows: incompatible columns {A.shape[1]} != {B.shape[1]}."
        )
    return np.vstack((A, B))


def concat_cols(A, B) -> np.ndarray:
    """
    Concatenates A and B horizontally keeping the same number of rows - Java: concatCols.
    """
    if is_sparse(A) or is_sparse(B):
        return sparse.hstack([A, B], format="csr")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"concat_cols: incompatible rows {A.shape[0]} != {B.shape[0]}."
        )
    return np.hstack((A, B))


def concat_quad(left_up, right_up, left_down, right_down) -> np.ndarray:
    """
    Concatenates the columns of the left and right upper matrices
    and this result is then concatenated by the rows with the concatenation of the left and right lower matrices.

    Java: concatQuad(leftUp, rightUp, leftDown, rightDown, res).
    """
    blocks = (left_up, right_up, left_down, right_down)
    if any(is_sparse(b) for b in blocks):
        return sparse.bmat([[left_up, right_up], [left_down, right_down]],
                           format="csr")
    lu = np.atleast_2d(np.asarray(left_up, dtype=float))
    ru = np.atleast_2d(np.asarray(right_up, dtype=float))
    ld = np.atleast_2d(np.asarray(left_down, dtype=float))
    rd = np.atleast_2d(np.asarray(right_down, dtype=float))
    if lu.shape[0] != ru.shape[0] or ld.shape[0] != rd.shape[0]:
        raise ValueError("concat_quad: incompatible rows between blocks.")
    if lu.shape[1] != ld.shape[1] or ru.shape[1] != rd.shape[1]:
        raise ValueError("concat_quad: incompatible columns between blocks.")
    return np.block([[lu, ru], [ld, rd]])


def concat_vectors(a, b) -> np.ndarray:
    """
    Concatenate two 1-D vectors. Java: concatVectors.
    """
    return np.concatenate((np.asarray(a, dtype=float).ravel(),
                           np.asarray(b, dtype=float).ravel()))


def mult_vector(a, b) -> np.ndarray:
    """
    Outer product a * b^T - Java: multVector(Vector A, Vector B, Matrix res).

    It appears in the closure operations when connecting the phases of one
    variable to those of the next: mult_vector(a1, alpha2).
    """
    return np.outer(as_vector(a, "a"), as_vector(b, "b"))


def mat_power(A, k: int, left_vec=None, right_vec=None):
    """
    Matrix power A^k, optionally pre/post-multiplied by vectors.

    Java: covers both matPower overloads:
      - matPower(Matrix A, int k)                        -> A^k
      - matPower(Matrix A, int k, Vector l, Vector r)    -> l * A^k * r

    Parameters
    ----------
    A : array_like
        Square base matrix.
    k : int
        Exponent, k >= 0. A^0 = I.
    left_vec, right_vec : array_like, optional
        If both are given, returns left_vec @ A^k @ right_vec.

        On sparse matrices, the scalar is computed by multiplying the vector by A
        repeatedly instead of explicitly forming A^k. This saves memory because powers
        of sparse matrices can quickly become dense. The cost is approximately
        O(k * nnz(A)).

    Returns
    -------
    np.ndarray or float
    """
    if k < 0:
        raise ValueError(f"mat_power: k must be >= 0; got {k}.")
    if (left_vec is None) != (right_vec is None):
        raise ValueError(
            "mat_power: 'left_vec' and 'right_vec' must be given together, or "
            "neither of them."
        )
    if is_sparse(A) and left_vec is not None:
        v = as_vector(left_vec, "left_vec")
        for _ in range(k):
            v = A.T @ v
        return float(v @ as_vector(right_vec, "right_vec"))
    if is_sparse(A):
        # A ** k is NOT used because the operator changes meaning between the two
        # scipy APIs. On spmatrix (csr_matrix) it is MATRIX power; on sparray
        # (csr_array) it is ELEMENT-WISE power, just as on ndarray. With
        # csr_array, A ** 2 returns the squared entries, not A@A, so we multiply explicitly.
        Ak = eye_like(A)
        for _ in range(k):
            Ak = Ak @ A
    else:
        A = np.asarray(A, dtype=float)
        Ak = np.linalg.matrix_power(A, k)
    if left_vec is None:
        return Ak
    left = as_vector(left_vec, "left_vec")
    right = as_vector(right_vec, "right_vec")
    return float(left @ (Ak @ right))


def sum_mat_power(A, k: int, left_vec=None, right_vec=None):
    """
    Partial sum of powers: S = sum_{j=1}^{k} A^{j-1} = I + A + ... + A^{k-1}.

    Java: sumMatPower(Matrix A, int k, Vector leftVec, Vector rightVec).

    If the vectors are given, returns the scalar left_vec @ S @ right_vec.

    Note:
    It shows up when accumulating the discrete CDF and in the loss functions.
    When sp(A) < 1 and k is large, the limit is (I - A)^{-1}; in those cases it
    is better to solve the system than to sum (see solve_power).

    Changes from Java version
    --------------------
    The Java version does not accumulate the powers. Its loop reads:

        Matrix temp = Matrices.identity(n);
        Matrix sum  = temp.copy();
        for (int i = 1; i < k; i++) {
            temp.mult(A, temp.copy());   // <-- result discarded
            sum.add(temp);
        }

    In MTJ, X.mult(B, C) computes C = X*B and returns C, leaving X
    untouched. Since the return value is never assigned, temp stays equal
    to the identity on every pass and sum ends up being k*I. In other
    words, Java returns k * (leftVec . rightVec) instead of the sum of
    powers. 

    Nothing else in jphase calls this method, so the bug never showed up.

    """
    if k < 1:
        raise ValueError(f"sum_mat_power: k must be >= 1; got {k}.")
    total = eye_like(A)
    term = eye_like(A)
    for _ in range(1, k):
        term = term @ A
        total = total + term
    if left_vec is None and right_vec is None:
        return total
    if (left_vec is None) != (right_vec is None):
        raise ValueError(
            "sum_mat_power: 'left_vec' and 'right_vec' must be given together, "
            "or neither of them."
        )
    left = as_vector(left_vec, "left_vec")
    right = as_vector(right_vec, "right_vec")
    return float(left @ (total @ right))


def distance(v1, v2) -> float:
    """
    Maximum relative distance between two arrays - Java: distance.

    For each entry it computes (v1_i - v2_i) / v1_i when v1_i > 0, and the
    absolute difference otherwise; it returns the maximum in absolute value.

    Note: 

    In Java the function says it calculates de maximum euclidean distance, 
    but it actually calculates the maximum relative distance. We used the function 
    as it is but we added a note to clarify this.

    """
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()
    if v1.shape != v2.shape:
        return -1.0
    delta = v1 - v2
    positive = v1 > 0
    rel = delta.copy()
    np.divide(delta, v1, out=rel, where=positive)
    return float(np.max(np.abs(rel)))


def scalar(A) -> float:
    """
    Extract the single entry of a one-column matrix. Java: scalar.
    """
    A = np.asarray(A, dtype=float)
    A2 = np.atleast_2d(A)
    if A2.shape[1] != 1:
        raise ValueError(
            f"scalar: expected a matrix with 1 column; got shape {A2.shape}."
        )
    return float(A2[0, 0])


def average(data) -> float:
    """
    Sample mean. Java: average(double[] datos)."""
    return float(np.mean(np.asarray(data, dtype=float)))


def average2(data) -> float:
    """
    Second (non-central) moment. 
    Java: average2(double[] data).
    """
    arr = np.asarray(data, dtype=float)
    return float(np.mean(arr * arr))


def variance(data) -> float:
    """
    Variance (divides by n). Java: variance(double[] data).
    """
    m = average(data)
    return average2(data) - m * m


def cv(data) -> float:
    """
    Coefficient of variation (squared). Java: CV(double[] data), which returns variance / mean^2.

    Note:
    That is the SQUARED coefficient of variation (SCV), not the CV. The CV is
    sqrt(variance) / mean. Java's name and behaviour are kept for
    traceability; for the true CV use cv_true.
    """
    m = average(data)
    return variance(data) / (m * m)


def cv_true(data) -> float:
    """
    True coefficient of variation, sd/mean. 
    Does not exist in Java.
    """
    return sqrt(variance(data)) / average(data)


def check_sub_stochastic_vector(a, tol: float = EPS) -> bool:
    """
    Check that a is a sub-stochastic vector: a_i >= 0 and sum(a) <= 1.

    Java: checkSubStochasticVector(Vector a).

    Changes from the Java version
    --------------------------------
    Java's sign loop starts at i = 1, so a[0] is added to the sum but
    never checked for being negative: a = [-5, 0.5] passes there (sum
    -4.5 <= 1) and is rejected here.
    """
    a = np.asarray(a, dtype=float).ravel()
    if np.any(a < -tol):
        return False
    return bool(a.sum() <= 1.0 + tol)


def check_sub_generator_matrix(A, tol: float = EPS) -> bool:
    """
    Check that A is a sub-generator (CONTINUOUS case).

    Java: checkSubGeneratorMatrix(Matrix A).

    Conditions:
      1. A_ij >= 0 for i != j
      2. A_ii <  0
      3. sum_j A_ij <= 0 for every row

    Changes from the Java version
    ---------------------------------
    The Java version has two gaps that are closed here:

    a) Its inner loop starts at j = 1, so column 0 is never inspected:
       neither A[0][0] < 0 nor A[i][0] >= 0 is verified.
    b) It starts from res = true and only runs the checks when the matrix
       is square, so a non-square matrix passes. Here it is rejected.

    """
    if is_sparse(A):
        if A.shape[0] != A.shape[1]:
            return False
        diag = A.diagonal()
        if np.any(diag >= -tol):
            return False
        off = A - (sparse.diags_array(diag) if _is_sparray(A)
                   else sparse.diags(diag))
        if off.nnz and off.min() < -tol:
            return False
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        if np.any(row_sums > tol):
            return False
        return True

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    off_diag = A[~np.eye(n, dtype=bool)]
    if off_diag.size and np.any(off_diag < -tol):
        return False
    if np.any(np.diag(A) >= -tol):
        return False
    row_sums = A.sum(axis=1)
    if np.any(row_sums > tol):
        return False
    return True


# =============================================================================
# PART 2 - Functions from the abstract classes of jphase
#
# In Java these two live as instance methods on AbstractContPhaseVar and
# AbstractDiscPhaseVar (getVec0 / getMat0). Since their definition is identical
# in both hierarchies apart from the sign, we decided to unify them in this file.
# =============================================================================

def vec0(alpha) -> float:
    """
    Initial probability mass on the absorbing state:

        alpha_0 = 1 - alpha * 1

    Java: getVec0() (identical in the continuous and the discrete
    hierarchy).

    Continuous: produces an atom at t = 0.  
    Discrete: it is P(X = 0).
    """
    return 1.0 - float(np.asarray(alpha, dtype=float).sum())


def mat0(A, discrete: bool) -> np.ndarray:
    """
    Absorption vector from each phase.

        Continuous:  a = -A * 1        (absorption rates)
        Discrete:    a =  1 - A * 1    (probability of absorbing in one step)

    Java: getMat0() in AbstractContPhaseVar and in
    AbstractDiscPhaseVar respectively.
    Both versions are unified through the discrete parameter.
    """
    ones = np.ones(A.shape[0])
    outflow = np.asarray(A @ ones, dtype=float).ravel()
    if discrete:
        return ones - outflow
    return -outflow


# =============================================================================
# PART 3 - Additions that do NOT exist in MatrixUtils.java
#
# In this section, every function is new and does not exist in the Java version. They are
# added to provide additional functionality that is not available in the original Java implementation.
# =============================================================================

def is_sparse(M) -> bool:
    """
    Is `M` a SciPy sparse matrix?

    Covers both scipy.sparse APIs: the old one based on spmatrix
    (csr_matrix) and the new one based on sparray (csr_array).
    """
    return bool(sparse.issparse(M))


def _is_sparray(M) -> bool:
    """
    Does M belong to the new API (sparray) rather than the old (spmatrix)?
    """
    kind = getattr(sparse, "sparray", None)
    return kind is not None and isinstance(M, kind)


def eye_like(A, n: int = None):
    """
    Identity of the same type and format as A.

    We created it because if we used A ** 0, it would raise NotImplementedError in scipy: 
    raising a sparse matrix to the zeroth power would produce a dense one
    Here the sparse identity is built explicitly.
    """
    n = A.shape[0] if n is None else n
    if not is_sparse(A):
        return np.eye(n)
    if _is_sparray(A):
        return sparse.eye_array(n, format="csr")
    return sparse.identity(n, format="csr")


def to_dense(M) -> np.ndarray:
    """
    Densify M if it is sparse; if it is already dense, return it as is.
    """
    if is_sparse(M):
        return np.asarray(M.todense(), dtype=float)
    return np.asarray(M, dtype=float)


def as_vector(v, name: str = "alpha") -> np.ndarray:
    """
    Convert v into a 1-D float64 array.

    It accepts a 2-D input with a single row or column and flattens it, because
    writing np.array([[0.7, 0.3]]) instead of 
    np.array([0.7, 0.3]) is a
    common mistake since both are ndarrays.

    It rejects SciPy sparse matrices with an explicit message rather than
    letting np.asarray raise an unhelpful ValueError.
    """
    if is_sparse(v):
        # We decided that vectors are always densified. That is O(n) against the O(n^2) it
        # costs to densify the matrix.
        v = v.todense()
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1-D; got shape {arr.shape}.")
    return arr


def as_square_matrix(M, name: str = "A") -> np.ndarray:
    """
    Convert M into a square 2-D float64 array.

    It rejects SciPy sparse matrices with an explicit message.
    """
    if is_sparse(M):
        if M.shape[0] != M.shape[1]:
            raise ValueError(
                f"'{name}' must be square; got {M.shape[0]}x{M.shape[1]}."
            )
        return M.tocsr() if hasattr(M, "tocsr") else M
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"'{name}' must be 2-D; got {arr.ndim} dimension(s)."
        )
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"'{name}' must be square; got {arr.shape[0]}x{arr.shape[1]}."
        )
    return arr


def coerce_representation(alpha, A) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Convert and check the dimensional compatibility of (alpha, A).

    The returned arrays are always copies. np.asarray and .tocsr()
    return the SAME object when the input already has the right type, so
    without the copy a variable would share memory with the caller's arrays
    and change whenever the caller modified them. 
    
    In Java, the constructor of DenseMatrix copies the input array, so the
    Python version mimics that behavior with the copy() calls.

    Returns
    -------
    (alpha, A, n_phases)
    """
    A_arr = as_square_matrix(A, "A").copy()
    a_arr = as_vector(alpha, "alpha").copy()
    n = A_arr.shape[0]
    if a_arr.shape[0] != n:
        raise ValueError(
            f"'alpha' must have length {n} to match A; "
            f"got length {a_arr.shape[0]}."
        )
    return a_arr, A_arr, n


def _spectral_radius_lt_one(A, tol: float = EPS) -> bool:
    """
    Check whether every phase can eventually reach absorption.

    For a non-negative matrix with row sums at most 1, a row whose sum is
    less than 1 represents a phase with a positive probability of being
    absorbed in the next step. The function checks whether every other phase
    can reach one of these "leaking" phases through a positive transition.

    If this is true, the matrix is transient and its spectral radius is less
    than 1. If a group of phases cannot reach a leaking phase, that group can
    keep all its probability mass forever, so the spectral radius is 1.

    The check uses a breadth-first search on the transition graph instead of
    computing eigenvalues. This is more efficient for large sparse matrices.

    The transience criterion follows Latouche and Ramaswami [2].
    """
    S = sparse.csr_array(A) if is_sparse(A) else sparse.csr_array(
        np.asarray(A, dtype=float))
    n = S.shape[0]
    leaking = np.asarray(S.sum(axis=1)).ravel() < 1.0 - tol
    if leaking.all():
        return True
    if not leaking.any():
        return False
    # Reverse edges (j -> i when A_ij > 0), plus a virtual node n pointing to
    # every leaking phase: the phases reachable from n are exactly the ones
    # that can reach a leaking phase in the original graph.
    S.eliminate_zeros()
    rows, cols = S.nonzero()
    src = np.concatenate([cols, np.full(leaking.sum(), n)])
    dst = np.concatenate([rows, np.flatnonzero(leaking)])
    G = sparse.csr_array((np.ones(src.size), (src, dst)), shape=(n + 1, n + 1))
    reached = csgraph.breadth_first_order(G, n, directed=True,
                                          return_predecessors=False)
    return reached.size == n + 1


def check_sub_stochastic_matrix(A, tol: float = EPS) -> bool:
    """
    Check that `A` is sub-stochastic and transient (DISCRETE case).

    It does not exist in MatrixUtils.java: the Java side only provides
    checkSubStochasticVector for alpha) and checkSubGeneratorMatrix
    (for the continuous case). The discrete matrix A is not validated in jphase.

    Conditions:
      1. A_ij >= 0
      2. sum_j A_ij <= 1 for every row
      3. sp(A) < 1

    About condition 3
    -----------------
    Conditions 1 and 2 alone are NOT enough. Counterexample:

        A = [[1.0, 0.0],
             [0.5, 0.4]]

    has row sums (1.0, 0.9), so it is sub-stochastic, but phase 0 never reaches
    absorption: sp(A) = 1, (I - A) is singular and E[X] does not exist.
    sp(A) < 1 is equivalent to absorption happening with probability 1 and to
    (I - A)^{-1} = sum_{k>=0} A^k converging.

    The current check in dtph.py (rows <= 1, at least one < 1) also accepts
    that counterexample.
    """
    if A.shape[0] != A.shape[1] if is_sparse(A) else (
            np.asarray(A).ndim != 2 or np.asarray(A).shape[0] != np.asarray(A).shape[1]):
        return False
    if is_sparse(A):
        if A.nnz and A.min() < -tol:
            return False
        row_sums = np.asarray(A.sum(axis=1)).ravel()
    else:
        A = np.asarray(A, dtype=float)
        if np.any(A < -tol):
            return False
        row_sums = A.sum(axis=1)
    if np.any(row_sums > 1.0 + tol):
        return False
    return _spectral_radius_lt_one(A, tol)


def solve_power(M, k: int, v0=None) -> np.ndarray:
    """
    Compute M^{-k} v0 by solving k linear systems.

    It replaces the Java pattern

        TInv = T.solve(I, ...)          // explicit inverse
        TInv = matPower(TInv, k)        // then raised to the k-th power

    which inverts the matrix explicitly. Solving the system propagates less
    rounding error: inverting and multiplying applies the condition number one
    more time than necessary.

    It is used with:
        M = -A,  v0 = 1            -> continuous, (-A)^{-k} 1
        M = I-A, v0 = A^{k-1} 1    -> discrete,  (I-A)^{-k} A^{k-1} 1
    """
    if k < 0:
        raise ValueError(f"solve_power: k must be >= 0; got {k}.")
    n = M.shape[0]
    v = np.ones(n) if v0 is None else as_vector(v0, "v0").copy()
    if k == 0:
        return v
    if is_sparse(M):
        # Factorized ONCE and reused across the k solves
        lu = spla.splu(sparse.csc_array(M) if _is_sparray(M)
                       else sparse.csc_matrix(M))
        for _ in range(k):
            v = lu.solve(v)
        return v
    M = np.asarray(M, dtype=float)
    for _ in range(k):
        v = np.linalg.solve(M, v)
    return v


def stirling_second(n: int, k: int) -> int:
    """
    Stirling number of the second kind, in exact integer arithmetic.

        S(n, k) = (1/k!) sum_{j=0}^{k} (-1)^{k-j} C(k, j) j^n
    """
    if n < 0 or k < 0:
        raise ValueError("stirling_second: n and k must be >= 0.")
    if k == 0:
        return 1 if n == 0 else 0
    total = sum((-1) ** (k - j) * comb(k, j) * j ** n for j in range(k + 1))
    return total // factorial(k)


def factorial_to_raw(factorial_moments: Sequence[float]) -> float:
    """
    Convert factorial moments into the raw moment of the same order.

        E[X^n] = sum_{k=1}^{n} S(n, k) * E[X(X-1)...(X-k+1)]

    Parameters
    ----------
    factorial_moments : sequence of float
        [E[(X)_1], E[(X)_2], ..., E[(X)_n]] in that order.

    Returns
    -------
    float
        E[X^n], with n = len(factorial_moments).

    Why this is needed
    ------------------
    In the DISCRETE case the closed form yields factorial moments, not raw ones.
    AbstractDiscPhaseVar.moment(k) in Java returns the factorial moment but
    variance() uses it as if it were raw, which makes jphase's discrete
    variance incorrect (it is missing the +E[X] term).

    """
    n = len(factorial_moments)
    if n == 0:
        return 1.0
    return float(
        sum(stirling_second(n, k) * factorial_moments[k - 1] for k in range(1, n + 1))
    )


def format_representation(title: str, alpha, A, matrix_name: str = "A",
                          stats: dict = None) -> str:
    """
    Multi-line textual description of a PH representation.

    It is used by description() in both the continuous and the discrete
    case. In Java each abstract class builds its own string by hand.
    """
    alpha = as_vector(alpha, "alpha")
    n = alpha.shape[0]
    width = 54
    lines = ["_" * width, title, f"Number of phases: {n}", "Vector alpha:"]
    lines.append("\t" + "\t".join(f"{x:8.4f}" for x in alpha))
    a0 = vec0(alpha)
    if abs(a0) > EPS:
        lines.append(f"\talpha_0 (mass at absorption) = {a0:.4f}")
    if is_sparse(A) and n > 12:
        density = A.nnz / (n * n)
        lines.append(f"Matrix {matrix_name}: sparse {type(A).__name__}, "
                     f"nnz={A.nnz:,} ({density:.2%} density)")
    else:
        Ad = to_dense(A)
        lines.append(f"Matrix {matrix_name}:")
        for i in range(n):
            lines.append("\t" + "\t".join(f"{Ad[i, j]:8.4f}" for j in range(n)))
    if stats:
        for key, value in stats.items():
            lines.append(f"{key}: {value:.6f}")
    lines.append("_" * width)
    return "\n".join(lines)
