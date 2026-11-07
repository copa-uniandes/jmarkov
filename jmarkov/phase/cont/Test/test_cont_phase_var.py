"""
Tests de compatibilidad para DenseContPhaseVar.

Este bloque reproduce las pruebas definidas en:
jMarkov/test/jphase/DenseContMomentTest.java

El objetivo es verificar que la implementación Python produzca los mismos
resultados numéricos que la implementación Java original de jPhase.

Tolerancia utilizada por Java: 1e-5.
"""

import numpy as np
import pytest

from DenseContPhaseVar import DenseContPhaseVar


# ============================================================================
# DATOS ORIGINALES DE DenseContMomentTest.java
# ============================================================================

# Matriz subgeneradora T utilizada por el test original de jPhase.
MATRIX = np.array([
    [-6,  2,  1,  2,  0,   0,  1,  0,  0,  0,  0],
    [ 1, -5,  1,  0,  2,   0,  1,  0,  0,  0,  0],
    [ 2,  1, -7,  0,  0,   2,  2,  0,  0,  0,  0],
    [ 2,  0,  0, -9,  2,   1,  0,  1,  3,  0,  0],
    [ 0,  2,  0,  1, -8,   1,  0,  1,  0,  3,  0],
    [ 0,  0,  2,  2,  1, -10,  0,  2,  0,  0,  3],
    [ 0,  0,  0,  0,  0,   0, -2,  2,  0,  0,  0],
    [ 0,  0,  0,  0,  0,   0,  2, -5,  0,  0,  0],
    [ 0,  0,  0,  0,  0,   0,  0,  0, -4,  2,  1],
    [ 0,  0,  0,  0,  0,   0,  0,  0,  1, -3,  1],
    [ 0,  0,  0,  0,  0,   0,  0,  0,  2,  1, -5],
], dtype=float)


# Vector inicial alpha utilizado por el test original.
VECTOR = np.array([
    0.02, 0.04, 0.04, 0.04, 0.08,
    0.08, 0.10, 0.20, 0.04, 0.08, 0.08
], dtype=float)


# Misma tolerancia utilizada en DenseContMomentTest.java.
TOL = 1e-5


# ============================================================================
# FUNCIÓN AUXILIAR
# ============================================================================

def create_var():
    """
    Crea una nueva DenseContPhaseVar para cada prueba.

    Es equivalente al setUp() del test Java:

        var = new DenseContPhaseVar(vector, matrix);

    Se usan copias para poder comprobar posteriormente que las operaciones
    no modifican MATRIX ni VECTOR.
    """
    return DenseContPhaseVar(VECTOR.copy(), MATRIX.copy())


def assert_representation_unchanged(var):
    """
    Comprueba que alpha y T no hayan sido modificados por la operación.

    Esto reproduce conceptualmente los assertTrue("Matrix changed", ...)
    y assertTrue("Vector changed", ...) presentes después de cada test Java.
    """
    np.testing.assert_allclose(var.alpha, VECTOR, atol=TOL, rtol=0)
    np.testing.assert_allclose(var.T, MATRIX, atol=TOL, rtol=0)


# ============================================================================
# TEST 1: EXPECTED VALUE
# Java: testExpectedValue()
# ============================================================================

def test_dense_expected_value():
    var = create_var()

    calculated = var.expected_value()
    java_expected = 0.786212

    assert calculated == pytest.approx(java_expected, abs=TOL)

    assert_representation_unchanged(var)


# ============================================================================
# TEST 2: VARIANCE
# Java: testVariance()
# ============================================================================

def test_dense_variance():
    var = create_var()

    calculated = var.variance()
    java_expected = 0.908712

    assert calculated == pytest.approx(java_expected, abs=TOL)

    assert_representation_unchanged(var)


# ============================================================================
# TEST 3: CV
# Java: testCV()
#
# IMPORTANTE:
# El método CV() de jPhase calcula realmente el squared coefficient of
# variation (SCV), no el CV convencional.
#
# Nuestra implementación mantiene este comportamiento mediante cv()
# por compatibilidad con Java.
# ============================================================================

def test_dense_cv():
    var = create_var()

    calculated = var.cv()
    java_expected = 1.470101

    assert calculated == pytest.approx(java_expected, abs=TOL)

    assert_representation_unchanged(var)


# ============================================================================
# TEST 4: SECOND MOMENT
# Java: testMoment2()
# ============================================================================

def test_dense_moment_2():
    var = create_var()

    calculated = var.moment(2)
    java_expected = 1.526841

    assert calculated == pytest.approx(java_expected, abs=TOL)

    assert_representation_unchanged(var)


# ============================================================================
# TEST 5: THIRD MOMENT
# Java: testMoment3()
# ============================================================================

def test_dense_moment_3():
    var = create_var()

    calculated = var.moment(3)
    java_expected = 4.382719

    assert calculated == pytest.approx(java_expected, abs=TOL)

    assert_representation_unchanged(var)


# ============================================================================
# TEST 6: FOURTH MOMENT
# Java: testMoment4()
# ============================================================================

def test_dense_moment_4():
    var = create_var()

    calculated = var.moment(4)
    java_expected = 16.677425

    assert calculated == pytest.approx(java_expected, abs=TOL)

    assert_representation_unchanged(var)