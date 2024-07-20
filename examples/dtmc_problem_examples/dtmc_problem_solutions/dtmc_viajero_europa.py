import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.dtmc import dtmc

estados = np.array(["Fr", "Al","In", "Es"])

matriz = np.array([[0.5, 0.2, 0, 0.3],
                   [0.15, 0.55, 0.15, 0.15],
                   [0.25, 0.25, 0.25, 0.25],
                   [0, 0.3, 0.7, 0]])

viajero = dtmc(matriz)