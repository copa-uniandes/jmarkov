import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.dtmc import dtmc

s1 = 0.4
s2 = 1 - s1

r1 = 0.7
r2 = 1 - r1

estados = [(4,0),(3,1),(2,2),(1,3),(0,4)]

matriz = np.array([[(s1*r1)+s2, s1*r2, 0, 0, 0],
                   [s2*r1, (s1*r1)+(s2*r2), s1*r2, 0, 0],
                   [0, s2*r1, (s1*r1)+(s2*r2), s1*r2, 0],
                   [0, 0, s2*r1, (s1*r1)+(s2*r2), s1*r2],
                   [0, 0, 0, s2*r1, s1+(s2*r2)]])

runner = dtmc(matriz)