#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.ctbd import ctbd

birth = np.array([1])
death = np.array([2])
bd = ctbd(birth, death)

print("Check if the BD chain is ergodic:")
print(bd.is_ergodic())

print("Compute the steady state distribution of the BD chain:")
ss = bd.steady_state()
print(ss)

n=4
print(f"Compute {n} entries of the steady state distribution of the BD chain:")
ssn = bd.steady_state(n)
print(ssn)

ss10 = ss.copy()
for i in range(100):
    ss10 = np.append(ss10, np.array(ss10[-1]*birth[-1]/death[-1]))
#print(ss10)
#print(sum(ss10))


