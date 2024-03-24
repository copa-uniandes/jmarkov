#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.phase.dtph import dtph

alpha = np.array([0.05,0.9,0,0.05])
T = np.array([[0.2, 0.7, 0, 0.1], [0.3, 0.2, 0.2,0.1], [0.1, 0, 0.5,0.1],[0.2,0.2,0.2,0.2]])

ph = dtph(alpha, T)
print(ph.alpha)
print(ph.T)

pmf1 = ph.pmf(5)
print(pmf1)
pmf2 = ph.pmf(10)
print(pmf2)
ev = ph.expected_value()
print(ev)

