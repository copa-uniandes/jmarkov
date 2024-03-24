#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.phase.ctph import ctph


alpha = np.array([1, 0, 0])
T = np.array([[-4, 1, 3], [2, -5, 2], [0, 2, -3]])

ph = ctph(alpha, T)
print(ph.alpha)
print(ph.T)

pdf1 = ph.pdf(0.5)
print(pdf1)

pdf2 = ph.pdf(5)
print(pdf2)

ev = ph.expected_value()
print(ev)
