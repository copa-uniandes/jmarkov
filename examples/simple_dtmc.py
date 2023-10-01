#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.dtmc import dtmc


P = np.array([[0.2, 0.8, 0], [0.3, 0.4, 0.3], [0.4, 0, 0.6]])

mc = dtmc(P)
print(mc.transition_matrix)

ss = mc.steady_state()
print(ss)