#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.dtmc import dtmc


P = np.array([[0.2, 0.7, 0, 0.1], [0.3, 0.4, 0.2,0.1], [0.4, 0, 0.5,0.1],[0.25,0.25,0.25,0.25]])

mc = dtmc(P)
print(mc.transition_matrix)

ss = mc.steady_state()
print(ss)
probas=mc.transient_probabilities(3,np.array([0.05,0.9,0,0.05]))
print(probas)
#Lets try the ocupation time function:
M=mc.occupation_time(10)
print("ocupation_matrix:")
print(M)