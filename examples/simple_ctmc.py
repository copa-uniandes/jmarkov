#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.ctmc import ctmc


Q = np.array([[-4, 1, 3], [2, -5, 3], [1, 2, -3]])

mc = ctmc(Q)
print(mc.generator)

ss = mc.steady_state()
print(ss)
print("lets try the transient method:")
a=mc.transient_probabilities(1,np.array([0.1,0.5,0.4]))
print(a)

F=ctmc(np.array([[-3,2,1],[2,-5,3],[1,1,-2]]))
print("Lets try first passage time in ctcm with this matrix: ")
print(np.array([[-3,2,1],[2,-5,3],[1,1,-2]]))
print("This is the result: ")
print(F.first_passage_time(0))