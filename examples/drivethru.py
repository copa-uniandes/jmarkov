#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.ctmc import ctmc

# Creating the Q of a drive Thru with one queue space, mic and window
Q = np.array([[-50,50, 0, 0, 0, 0, 0, 0],
[0, -80, 50, 30, 0, 0, 0, 0],
[0, 0, -30, 0, 30, 0, 0, 0],
[40, 0, 0, -90, 50, 0, 0, 0],
[0, 40, 0, 0, -120, 50, 30, 0],
[0, 0, 40, 0, 0, -70, 0, 30],
[0, 0, 0, 40, 0, 0, -90, 50],
[0, 0, 0, 0, 40, 0, 0, -40]])

mc = ctmc(Q)
print(mc.generator)

ss = mc.steady_state()
print(ss)
print("lets try the transient method:")
a=mc.transient_probabilities(1,np.array([1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8]))
print(a)
