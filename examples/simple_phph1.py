#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.phph1 import phph1
from jmarkov.phase.ctph import ctph

alpha = np.array([0.9, 0.1])
T = np.array([[-2, 1], [0, -3]])
IAT = ctph(alpha, T)
beta = np.array([0.3, 0.7])
S = np.array([[-5, 2], [0, -4]])
ST = ctph(beta, S)
q = phph1(IAT,ST) 
q._solve_mc(verbose=True)


print("Distributions of the number of entities in system:")
print(q.number_entities_dist())

L = q.mean_number_entities()
print("Mean number of entities in system:")
print(L)

Lq = q.mean_number_entities_queue()
print("Mean number of entities in queue:")
print(Lq)

W = q.mean_time_system()
print("Mean time in the system:")
print(W)

Wq = q.mean_time_queue()
print("Mean time in queue:")
print(Wq)

WT = q.wait_time_dist()
print("Waiting time distribution:")
print(f'alpha: {WT.alpha}')
print(f'T: {WT.T}')


