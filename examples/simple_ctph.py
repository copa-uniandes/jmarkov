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

ev = ph.mean()
print(ev)

m1 = ph.moment(1)
print(f'm1: {m1}')
m2 = ph.moment(2)
print(f'm2: {m2}')
m3 = ph.moment(3)
print(f'm3: {m3}')
m4 = ph.moment(4)
print(f'm4: {m4}')


#### an Erlang 2 example
alpha = np.array([1, 0])
T = np.array([[-1, 1], [0, -1]])
ph = ctph(alpha, T)
print(ph.alpha)
print(ph.T)
m1 = ph.moment(1)
print(f'm1: {m1}')
m2 = ph.moment(2)
print(f'm2: {m2}')
m3 = ph.moment(3)
print(f'm3: {m3}')
m4 = ph.moment(4)
print(f'm4: {m4}')
var = ph.var()
print(f'var: {var}')
std = ph.std()
print(f'std: {std}')

## another example 
alpha = np.array([0.6, 0.4])
T = np.array([[-4, 2], [1, -5]])
ph = ctph(alpha, T)
print(ph.alpha)
print(ph.T)
m1 = ph.moment(1)
print(f'm1: {m1}')
m2 = ph.moment(2)
print(f'm2: {m2}')
m3 = ph.moment(3)
print(f'm3: {m3}')
m4 = ph.moment(4)
print(f'm4: {m4}')
var = ph.var()
print(f'var: {var}')
std = ph.std()
print(f'std: {std}')
