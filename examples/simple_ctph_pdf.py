#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.phase.ctph import ctph
import time

# small example
alpha = np.array([0.5, 0.5])
T = np.array([[-2, 0], [0, -5]])
ph = ctph(alpha, T)
t = 0.2

res = ph.pdf([t], unif=False)
print(f"Direct method result: {res}")

res_unif = ph.pdf([t], unif=True)
print(f"Uniformization method result: {res_unif}")



# large example
lda = 1
n = 1000
alpha = np.zeros(n)
alpha[0] = 1
T = np.array([[-2, 0], [0, -5]])
T = np.zeros((n,n))
for i in range(n-1):
    T[i,i] = -lda
    T[i,i+1] = lda
T[n-1,n-1] = -lda
ph = ctph(alpha, T)
x = ph.mean()*3
start_time = time.time()
res = ph.pdf([x], unif=False)
end_time = time.time()
print(f"Direct method result: {res}")
print(f"Time taken: {end_time - start_time}")

start_time = time.time()
res_unif = ph.pdf([x], unif=True)
end_time = time.time()
print(f"Uniformization method result: {res_unif}")
print(f"Time taken: {end_time - start_time}")