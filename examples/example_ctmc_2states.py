#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.ctmc import ctmc

lda = 1
mu = 2 

Q = np.array([[-lda, lda], [mu, -mu]])

mc = ctmc(Q)
print(f'Q={mc.generator}')

alpha = np.array([1, 0])
print(f'alpha={alpha}')

t= 0.01
pi_t=mc.transient_probabilities(t, alpha)
print(f'pi({t})={pi_t}')

t= 0.1
pi_t=mc.transient_probabilities(t, alpha)
print(f'pi({t})={pi_t}')

t= 0.5
pi_t=mc.transient_probabilities(t, alpha)
print(f'pi({t})={pi_t}')

t= 1
pi_t=mc.transient_probabilities(t, alpha)
print(f'pi({t})={pi_t}')

ss = mc.steady_state()
print(ss)
