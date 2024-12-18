#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.mmk import mmk

lda = 1 
mu = 2
k =3

q = mmk(k,lda,mu) 
L = q.mean_number_entities()

print("Mean number of entities in system:")
print(L)

Lq = q.mean_number_entities_queue()
print("Mean number of entities in queue:")
print(Lq)
