#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.mmkn import mmkn

# M/M/1/5

lda = 1 
mu = 2
k = 1
n = 5

q = mmkn(k,lda,mu,n) 
L = q.mean_number_entities()
print("Mean number of entities in system:")
print(L)

Lq = q.mean_number_entities_queue()
print("Mean number of entities in queue:")
print(Lq)

Ls = q.mean_number_entities_service()
print("Mean number of entities in service:")
print(Ls)

W = q.mean_time_system()
print("Mean time in system:")
print(W)

Wq = q.mean_time_queue()
print("Mean time in queue:")
print(Wq)

Ws = q.mean_time_service()
print("Mean time in service:")
print(Ws)

# M/M/2/5

lda = 1 
mu = 2
k = 2
n = 5

q = mmkn(k,lda,mu,n) 
L = q.mean_number_entities()
print("Mean number of entities in system:")
print(L)

Lq = q.mean_number_entities_queue()
print("Mean number of entities in queue:")
print(Lq)

Ls = q.mean_number_entities_service()
print("Mean number of entities in service:")
print(Ls)

W = q.mean_time_system()
print("Mean time in system:")
print(W)

Wq = q.mean_time_queue()
print("Mean time in queue:")
print(Wq)

Ws = q.mean_time_service()
print("Mean time in service:")
print(Ws)
