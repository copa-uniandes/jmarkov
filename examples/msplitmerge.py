#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.phph1 import phph1
from jmarkov.phase.ctph import ctph

# arrival rate
lda = 0.1
alpha = np.array([1])
T = np.array([[-lda]])
IAT = ctph(alpha, T)

# job service time distribution
mean_task_service_time = 1
num_tasks = 10
task_service_rate = 1/mean_task_service_time
beta = np.zeros(num_tasks)
beta[0] = 1
S = np.zeros((num_tasks,num_tasks))
for i in range(num_tasks-1):
    S[i,i] = -(num_tasks-i)*task_service_rate
    S[i,i+1] = (num_tasks-i)*task_service_rate
S[num_tasks-1,num_tasks-1] = -task_service_rate
print(f'beta: {beta}')
print(f'S: {S}')
ST = ctph(beta, S)
q = phph1(IAT,ST) 

print("Distributions of the number of entities in system:")
print(q.number_entities_dist())

L = q.mean_number_entities()
print("Mean number of entities in system:")
print(L)

Lq = q.mean_number_entities_queue()
print("Mean number of entities in queue:")
print(Lq)

W = q.mean_time_system()
print("Mean time in system:")
print(W)

Wq = q.mean_time_queue()
print("Mean time in queue:")
print(Wq)
