#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.queue.phph1 import phph1
from jmarkov.phase.ctph import ctph

# arrival rate
INTER_ARRIVAL_TIME = 5
lda = 1/INTER_ARRIVAL_TIME
alpha = np.array([1])
T = np.array([[-lda]])
IAT = ctph(alpha, T)

# paramters
lambda1 = 1.3862
mu1 = 3.8158
N = 20

# job service time distribution
mean_task_service_time = (1/lambda1)+(1/mu1)
num_tasks = N
task_service_rate = 1/mean_task_service_time
beta = np.zeros(num_tasks+1)
beta[0] = 1
S = np.zeros((num_tasks+1,num_tasks+1))
for i in range(num_tasks):
    if i==0:
        S[i,i] = -lambda1
        S[i,i+1] = lambda1
    else:
      #S[i,i] = -(num_tasks-i+1)*task_service_rate
      #S[i,i+1] = (num_tasks-i+1)*task_service_rate
      S[i,i] = -(num_tasks-i+1)*mu1
      S[i,i+1] = (num_tasks-i+1)*mu1



S[num_tasks,num_tasks] = -mu1
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

WT = q.wait_time_dist()
print("Waiting time distribution:")
print(f'alpha: {WT.alpha}')
print(f'T: {WT.T}')

