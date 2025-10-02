#!/usr/bin/env python

import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

rango = np.arange(0,1,0.001)*3*W

WT_pdf = pd.DataFrame(columns=['x','pdf'])
WT_cdf = pd.DataFrame(columns=['x','cdf'])
for x in rango:
    new_pdf = pd.DataFrame([[x, WT.pdf(x)]], columns=['x','pdf'])
    new_cdf = pd.DataFrame([[x, WT.cdf(x)]], columns=['x','cdf'])
    #print(new_pdf.head())
    WT_pdf = pd.concat([WT_pdf, new_pdf])
    WT_cdf = pd.concat([WT_cdf, new_cdf])


WT_pdf.plot(x='x', y='pdf')
plt.xlabel('x')
plt.ylabel('Waiting time pdf')
plt.show()

WT_cdf.plot(x='x', y='cdf')
plt.xlabel('x')
plt.ylabel('Waiting time cdf')
plt.show()


RT = q.resp_time_dist()
print("Response time distribution:")
print(f'alpha: {RT.alpha}')
print(f'T: {RT.T}')

mean_RT = RT.expected_value()
print(f'mean RT: {mean_RT}')
rango = np.arange(0,1,0.001)*3*mean_RT

RT_pdf = pd.DataFrame(columns=['x','pdf'])
RT_cdf = pd.DataFrame(columns=['x','cdf'])
for x in rango:
    new_pdf = pd.DataFrame([[x, RT.pdf(x)]], columns=['x','pdf'])
    new_cdf = pd.DataFrame([[x, RT.cdf(x)]], columns=['x','cdf'])
    #print(new_pdf.head())
    RT_pdf = pd.concat([RT_pdf, new_pdf])
    RT_cdf = pd.concat([RT_cdf, new_cdf])


RT_pdf.plot(x='x', y='pdf')
plt.xlabel('x')
plt.ylabel('Response time pdf')
plt.show()

RT_cdf.plot(x='x', y='cdf')
plt.xlabel('x')
plt.ylabel('Response time cdf')
plt.show()