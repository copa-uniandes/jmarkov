#!/usr/bin/env python

import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.dtmc import dtmc


P = np.array([[0.2, 0.7, 0, 0.1], [0.3, 0.4, 0.2,0.1], [0.4, 0, 0.5,0.1],[0.25,0.25,0.25,0.25]])

mc = dtmc(P)
print(mc.transition_matrix)

ss = mc.steady_state()
print(ss)
probas=mc.transient_probabilities(3,np.array([0.05,0.9,0,0.05]))
print(probas)


#Let's check the ergodicity of the chain by calculating its period and checking if it is irreducible:
print('Period of the chain:')
print(mc.period())
print('Is the chain irreducible?')
print(mc.is_irreducible())
print('Is the chain ergodic?')
print(mc.is_ergodic())

#Let's check the oergodicity of some other examples of aperiodic chains
print('\n')
print('Let\'s check the ergodicity of some other examples of aperiodic chains')
print('\n')
P = np.array([[0.1, 0.4, 0.5, 0.0, 0.0, 0.0],
              [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
              [0.0, 0.0, 0.1, 0.4, 0.5, 0.0],
              [0.0, 0.0, 0.0, 0.2, 0.3, 0.5],
              [0.5, 0.0, 0.0, 0.0, 0.2, 0.3],
              [0.3, 0.5, 0.0, 0.0, 0.0, 0.2]])
P = dtmc(P)
print('Period of the chain:')
print(P.period())
print('Is the chain irreducible?')
print(P.is_irreducible())
print('Is the chain ergodic?')
print(P.is_ergodic())

print('\n')

P = np.array([[0, 1/3, 2/3],
              [0,0,1],
              [1,0,0]])
P = dtmc(P)
print('Period of the chain:')
print(P.period())
print('Is the chain irreducible?')
print(P.is_irreducible())
print('Is the chain ergodic?')
print(P.is_ergodic())

#Now, let's check the ergodicity of some non-aperiodic chains:
print('\n')
print('Now, let\'s check the ergodicity of some non-aperiodic chains:')
print('\n')

#Chain of period 2:
P = np.array([[0, 0, 0.8, 0.2],
              [0, 0, 0.4, 0.6],
              [0.7, 0.3, 0, 0],
              [0.2, 0.8, 0, 0]])
P = dtmc(P)
print('Period of the chain:')
print(P.period())
print('Is the chain irreducible?')
print(P.is_irreducible())
print('Is the chain ergodic?')
print(P.is_ergodic())

#Chain of period 3:
print('\n')
P = np.array([[0, 0, 0.5, 0.25, 0.25, 0, 0],
              [0, 0, 1/3, 0, 2/3, 0, 0],
              [0, 0, 0, 0, 0, 1/3, 2/3],
              [0, 0, 0, 0, 0, 1/2, 1/2],
              [0, 0, 0, 0, 0, 3/4, 1/4],
              [0.5, 0.5, 0, 0, 0, 0, 0],
              [0.25, 0.75, 0, 0, 0, 0, 0]])
P = dtmc(P)
print('Period of the chain:')
print(P.period())
print('Is the chain irreducible?')
print(P.is_irreducible())
print('Is the chain ergodic?')
print(P.is_ergodic())


#Chain of period 4:
print('\n')
P = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [1, 0, 0, 0]])
P = dtmc(P)
print('Period of the chain:')
print(P.period())
print('Is the chain irreducible?')
print(P.is_irreducible())
print('Is the chain ergodic?')
print(P.is_ergodic())

print('\n')


#Lets try the ocupation time function:
M=mc.occupation_time(10)
print("ocupation_matrix:")
print(M)

F=dtmc(np.array([[0.8,0.2,0],[0,0.2,0.8],[0.8,0.2,0]]))
print("Lets try first passage time in dtcm with this matrix: ")
print(np.array([[0.8,0.2,0],[0,0.2,0.8],[0.8,0.2,0]]))
print("This is the result: ")
print(F.first_passage_time(0))
O=dtmc(np.array([[0.9,0.1],[0.2,0.8]]))
print("Lets try another first passage time in dtcm with this matrix: ")
print(np.array([[0.9,0.1],[0.2,0.8]]))
print("This is the result: ")
print(O.first_passage_time(1))

print("lets try an error creating a dtcm")
#error_mc=dtmc(np.array([[0.5,0.5],[0.7,0.7]]))

print("let's try the methods for absorbing chains")

abs_mc = dtmc(np.array([[0.2,0.3,0.1,0.4],
          [0.1,0.2,0.5,0.2],
          [0,0,1,0],
          [0,0,0,1]]), states = [0,1,2,3])

for i in range(0,2):
    for j in range(0,2):
        print(f'The mean time of being in state {j} given that it started in state {i}, before absorbing is:')
        print(abs_mc.absorbtion_times(target=j, start=i))

for i in range(0,2):
    for j in range(2,4):
        print(f'The probability of being absorbed by state {j} given that it started in state {i} is:')
        print(abs_mc.absorbtion_probabilities(target=j, start=i))