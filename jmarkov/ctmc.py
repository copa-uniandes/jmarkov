import numpy as np
from scipy import linalg
from .markov_chain import markov_chain

class ctmc(markov_chain):

    # number of states in string array form
    n_states:int=1

    # states in string array form
    states:np.array = np.array([1])

    # generator matrix as 2d numpy array
    generator:np.array
    
    # empty initializer 
    # chain with a single state
    def __init__(self):
        self.n_states = 1
        self.generator = np.array([0])

    def __init__(self,n:int):
        self.n_states = n
        self.generator = np.eye(2, dtype=float)

    # initializer with a transition matrix
    def __init__(self,generator:np.array):
        self.n_states=generator.shape[0]
        self.generator = generator


    def steady_state(self):
        # computes the steady state distribution
        if self.generator.any():
            # sets A to be Q and replace first column with ones for the normalizing equation
            A = self.generator
            A[:, 0] =  1
            # sets b as the right hand vector with a 1 for the normalizing equation
            b = np.zeros(self.n_states, dtype=float)
            b[0] = 1
            # solves transposed system as pi is a row vector
            res = linalg.solve(A,b,transposed=True)
            return res
        else:    
            print("Empty generator matrix")
        return 0

    def first_passage_time(self, target:str):
        #TODO computes the expected first passage time to target state
        print("TODO")
        return 0

    def occupation_time(self, nsteps:int):
        #TODO computes the expected occupation time matrix in nsteps steps
        print("TODO")
        return 0

    def is_ergodic(self):
        #TODO determines id the chain is ergodic or not
        print("TODO")
        return True


