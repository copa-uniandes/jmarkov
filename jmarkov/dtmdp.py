import numpy as np
from scipy import linalg
from typing import Dict

class dtmdp():
    # number of states in string array form
    n_states:int=1
    # states in string array form
    states:np.array = np.array([1])
    # number of actions
    n_actions:int=1
    # actions in array form
    actions:np.array = np.array([1])
    # transition matrix as dict of numpy arrays
    transition_matrices:Dict[str, np.array]
    # immediate return matrix as 2d numpy array 
    immediate_returns:np.array = np.array([1])   
    # discount factor as int
    discount_factor:int = 0.8 

    # empty initializer 
    # chain with a single state
    def __init__(self):
        self.n_states = 1
        self.transition_matrix = np.array([1])
    def __init__(self,n:int):
        self.n_states = n
        self.transition_matrices = {"a1": np.eye(1, type=float)}

    # initializer with a transition matrix
    def __init__(self,transition_matrix:np.array):
        if not self._check_transition_matrix(transition_matrix):#Lets check if transition matrix is logical (i.e the rows sum 1)
            raise ValueError("the rows of transition matrix do not sum 1 or has non positive values")
        self.n_states=transition_matrix.shape[0]
        self.transition_matrix = transition_matrix

    def _check_transition_matrix(self,M:np.ndarray):
        #TODO determines if the transition matrices are logical
        print("TODO")
        return True
    
    def _check_immediate_returns(self,M:np.array):
        #TODO determines if the immeadiate return matrix has correct dimensions
        print("TODO")
        return True
    
    def _check_discount_factor(self,beta:int):
        #TODO determines if the discount factor is a number between 0 and 1 
        print("TODO")
        return True
    
    


