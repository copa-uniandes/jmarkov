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
    # initializer with a transition matrix, immediate returns and discount factor
    def __init__(self,transition_matrices:Dict, immediate_returns: np.array,discount_factor:int):
        if not self._check_transition_matrices(transition_matrices):#Lets check if transition matrix is logical (i.e the rows sum 1)
            raise ValueError("the rows of transition matrices do not sum 1 or have non positive values")
        if not self._check_immediate_returns(immediate_returns,transition_matrices): # Lets check if immediate returns are consistent
            raise ValueError("the dimensions of the immediate returns are not coherent")
        if not self._check_discount_factor(discount_factor): # Lets check if discount factor is logical
            raise ValueError("discount factor should be a number between 0 and 1")
        self.n_actions = len(transition_matrices)
        self.n_states=len(transition_matrices[next(iter(transition_matrices))])
        self.transition_matrices = transition_matrices
        self.immediate_returns = immediate_returns
        self.discount_factor = discount_factor

    def _check_transition_matrices(self,M:Dict):
        # determines if the transition matrices are logical
        # i.e: if the sum of the rows equals to 1 (is an stochastic matrix)
        # and if all of the elements are positive
        if all(np.all((np.array(value) >= 0) & (np.array(value) <= 1)) for value in M.values()) and np.all(np.isclose(np.sum(list(M.values()), axis=2), 1, atol=1e-5)):
            return True
        else:
            return False
        
    def _check_immediate_returns(self,R:np.array, M:Dict):    
        # checks if the dimensions of immediate returns are coherent
        # expected: len(S) x len(actions)
        if R.shape == (len(M),len(list(M.values())[0])):
            return True

    def _check_discount_factor(self,beta:int):
        # checks if the discount factor is a number between 0 and 1 
        if beta >= 0 and beta < 1:
            return True
        else: 
            return False
    