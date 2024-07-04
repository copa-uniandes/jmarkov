import numpy as np
from scipy import linalg

class dtmdp():

    # number of states as integer
    n_states:int=1

    # states in string array form
    states:np.array = np.array([1])

    # number of actions as integer
    n_actions:int=1

    # actions in string array form
    actions:np.array = np.array([1])

    # transition matrix as dictionary of 2d numpy arrays
    transition_matrix:dict = {"1": np.eye(1, dtype=float)}

    # immediate returns in array form 
    returns:np.array = np.array([1])

    # discount factor
    dfactor:float = 0.8
    
    # empty initializer 
    # chain with a single state
    #TODO
    def __init__(self):
        print("TODO")
        
        #self.n_states = 1
        #self.transition_matrix = np.array([1])

    #def __init__(self,n:int):
    #    self.n_states = n
    #    self.transition_matrix = np.eye(2, dtype=float)

    # initializer with a transition matrix
    #def __init__(self,transition_matrix:np.array):
    #    if not self._check_transition_matrix(transition_matrix):#Lets check if transition matrix is logical (i.e the rows sum 1)
    #        raise ValueError("the rows of transition matrix do not sum 1 or has non positive values")
    #    self.n_states=transition_matrix.shape[0]
    #    self.transition_matrix = transition_matrix

    #TODO
    #def _check_transition_matrix(self,M:np.ndarray):
    #    print("TODO")
        
