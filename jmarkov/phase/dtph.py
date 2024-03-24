import numpy as np
from scipy import linalg
from scipy import sparse

class dtph():

    # number of phases
    n_phases:int=1

    # substochastic matrix as 2d numpy array
    T:np.array

    # initial phase vector as 1d numpy array
    alpha:np.array
    
    # empty initializer 
    # chain with a single state
    def __init__(self):
        self.n_phases = 1
        self.T = np.array([0.8])
        self.alpha = np.array([1])

    # initializer with only the number of states
    def __init__(self,n:int):
        self.n_phases = n
        self.generator = np.zeros((n,n), dtype=float)
        self.alpha = np.zeros(n, dtype=float)
        self.alpha[0] = 1

    # initializer with both alpha and T
    def __init__(self, alpha:np.array, T:np.array):
        if not self._check_sub_stochastic_matrix(T): #Lets check if transition matrix is logical (i.e the rows sum <=1)
            raise ValueError("the rows of transition matrix do not sum <= 1")
        self.n_phases=T.shape[0]
        self.T = T
        shape = alpha.shape
        if shape[0]!=self.n_phases:
            raise ValueError("The dimensions of alpha vector are incorrect. Its length must coincide with the size of T")
        self.alpha = alpha

    def _check_sub_stochastic_matrix(self, M:np.ndarray):
        #Check if a given rate matrix has the condition that every row sum <= 0, with at least one < 0 
        if np.max(np.sum(M, axis = 1)) <= 1 and np.min(np.sum(M, axis = 1)) < 1:
            return True
        else:
            return False

    def pmf(self,n:int) -> np.float64:
        P=np.linalg.matrix_power(self.T,n)
        probs=self.alpha@P
        return sum(probs)


    def expected_value(self):
        # Compute the expected value as alpha*inv(I - T)*one
        return np.sum(self.alpha@(np.eye(self.n_phases) -  np.linalg.inv(-self.T)))

