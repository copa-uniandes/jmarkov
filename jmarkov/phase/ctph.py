import numpy as np
from scipy import linalg
from scipy import sparse

class ctph():

    # number of phases
    n_phases:int=1

    # subgenerator matrix as 2d numpy array
    T:np.array

    # initial phase vector as 1d numpy array
    alpha:np.array
    
    # empty initializer 
    # chain with a single state
    def __init__(self):
        self.n_phases = 1
        self.T = np.array([-1])
        self.alpha = np.array([1])

    # initializer with only the number of states
    def __init__(self,n:int):
        self.n_phases = n
        self.generator = -1*np.eye(n, dtype=float)
        self.alpha = np.zeros(n, dtype=float)
        self.alpha[0] = 1

    # initializer with both alpha and T
    def __init__(self, alpha:np.array, T:np.array):
        if T.shape[0]!=T.shape[1]:
            raise ValueError("The dimensions of the T matrix are incorrect. It must be a square matrix.")
        self.n_phases=T.shape[0]
        self.T = T
        shape = alpha.shape
        self.alpha = alpha
        if shape[0]!=self.n_phases:
            raise ValueError("The dimensions of alpha vector are incorrect. Its length must coincide with the size of T")
        if not self._check_sub_generator_matrix(T): #Lets check if transition matrix is logical (i.e the rows sum 0)
            raise ValueError("the rows of transition matrix do not sum 0 or the diagonal has non negative values")
        
        

    def _check_sub_generator_matrix(self, M:np.ndarray):
        #Check if a given rate matrix has the condition that every row sum <= 0, with at least one < 0 
        check1 = np.max(np.sum(M, axis = 1)) <= 0 and np.min(np.sum(M, axis = 1)) < 0 
        # Check if a given transition matrix has all diagonal elements non positive
        if check1 and np.all(np.diag(M)<0):
            return True
        else:
            return False

    def pdf(self,t:float) -> np.float64:
        P=linalg.expm(self.T*t)  #exponentiate the diferential generator following the method of Mohy et al (https://eprints.maths.manchester.ac.uk/1300/1/alhi09a.pdf)
        probs=self.alpha@P
        return sum(probs)


    def expected_value(self):
        # Compute the expected value as alpha*inv(T)*one
        return np.sum(self.alpha@np.linalg.inv(-self.T))

