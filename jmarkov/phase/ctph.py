import math
import numpy as np
from scipy import linalg
from scipy import sparse

class ctph():

    # number of phases
    n_phases:int=1

    # subgenerator matrix as 2d numpy array
    T:np.ndarray

    # initial phase vector as 1d numpy array
    alpha:np.ndarray
    
    # initializer with both alpha and T
    def __init__(self, alpha:np.ndarray, T:np.ndarray):
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
        check1 = np.max(np.sum(M, axis = 1)) <= 1e-10 and np.min(np.sum(M, axis = 1)) < -1e-10 
        # Check if a given transition matrix has all diagonal elements non positive
        if check1 and np.all(np.diag(M)<0):
            return True
        else:
            return False

    def pdf(self,t:float) -> float:
        res = 0.0
        if t == 0:
            res = 1 - self.alpha.sum()
        elif t > 0:                 
            P=linalg.expm(self.T*t)  #exponentiate the diferential generator following the method of Mohy et al (https://eprints.maths.manchester.ac.uk/1300/1/alhi09a.pdf)
            res = -self.alpha@P@(self.T).sum(1)
        return res
    
    def cdf(self,t:float,stop:float=0,step:float=0) -> np.ndarray:
        P=linalg.expm(self.T*t)  #exponentiate the diferential generator following the method of Mohy et al (https://eprints.maths.manchester.ac.uk/1300/1/alhi09a.pdf)
        if stop == 0 or step == 0:
            probs= 1 - np.float64(sum(self.alpha@P))
            return np.array(probs)
        elif stop > t and step > 0:
            n = math.ceil((stop-t)/step)
            probs = np.zeros(n)
            probs[0] = 1 - np.float64(sum(self.alpha@P))
            Pstep=linalg.expm(self.T*step)
            Ptemp = P
            for i in range(1,n):
                Ptemp = Ptemp*Pstep
                probs[i] = 1 - np.float64(sum(self.alpha@Ptemp)) 
            return probs
        else:
            return np.array(0.0)

  
    def mean(self):
        # Compute the expected value as alpha*inv(T)*one
        return np.sum(self.alpha@np.linalg.inv(-self.T))
    
    def moment(self, k:int=1)->np.float64:
        # Computes the k-th moment of a CTPH distribution
        result:np.float64 = np.float64(0.0)
        beta = np.ones([self.n_phases,1])
        factk = 1
        for i in range(1,k+1):
            beta = np.linalg.solve(-self.T,beta)
            factk*=i
        result = factk*np.float64(self.alpha@beta)

        return result

    def var(self)->np.float64:
        # compute variance 
        mean = self.moment(1)
        return self.moment(2) - mean*mean

    def std(self)->np.float64:
        # compute standard deviation 
        return np.float64(math.sqrt(self.var()))