import numpy as np
from scipy import linalg
from scipy import sparse
from .markov_chain import markov_chain
import math

class dtmc(markov_chain):
    """
    Implements a finite discrete-time Markov chain (CTMC) 
    
    The chain is defined by its number of states, states, and a 
    transition matrix. The class provides methods to compute both
    stationary and transient metrics, as well as to check the 
    chain properties (aperiodicity and ergodicity). 
    """

    # number of states in string array form
    n_states:int=1

    # states in string array form   
    states:np.array = np.array([1])

    # transition matrix as 2d numpy array
    transition_matrix:np.array
    
    # empty initializer 
    # chain with a single state
    def __init__(self):
        """
        Creates a continuous-time Markov chain from its transition matrix
        """
        self.n_states = 1
        self.transition_matrix = np.array([1])

    # initializer with only the number of states
    def __init__(self,n:int):
        self.n_states = n
        self.transition_matrix = np.eye(n, dtype=float)

    # initializer with a transition matrix
    def __init__(self,transition_matrix:np.array):
        if not self._check_transition_matrix(transition_matrix):#Lets check if transition matrix is logical (i.e the rows sum 1)
            raise ValueError("the rows of transition matrix do not sum 1 or has non positive values")
        self.n_states=transition_matrix.shape[0]
        self.transition_matrix = transition_matrix

    def _check_transition_matrix(self):
        """
        Checks that a matrix is a Markov chain transition matrix
         
        Checks that all entries are non-negative and all row sums are equal to one
        """
        M = self.transition_matrix
        #Check if a given transition matrix has the condition that for every row the sum of all elements is equal to 1
        #Check if a given transition matrix has the condition that every element of the matrix is between 0 and 1
        if np.all(np.isclose(np.sum(M,axis=1),1,1e-5)) and np.all((M>=0) & (M<=1)):
            return True
        else:
            return False
        

    def steady_state(self) -> np.ndarray:
        """
        Computes the steady state distribution of the discrete-time Markov chain

        Computes the steady state probability distribution by replacing one of the 
        matrix equations with a normalizing equation that ensures the result is a 
        probability distribution. 

        Returns the stationary probability distribution in array form
        """
        # computes the steady state distribution
        if self.transition_matrix.any():
            # sets A to be I-P and replace first column with ones for the normalizing equation
            A = self.transition_matrix - np.eye(self.n_states, dtype=float)
            A[:, 0] =  1
            # sets b as the right hand vector with a 1 for the normalizing equation
            b = np.zeros(self.n_states, dtype=float)
            b[0] = 1
            # solves transposed system as pi is a row vector
            res = linalg.solve(A,b,transposed=True)
            return res
        else:    
            print("Empty transition matrix")
        return 0
    

    def transient_probabilities(self,n,alpha):
        """
        Computes the transient distribution at transition n with initial state alpha

        Computes alpha*(P^n)*ones to obtain the probability distribution
        at time/transition n given the initial probability distribution alpha
        """
         
        #First, lets verify n is integer
        if not isinstance(n,int):
            raise ValueError("The number of transitions n must be integer")
        #Now lets verify the dimensions of the vector alpha:
        try:
            shape=alpha.shape
        except:
            print("alpha vector must be a numpy array")
            return None
        if shape[0]!=self.n_states:
            raise ValueError("The dimensions of alpha vector are incorrect. It must be a vector 1xn_states")
        
        #Computes transient_matrix**n
        matrix_n=np.linalg.matrix_power(self.transition_matrix,n)
        vector=alpha@matrix_n
        return vector


    def first_passage_time(self, target:str):
        #The transition matrix and the number of states are brought
        n=self.n_states 
        M=self.transition_matrix.copy()
        # The identity matrix is created with size n-1 states
        i=np.identity(n-1)
        # Matrix of ones with n-1 rows is created
        u=np.full([n-1,1],1)
        # The column and row corresponding to the target state are eliminated from the transition matrix
        p=np.delete(M,target, axis=0)
        p=np.delete(p,target, axis=1)
        
        t=np.matmul(np.linalg.inv(i-p),u)
        return t 

    def occupation_time(self, n:int):
        #computes the expected occupation time matrix in nsteps steps:
        ocupation=np.eye(self.transition_matrix.shape[0],self.transition_matrix.shape[1])#Create an identity matrix with the same shape of transition matrix
        for i in range(1,n+1):
            ocupation+=np.linalg.matrix_power(self.transition_matrix,i)
        return ocupation
    
    def period(self):
        '''
        Initializes the transition matrix and the greatest common divisor for the periodicity calculations
        '''
        P_k = self.transition_matrix
        gcd = 0
        '''
        Loop to elevate the transition matrix to the k th power and evaluate its periodicity:
        '''
        for k in range(2, (self.n_states)+1):
            P_k = np.dot(self.transition_matrix, P_k)
            '''
            If the transition matrix to the k th power has a non-zero element in its diagonal, the chain could have a period k:
            '''
            if np.any(np.diagonal(P_k) != 0):
                '''
                GCD between the current GCD for the period of the matrix and the potential new GCD
                '''
                gcd = math.gcd(gcd, k)
            '''
            If the GCD is 1, there is no need to keep looking for the period, the chain is aperiodic:
            '''
            if gcd == 1:
                break
        return gcd
    
    def is_irreducible(self):
        '''
        We check if the chain is irreducible
        '''
        P = np.copy(self.transition_matrix)
        if sparse.csgraph.connected_components(P, directed=True,connection='strong',return_labels=False)==1:
            return True
        else:
            return False
       

    def is_ergodic(self):
        '''
        A Finite Discrete Time Markov Chain is ergodic if it is aperiodic and irreducible:
        '''
        if self.period() == 1 & self.is_irreducible() == True:
            return True
        else:
            return False
    



