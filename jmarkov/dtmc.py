import numpy as np
from scipy import linalg
from markov_chain import markov_chain

class dtmc(markov_chain):

    # number of states in string array form
    n_states:int=1

    # states in string array form
    states:np.array = np.array([1])

    # transition matrix as 2d numpy array
    transition_matrix:np.array
    
    # empty initializer 
    # chain with a single state
    def __init__(self):
        self.n_states = 1
        self.transition_matrix = np.array([1])

    def __init__(self,n:int):
        self.n_states = n
        self.transition_matrix = np.eye(2, dtype=float)

    # initializer with a transition matrix
    def __init__(self,transition_matrix:np.array):
        if not self._check_transition_matrix(transition_matrix):#Lets check if transition matrix is logical (i.e the rows sum 1)
            raise ValueError("the rows of transition matrix do not sum 1 or has non positive values")
        self.n_states=transition_matrix.shape[0]
        self.transition_matrix = transition_matrix

    def _check_transition_matrix(self,M:np.ndarray):
        #Check if a given transition matrix has the condition that for every row the sum of all elements is equal to 1
        vector = np.isclose(np.sum(M, axis = 1),1,1e-5) == True  
        if vector.all():
        #Check if a given transition matrix has the condition that every element of the matrix is between 0 and 1
            if np.all((M >= 0) & (M <= 1)):
                return True
            else:
                return False
        else:
            return False

    def steady_state(self):
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
        #First, lets verify n is integer
        if not isinstance(n,int):
            raise ValueError("The number of transitions n must be integer")
        #Now lets verify the demensions of the vector alpha:
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
        
       

    def is_ergodic(self):
        #TODO determines id the chain is ergodic or not
        print("TODO")
        return True
    



