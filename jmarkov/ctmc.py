import numpy as np
from scipy import linalg
from scipy import sparse
from .markov_chain import markov_chain

class ctmc(markov_chain):
    """
    Implements a finite continuous-time Markov chain (CTMC) 
    
    The chain is defined by its number of states, states, and a 
    generator matrix. The class provides methods to compute both
    stationary and transient metrics, as well as to check the 
    chain properties (ergodicity). 
    """

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

    # initializer with only the number of states
    def __init__(self,n:int):
        self.n_states = n
        self.generator = np.zeros((n,n), dtype=float)

    # initializer with a generator matrix
    def __init__(self,generator:np.array):
        """
        Creates a continuous-time Markov chain from its generator matrix
        """
        if not self._check_generator_matrix(generator):#Lets check if transition matrix is logical (i.e the rows sum 0)
            raise ValueError("the rows of generator matrix do not sum 0 or the diagonal has non negative values")
        self.n_states=generator.shape[0]
        self.generator = generator


    def _check_generator_matrix(self,M:np.ndarray,tol:1e-14):
        """
        Checks that a matrix is a Markov chain generator
         
        Checks that all row sums are equal to zero, diagonal entries are non-positive, and 
        non-diagonal entries are non-negative
        """
        # Check if the sum of the rows of the transition matrix equals (or is sufficiently close) to zero 
        check1 = np.max(np.sum(M, axis = 1)) <= tol and np.min(np.sum(M,axis=1)) >= tol
        # Check if a given transition matrix has all diagonal elements non positive
        if check1 and np.all(np.diag(M)<0):
            return True
        else:
            return False


    def steady_state(self) -> np.ndarray:
        """
        Computes the steady state distribution of the continuous-time Markov chain

        Computes the steady state probability distribution by replacing one of the 
        matrix equations with a normalizing equation that ensures the result is a 
        probability distribution. 

        Returns the stationary probability distribution in array form
        """

        # 
        if self.generator.any():
            # sets A to be Q and replace first column with ones for the normalizing equation
            A = self.generator.copy()
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
    

    def transient_probabilities(self,t:float,alpha:np.ndarray) -> np.ndarray:
        """
        Computes the transient distribution at time t with initial state alpha

        Computes alpha*exp(Q*t)*ones to obtain the probability distribution
        at time t given the initial probability distribution alpha
        """
        shape=alpha.shape#lets get the shape of alpha vector
        if shape[0]!=self.n_states:
            raise ValueError("The dimensions of alpha vector are incorrect. It must be a vector 1xn_states")
        P=linalg.expm(self.generator*t)#exponentiate the diferential generator following the method of Mohy et al (https://eprints.maths.manchester.ac.uk/1300/1/alhi09a.pdf)
        probs=alpha@P
        return probs


    def first_passage_time(self, target:str):
        """
        Computes the expected first passage time to a target state from a start state.

        This method calculates the expected number of steps required for the Markov chain to reach
        the specified target state from the start state by creating a sub-matrix of the generator matrix with the target removed.

        Returns the expected steps to reach the target state from any start state (except target) in array form        
        """
        #The transition matrix and the number of states are brought
        e=self.n_states 
        q=self.generator.copy()
        
        # Matrix of ones with n-1 rows is created
        u=np.full([e-1,1],1)

        #The column and row corresponding to the target state are eliminated from the transition matrix 
        m=np.delete(q,target, axis=0)
        m=np.delete(m,target, axis=1)


        t=np.matmul(np.linalg.inv(-m),u)
        return t

    def occupation_time(self,T,Epsilon=0.00001):
        """
        Computes the expected occupation time in each state until time T.

        This method computes the expected time the chain spends in each state,
        from time 0 until time T. To this end it uses the embedded matrix and 
        the uniformization method. 

        Returns the expected time in each state from 0 to T   
        """
        n=self.n_states
        m=self.generator.copy()
        def P_from_R(n,m):            
            # Inicializamos el valor máximo con el primer elemento de la diagonal
            Vector_ri=np.diagonal(m)   
            #Hallamos la r como el máximo de las r_i
            r=max(-Vector_ri)
            return r 
        def creacion_P(n,m):
            r= P_from_R(n,m)
            #Hallamos la matriz P^        
            h=np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i==j:
                        h[i][j]=1+(m[i][j]/r)
                    else:
                        h[i][j]=m[i][j]/r
            return h 
        r=P_from_R(n,m)
        P=creacion_P(n,m)
        i=np.identity(n)
        A=P
        k=0
        yek=np.exp(-r*T)
        ygk=1-yek
        suma=ygk
        B=ygk*i
        while suma/r < T-Epsilon:
            k=k+1
            yek=yek*(r*T)/k
            ygk=ygk-yek
            B=B+ygk*A
            A=A@P
            suma=suma+ygk
        return (B/r)
    

    def is_irreducible(self):
        """
        Checks if the given continous-time Markov Chain is irreducible.
        
        This method determines if the continous-time Markov Chain is irreducible by checking if, starting in
        any state, it is possible to reach any other state in a sequence of transitions.

        Returns a boolean
        """
        # the finite case:

        # We bring the generator matrix and replace all the diagonal elements for 0
        Q = np.copy(self.generator)
        np.fill_diagonal(Q,0)
        # We check if all the components in the chain are strongly connected
        if sparse.csgraph.connected_components(Q, directed=True,connection='strong',return_labels=False)==1:
            return True
        else:
            return False

    def is_ergodic(self):
        """
        Checks if a given continous-time Markov Chain is ergodic.
        
        Given that a finite continous-time Markov Chain is ergodic if it is irreducible, this
        method checks if it is irreducible to determine if the provided chain is ergodic or not.
        """
        # We check if the chain is irreducible
        if self.is_irreducible()==True:
            return True
        else:
            return False
