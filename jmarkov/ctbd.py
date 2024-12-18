import numpy as np
from scipy import linalg
from scipy import sparse
from .markov_chain import markov_chain

class ctbd():
    """
    Implements an infinite continuous-time birth-death (CTBD) Markov chain 
    
    The CTBD chain is defined by its birth and death rates, which themselves
    define the number of states, states, and the generator matrix. 
    The representation in this class relies on the birth and death rates between
    states 0 to K, and it is assumed that from state K onwards the birth and death rates
    have the same values.  
    The class provides methods to compute both
    stationary and transient metrics, as well as to check the 
    chain properties (ergodicity). 
    """

    # birth rates states in float array form
    birth_rates:np.array = np.array([1])

    # death rates states in float array form
    death_rates:np.array = np.array([1])
    
    # number of states in string array form
    n_states:int=1

    # states in string array form
    states:np.array = np.array([1])

    # generator matrix as 2d numpy array
    generator:np.array
    
    # empty initializer 
    # BD chain with a single state
    def __init__(self):
        self.birth_rates = np.array([0.5])
        self.death_rates = np.array([1])

    # initializer with birth and death rates up to state K
    def __init__(self,birth_rates:np.array,death_rates:np.array):
        """
        Creates a continuous-time Birth-Death Markov chain from its birth and death rates
        """
        if not self._check_birth_death_rates(birth_rates,death_rates): 
            raise ValueError("The birth and death rate are not positive or do not have the same dimension")
        self.birth_rates = birth_rates
        self.death_rates = death_rates

    def _check_birth_death_rates(self, birth_rates:np.ndarray, death_rates:np.ndarray):
        """
        Checks that all birth and death rates are positive 
         
        Checks that arrays of birth and death rates have the same nonzero size
        """
        if (birth_rates > 0).all() and (death_rates > 0).all() and len(birth_rates) == len(death_rates):
            return True
        else:
            return False


    def steady_state(self, n:int=0) -> np.ndarray:
        """
        Computes the steady state distribution of the continuous-time BD Markov chain

        Computes the steady state probability distribution by solving the balance equations
        to first find pi0 and then the remaining entries of the distribution. 

        n is an optional parameter to specify the length of the steady state distribution 
        vector. It should be at least as large as the size of the birth and death rate arrays. 
        If not specified, the returned vector is equal to the length of the birth and death arrays. 

        Returns the stationary probability distribution in array form
        """
        if self.is_ergodic():
            # Compute cumulative products of birth and death rates
            birth_prod = np.cumprod(self.birth_rates)
            death_prod = np.cumprod(self.death_rates)
            # Compute C factors as division of cumulative products
            c_factors = np.divide(birth_prod,death_prod)
            # Compute pi_0: probability that the chain is empty in steady state
            pi0 = 1/(1+np.sum(c_factors[:-1])+c_factors[-1]*self.death_rates[-1]/(self.death_rates[-1]-self.birth_rates[-1]))
            # Compute remaining entries of pi up to length of birth/death rates arrays
            pi = pi0*c_factors
            pi = np.append(np.array(pi0), pi)
            # Compute remaining entries pi up to n    
            final_factor = self.birth_rates[-1]/self.death_rates[-1]
            if n > len(self.birth_rates):
                for _ in range(n-len(self.birth_rates)-1):
                    pi = np.append(pi, pi[-1]*final_factor)
            return pi
        else:    
            print("Birth-death Markov chain is not ergodic")
        return 0
    

    def is_irreducible(self):
        """
        Checks if the given continous-time BD Markov Chain is irreducible.
        
        This method determines if the birth-death continous-time Markov Chain is irreducible 
        by checking if, starting in any state, it is possible to reach any other state 
        in a sequence of transitions.

        Returns a boolean
        """
        return self._check_birth_death_rates(self.birth_rates, self.death_rates)

    def is_ergodic(self):
        """
        Checks if the given continous-time BD Markov Chain is ergodic.
        
        Given that a finite continous-time Markov Chain is ergodic if it is irreducible, this
        method checks if it is irreducible to determine if the provided chain is ergodic or not.
        """
        # We check if the chain is irreducible
        if self.is_irreducible()==True:
            return self.death_rates[-1] > self.birth_rates[-1]
        else:
            return False
