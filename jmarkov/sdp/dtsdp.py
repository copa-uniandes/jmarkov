import numpy as np
from typing import Dict


class dtsdp():
    """
    Implements an Stochastic Decision Process (SDP)  
    
    The process is defined by its time horizon, number of states, states, number of actions, actions,
    the immediate returns of implementing each action and a transition matrix for each action. 
    The class provides methods to solve SDPs and to compute the expected value of the optimal policy.     
    """
    # time steps in array form
    periods:np.array=np.array([1])
    # number of time steps
    n_periods:int=1
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
    # immediate return matrix as 3d numpy array 
    immediate_returns:np.array = np.array([[[1]]])
    # discount factor as int
    discount_factor:float = 0.9
    # initializer with a transition matrix, immediate returns and discount factor
    def __init__(self,periods:np.array,states:np.array, actions: np.array, transition_matrices:Dict, immediate_returns: np.array,discount_factor:int):
        """
        Creates a markov decision process from its transition matrices, immediate returns and discount factor
        """
        if not self._check_transition_matrices(transition_matrices):#Lets check if transition matrix is logical (i.e the rows sum 1)
            raise ValueError("the rows of transition matrices do not sum 1 or have non positive values")
        if not self._check_immediate_returns(immediate_returns,states,periods,actions): # Lets check if immediate returns are consistent
            raise ValueError("the dimensions of the immediate returns are not coherent")
        if not self._check_discount_factor(discount_factor): # Lets check if discount factor is logical
            raise ValueError("discount factor should be a number between 0 and 1")
        if not self._check_time_period(periods):
            raise ValueError("the time period should be greater than 0")
        self.periods = periods
        self.n_periods = len(periods)
        self.n_actions = len(actions)
        self.n_states=len(states)
        self.transition_matrices = transition_matrices
        self.immediate_returns = immediate_returns
        self.discount_factor = discount_factor
        self.actions = actions
        self.states = states
    def _check_transition_matrices(self, M:Dict):
        """
        Checks that all matrices are stochastic
         
        Checks that all row sums are equal to one and all elements are non negative
        """
        # determines if the transition matrices are logical
        # i.e: if the sum of the rows equals to 1 (is an stochastic matrix)
        # and if all of the elements are positive
        for epoch, matrices in M.items():
            for action, probabilities in matrices.items():
                for state in range(probabilities.shape[0]):
                    row_sum = probabilities[state,].sum()
                    if not np.allclose(row_sum,1,atol=1e-6) and not np.all((probabilities[state,] >= 0) & (probabilities[state,] <= 1)):
                        return False
                    else:
                        return True
    def _check_immediate_returns(self,R:np.array,states:np.array,periods:np.array,actions:np.array):  
        """
        Checks that the immediate returns are valid and dimensionally-coherent
         
        Checks that immediate return array has dimensions length of epochs x length of states x length of actions
        """  
        # checks if the dimensions of immediate returns are coherent
        # expected: len(E)xlen(S) x len(actions)
        if R.shape == (len(periods),len(states),len(actions)):
            return True

    def _check_discount_factor(self,beta:int):
        """
        Checks that the discount factor is valid
         
        Checks that discount factor is a number equal to or greater than 0 and  less than 1
        """  
        # checks if the discount factor is a number between 0 and 1 
        if beta >= 0 and beta <= 1:
            return True
        else: 
            return False

    def _check_time_period(self,periods:int):
        """"
        Checks that the time period is valid

        Checks that the time period is a number greater than 0
        """     
        # checks if the time period is greater than 0
        if len(periods) > 0:
            return True
        else:
            return False
        
    def solve(self, minimize = False):
        """
        Solves SDP's with backward iteration

        Returns the expected value of following the optimal policy at each state and the optimal policy for each state
        """
        E = self.periods
        S = self.states
        A = self.actions
        M = self.transition_matrices
        R = self.immediate_returns
        beta = self.discount_factor
        if minimize == True:
            R = -R
            best_init = 100000
        else:
            best_init = -100000

        # creates an array to save the value of Bellman's equation
        # for each state, for each time period
        Ft_optimo = np.zeros((len(S),len(E)))
        # creates an array to save the optimal decision 
        # for each state, for each time period
        Mat_Dec_optimo = np.empty((len(S),len(E)), dtype=str)

        # to begin backward iteration, it is required to initialize
        # the last time period
        f = R
        for s_index,i in enumerate(S):
            Ft_optimo[s_index,-1] = max(f[-1,s_index])
            dec = int(np.argmax(f[-1, s_index]))
            Mat_Dec_optimo[s_index,-1] = A[dec]
        # start backward iteration
        # iterate through time steps (from second to last, to first)
        for t in range(len(E)-2,-1,-1):
            # iterate through states
            for s_index,i in enumerate(S):
                # initialize the value of bellman's equation
                best_value = best_init
                # initialize the optimal decision
                a_optima = len(A)-1
                # iterate through decisions
                for posA, a in enumerate(A):
                    expected_value = R[t,s_index,posA] + beta*np.sum(M[t+1][a][s_index,:]*Ft_optimo[:,t+1])
                    # check if it promises improvement
                    if expected_value>best_value:
                        # if it improves, update the values
                        best_value = expected_value
                        a_optima = posA
                # update de matrix of bellman equation and optimal decisions
                Ft_optimo[s_index,t] = best_value
                Mat_Dec_optimo[s_index,t] = A[a_optima]  
        if minimize == True:
            Ft_optimo = -Ft_optimo   
        return(Ft_optimo,Mat_Dec_optimo)     