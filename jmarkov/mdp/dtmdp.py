import numpy as np
from typing import Dict
import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from jmarkov.dtmc import dtmc

class dtmdp():
    """
    Implements a Markov decision process (MDP)  
    
    The process is defined by its number of states, states, number of actions, actions,
    the immediate returns of implementing each action and a transition matrix for each action. 
    The class provides methods to solve MDPs and to compute the expected value of the optimal policy. 
    """
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
    def __init__(self,states:np.array, actions: np.array, transition_matrices:Dict, immediate_returns: np.array,discount_factor:int):
        """
        Creates a markov decision process from its transition matrices, immediate returns and discount factor
        """
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
        if all(np.all((np.array(value) >= 0) & (np.array(value) <= 1)) for value in M.values()) and np.all(np.isclose(np.sum(list(M.values()), axis=2), 1, atol=1e-5)):
            return True
        else:
            return False
        
    def _check_immediate_returns(self,R:np.array, M:Dict):  
        """
        Checks that the immediate returns are valid and dimensionally-coherent
         
        Checks that immediate return array has dimensions length of states x length of actions
        """  
        # checks if the dimensions of immediate returns are coherent
        # expected: len(S) x len(actions)
        if R.shape == (len(list(M.values())[0]),len(M)):
            return True

    def _check_discount_factor(self,beta:int):
        """
        Checks that the discount factor is valid
         
        Checks that discount factor is a number equal to or greater than 0 and  less than 1
        """  
        # checks if the discount factor is a number between 0 and 1 
        if beta >= 0 and beta < 1:
            return True
        else: 
            return False
    
    def solve(self, tolerance, minimize = False, method = "value_iteration"):
        """
        Solves MDP's with defined method

        Returns the expected value of following the optimal policy at each state and the optimal policy for each state
        """
        S = self.states
        A = self.actions
        M = self.transition_matrices
        R = self.immediate_returns
        beta = self.discount_factor

        # if the sense of the search is minimize, change the values of the immediate returns
        # so that we maximize minimums
        if minimize == True:
            R = -1*R
        
        if method == "value_iteration":
            result = value_iteration(S, A, M, R, beta, tolerance)
            V = result[0]
            optimal_policy = result[1]
        elif method == "policy_iteration":
            result = policy_iteration(S, A, M, R, beta, tolerance)
            V = result[0]
            optimal_policy = result[1]

        if minimize == True:
            V = -1 * V
        # return the optimal policy
        return V, optimal_policy
    
    def policy_transition_matrix(self, policy):
        """
        Calculates the expected value of a policy according 

        Returns the transition matrix for following the optimal policy
        """
        # gets the transition matrix for the optimal policy
        S = self.states
        M = self.transition_matrices
        optimal_transition_matrix = []
        for i in range(0, len(S)):
            optimal_transition_matrix.append(M[policy[S[i]]][i])
        optimal_transition_matrix = np.array(optimal_transition_matrix)
        return optimal_transition_matrix
        
    def expected_policy_value(self, value_function, policy):
        """
        Calculates the expected value of a policy according to the steady state probability of being in each state

        Returns the expected value of following the optimal policy
        """
        # gets the transition matrix for the optimal policy
        optimal_transition_matrix = self.policy_transition_matrix(policy)
        # creates the dtmc object
        mc = dtmc(optimal_transition_matrix)
        expected_value = sum(mc.steady_state()*value_function)
        return expected_value
    
def value_iteration(S, A, M, R, beta, tolerance):
    # initialize value functions
    V = np.zeros(len(S))
    # initialize optimal policy
    optimal_policy = {S[i]: A[0] for i in range(0, len(S))}
    # iterate while there is no improvement
    while True:
        # save values from previous iteration
        oldV = V.copy()
        # iterate through states
        for i in range(0, len(S)):
            # initialize Q -> value-action function
            Q = {}
            # iterate through actions
            for a in range(0,len(A)):
                # evaluate the new value function
                Q[A[a]] = R[i, a] + beta*sum(M[A[a]][i,j]*oldV[j] for j in range(0, len(S)))
                # update the new value function for each state
                V[i] = max(Q.values())
                # update the action for each state
                optimal_policy[S[i]] = max(Q, key = Q.get)

        # if there is no improvement break the cycle
        if np.allclose(oldV, V, tolerance):
            break
    return V, optimal_policy

def policy_improvement(V, S, A, M, R, beta):
    # initialize optimal policy
    optimal_policy = {S[i]: A[0] for i in range(0, len(S))}
    # iterate through states
    for i in range(0, len(S)):
        # initialize Q -> value-action function
        Q = {}
        # iterate through actions
        for a in range(0, len(A)):
            # evaluate the new value function
            Q[A[a]] = R[i,a] + beta*sum(M[A[a]][i,j]*V[j] for j in range(0, len(S)))
        # update the action for each state
        optimal_policy[S[i]] = max(Q, key = Q.get)
    return optimal_policy

def policy_evaluation(policy, S, A, M, R, beta, tolerance):
    # initialize value functions
    V = np.zeros(len(S))
    # iterate while there is improvement
    while True:
        # save old values of value functions
        oldV = V.copy()
        # iterate through states
        for i in range(0,len(S)):
            # save current policy
            a = np.where(A == policy[S[i]])[0][0]
            # evaluate the current policy for each state
            V[i] = R[i,a] + beta*sum(M[A[a]][i,j]*V[j] for j in range(0,len(S)))
        # if there is no improvement break the cycle    
        if np.allclose(oldV, V, tolerance):
            break
    return V

def policy_iteration(S, A, M, R, beta, tolerance):
    # initialize optimal policy
    optimal_policy = {S[i]: A[0] for i in range(0, len(S))}
    # iterate while there is improvement
    while True:
        # save values of current policy
        old_policy = optimal_policy.copy()
        # evaluate current policy
        V = policy_evaluation(optimal_policy, S, A, M, R, beta, tolerance)
        # improve current policy
        optimal_policy = policy_improvement(V, S, A, M, R, beta)
        # if old policy and new policy are the same (there is no improvement) break cycle 
        if all(old_policy[S[i]] == optimal_policy[S[i]] for i in range(0, len(S))):
            break
    return V, optimal_policy
