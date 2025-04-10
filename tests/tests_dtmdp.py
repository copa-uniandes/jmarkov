import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.mdp.dtmdp import dtmdp

class TestSolver(unittest.TestCase):
    def test_V_value_iteration(self):
        # number of states:
        N = 2
        # states:
        states = np.array([i for i in range(0,N)])
        # actions
        actions = np.array([str(a) for a in range(0,N)]) 
        # immediate returns:
        immediate_returns = np.array([[3, 1], [2, 3]])
        # discount factor:
        discount_factor = 0.8
        # transition matrices
        transition_matrices = {}
        transition_matrices["0"] = np.array([[1/2, 1/2],[1/3, 2/3]])
        transition_matrices["1"] = np.array([[1/4, 3/4],[2/3, 1/3]])

        mdp = dtmdp(states, actions, transition_matrices, immediate_returns, discount_factor)
        result = mdp.solve(0, minimize = True)[0]
        assert_allclose(result, ([7.81249996, 8.74999996]),err_msg="should be ([7.81249996, 8.74999996])")
    def test_policy_value_iteration(self):
        # number of states:
        N = 2
        # states:
        states = np.array([i for i in range(0,N)])
        # actions
        actions = np.array([str(a) for a in range(0,N)]) 
        # immediate returns:
        immediate_returns = np.array([[3, 1], [2, 3]])
        # discount factor:
        discount_factor = 0.8
        # transition matrices
        transition_matrices = {}
        transition_matrices["0"] = np.array([[1/2, 1/2],[1/3, 2/3]])
        transition_matrices["1"] = np.array([[1/4, 3/4],[2/3, 1/3]])

        mdp = dtmdp(states, actions, transition_matrices, immediate_returns, discount_factor)
        result = mdp.solve(0, minimize = True)[1]
        self.assertEqual(result, {0: '1', 1: '0'})
    def test_V_policy_iteration(self):
        # number of states:
        N = 2
        # states:
        states = np.array([i for i in range(0,N)])
        # actions
        actions = np.array([str(a) for a in range(0,N)]) 
        # immediate returns:
        immediate_returns = np.array([[3, 1], [2, 3]])
        # discount factor:
        discount_factor = 0.8
        # transition matrices
        transition_matrices = {}
        transition_matrices["0"] = np.array([[1/2, 1/2],[1/3, 2/3]])
        transition_matrices["1"] = np.array([[1/4, 3/4],[2/3, 1/3]])

        mdp = dtmdp(states, actions, transition_matrices, immediate_returns, discount_factor)
        result = mdp.solve(0, minimize = True)[0]
        assert_allclose(result, ([7.81249996, 8.74999996]),err_msg="should be ([7.81249996, 8.74999996])")
    def test_policy_policy_iteration(self):
        # number of states:
        N = 2
        # states:
        states = np.array([i for i in range(0,N)])
        # actions
        actions = np.array([str(a) for a in range(0,N)]) 
        # immediate returns:
        immediate_returns = np.array([[3, 1], [2, 3]])
        # discount factor:
        discount_factor = 0.8
        # transition matrices
        transition_matrices = {}
        transition_matrices["0"] = np.array([[1/2, 1/2],[1/3, 2/3]])
        transition_matrices["1"] = np.array([[1/4, 3/4],[2/3, 1/3]])

        mdp = dtmdp(states, actions, transition_matrices, immediate_returns, discount_factor)
        result = mdp.solve(0, minimize = True)[1]
        self.assertEqual(result, {0: '1', 1: '0'})
        
if __name__ == '__main__':
    unittest.main()
