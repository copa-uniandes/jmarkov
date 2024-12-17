import unittest
import sys
import os.path
import numpy as np
from numpy.testing import assert_allclose
# context for the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jmarkov.sdp.dtsdp import dtsdp
class TestSolver(unittest.TestCase):
    def test_value_solver(self):
            # number of epochs:
            epochs = np.array([i for i in range(1,4)])
            # states:
            states = np.array(["Excelente","Bueno","Promedio","Malo"])
            # actions
            actions = np.array(["Reemplazar","No Reemplazar"]) 
            # immediate returns:
            immediate_returns = np.zeros((len(epochs), len(states), len(actions)))
            for t in range(len(epochs)): 
                for s_index,i in enumerate(states):
                    for posA,a in enumerate(actions):
                        if(i=="Excelente" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-1000000
                        elif(i=="Excelente" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=100
                        elif(i=="Bueno" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-100
                        elif(i=="Bueno" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=80
                        elif(i=="Promedio" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-100
                        elif(i=="Promedio" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=50
                        elif(i=="Malo" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-100
                        elif(i=="Malo" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=10
            # discount factor:
            discount_factor = 1
            # transition matrices
            transition_matrices = {t:np.zeros((len(actions), len(states), len(states))) for t in epochs}

            matNoReemplazar = np.array([[0.7,0.3,0,0],
                                    [0,0.7,0.3,0],
                                    [0,0,0.6,0.4],
                                    [0,0,0,1]])

            matReemplazar = np.array([[0,0,0,0],
                                    [0.7,0.3,0,0],
                                    [0.7,0.3,0,0],
                                    [0.7,0.3,0,0]])
            transition_matrices = {}
            for t in epochs:  # Iterar sobre cada época
                decisiones_dict = {}
                for posA, a in enumerate(actions):
                    if a == "Reemplazar":
                        decisiones_dict[a] = matReemplazar
                    elif a == "No Reemplazar":
                        decisiones_dict[a] = matNoReemplazar
                transition_matrices[t] = decisiones_dict
            sdp = dtsdp(epochs, states, actions, transition_matrices, immediate_returns, discount_factor)
            result = sdp.solve(minimize = False)[0]
            assert_allclose(result, ([281.1, 194. , 100. ],[210.9, 151. ,  80. ],[108.4,  84. ,  50. ],[ 81.1,  20. ,  10. ]),err_msg="should be ([281.1, 194. , 100. ],[210.9, 151. ,  80. ],[108.4,  84. ,  50. ],[ 81.1,  20. ,  10. ])")
    def test_policy_solver(self):
            # number of epochs:
            epochs = np.array([i for i in range(1,4)])
            # states:
            states = np.array(["Excelente","Bueno","Promedio","Malo"])
            # actions
            actions = np.array(["Reemplazar","No Reemplazar"]) 
            # immediate returns:
            immediate_returns = np.zeros((len(epochs), len(states), len(actions)))
            for t in range(len(epochs)): 
                for s_index,i in enumerate(states):
                    for posA,a in enumerate(actions):
                        if(i=="Excelente" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-1000000
                        elif(i=="Excelente" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=100
                        elif(i=="Bueno" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-100
                        elif(i=="Bueno" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=80
                        elif(i=="Promedio" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-100
                        elif(i=="Promedio" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=50
                        elif(i=="Malo" and a=="Reemplazar"):
                            immediate_returns[t,s_index,posA]=-100
                        elif(i=="Malo" and a=="No Reemplazar"):
                            immediate_returns[t,s_index,posA]=10
            # discount factor:
            discount_factor = 1
            # transition matrices
            transition_matrices = {t:np.zeros((len(actions), len(states), len(states))) for t in epochs}

            matNoReemplazar = np.array([[0.7,0.3,0,0],
                                    [0,0.7,0.3,0],
                                    [0,0,0.6,0.4],
                                    [0,0,0,1]])

            matReemplazar = np.array([[0,0,0,0],
                                    [0.7,0.3,0,0],
                                    [0.7,0.3,0,0],
                                    [0.7,0.3,0,0]])
            transition_matrices = {}
            for t in epochs:  # Iterar sobre cada época
                decisiones_dict = {}
                for posA, a in enumerate(actions):
                    if a == "Reemplazar":
                        decisiones_dict[a] = matReemplazar
                    elif a == "No Reemplazar":
                        decisiones_dict[a] = matNoReemplazar
                transition_matrices[t] = decisiones_dict
            sdp = dtsdp(epochs, states, actions, transition_matrices, immediate_returns, discount_factor)
            result = sdp.solve(minimize = False)[1]
            self.assertTrue(np.array_equal(result, ([['N', 'N', 'R'],['N', 'N', 'R'],['N', 'N', 'R'],['R', 'N', 'R']])))
if __name__ == '__main__':
    unittest.main()