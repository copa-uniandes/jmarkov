import numpy as np
from dtmc import dtmc


P = np.array([[0.2, 0.8, 0], [0.3, 0.4, 0.3], [0.4, 0, 0.6]])
mc = dtmc(P)
#print(mc.transition_matrix)
ss = mc.steady_state()
probas=mc.transient_probabilities(1,np.array([0.3,0.4,0.3]))
print(probas)

P_neg = np.array([[0.2, -0.8, 0], [0.3, 0.4, 0.3], [0.4, 0, 0.6]])
dtmc(P_neg)

P_sumOver1 = np.array([[0.2, 0.8, 0], [0.309, 0.4, 0.3], [0.4, 0, 0.6]])
dtmc(P_sumOver1)


