import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.sdp.dtsdp import dtsdp


#Vector de épocas
E = np.array([i for i in range(1,4)])
# Variables
estados = np.array(["Excelente", "Bueno", "Promedio", "Malo"])    # Estado de la máquina al inicio de la semana t
# Decisiones
A = np.array(["Reemplazar","No Reemplazar"])
# Retornos Inmediatos
R = np.array([[-1000000,100],
              [-100,80],
              [-100,50],
              [-100,10]])
# Matrices de transición
probs = {a:[] for a in A}
matrizReemplazar = np.array([[0, 0, 0, 1],
                             [0.7, 0.3, 0, 0],
                             [0.7, 0.3, 0, 0],
                             [0.7, 0.3, 0, 0]])

matrizNoReemplazar = np.array([[0.7,0.3,0,0],
                               [0,0.7,0.3,0],
                               [0,0,0.6,0.4],
                               [0,0,0,1]])

probs["Reemplazar"] = matrizReemplazar
probs["No Reemplazar"] = matrizNoReemplazar

sdpMaquinas = dtsdp(E,estados,A,probs,R,0.99)
print(sdpMaquinas.solve(minimize = False))