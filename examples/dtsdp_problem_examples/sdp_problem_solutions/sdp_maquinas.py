import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.sdp.dtsdp import dtsdp


#Vector de épocas
E = np.array([i for i in range(1,4)])
# Variables
estados = np.array(["Excelente","Bueno","Promedio","Malo"]) # Estado de la máquina al inicio de la semana
# Decisiones
A = np.array(["Reemplazar","No Reemplazar"])
# Retornos Inmediatos
R = np.zeros((len(E),len(S),len(A)))
# Recorremos sobre las épocas
for t in range(len(E)):
    # Recorremos sobre los estados:
    for s_index, i in enumerate(S):
        # Recorremos sobre las decisiones:
        for a_index, a in enumerate(A):
            if i=='Excelente' and a=='Reemplazar':
                R[t,s_index,a_index] = -1000
            elif i=='Excelente' and a=='No Reemplazar':
                R[t,s_index,a_index] = 100
            elif(i=="Bueno" and a=="Reemplazar"):
                R[t,s_index,a_index]=-100
            elif(i=="Bueno" and a=="No Reemplazar"):
                R[t,s_index,a_index]=80
            elif(i=="Promedio" and a=="Reemplazar"):
                R[t,s_index,a_index]=-100
            elif(i=="Promedio" and a=="No Reemplazar"):
                R[t,s_index,a_index]=50
            elif(i=="Malo" and a=="Reemplazar"):
                R[t,s_index,a_index]=-100
            elif(i=="Malo" and a=="No Reemplazar"):
                R[t,s_index,a_index]=10

# Matrices de transición
matNoReemplazar = np.array([[0.7,0.3,0,0],
                          [0,0.7,0.3,0],
                          [0,0,0.6,0.4],
                          [0,0,0,1]])

matReemplazar = np.array([[0,0,0,0],
                          [0.7,0.3,0,0],
                          [0.7,0.3,0,0],
                          [0.7,0.3,0,0]])
probs = {}
for t in E:  # Iterar sobre cada época
    decisiones_dict = {}
    for posA, a in enumerate(A):
        if a == "Reemplazar":
            decisiones_dict[a] = matReemplazar
        elif a == "No Reemplazar":
            decisiones_dict[a] = matNoReemplazar
    probs[t] = decisiones_dict

sdpMaquinas = dtsdp(E,estados,A,probs,R,0.9)
print(sdpMaquinas.solve(minimize = False))