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
R = np.zeros((len(E), len(estados), len(A)))
for t in range(len(E)): 
    for s_index,i in enumerate(estados):
        for posA,a in enumerate(A):
            if(i=="Excelente" and a=="Reemplazar"):
                R[t,s_index,posA]=-1000000
            elif(i=="Excelente" and a=="No Reemplazar"):
                R[t,s_index,posA]=100
            elif(i=="Bueno" and a=="Reemplazar"):
                R[t,s_index,posA]=-100
            elif(i=="Bueno" and a=="No Reemplazar"):
                R[t,s_index,posA]=80
            elif(i=="Promedio" and a=="Reemplazar"):
                R[t,s_index,posA]=-100
            elif(i=="Promedio" and a=="No Reemplazar"):
                R[t,s_index,posA]=50
            elif(i=="Malo" and a=="Reemplazar"):
                R[t,s_index,posA]=-100
            elif(i=="Malo" and a=="No Reemplazar"):
                R[t,s_index,posA]=10

# Matrices de transición
probs = {a:np.zeros((len(E), len(estados), len(estados))) for a in A}

matNoReemplazar = np.array([[0.7,0.3,0,0],
                          [0,0.7,0.3,0],
                          [0,0,0.7,0.3],
                          [0,0,0,1]])

matReemplazar = np.array([[1,0,0,0],
                          [0.7,0.3,0,0],
                          [0.7,0.3,0,0],
                          [0.7,0.3,0,0]])
for t in range(len(E)):
    probs["Reemplazar"][t, :, :] = matReemplazar
    probs["No Reemplazar"][t,:,:] = matNoReemplazar

sdpMaquinas = dtsdp(E,estados,A,probs,R,0.99)
print(sdpMaquinas.solve(minimize = False))