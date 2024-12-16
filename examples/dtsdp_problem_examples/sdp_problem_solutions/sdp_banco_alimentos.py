import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.sdp.dtsdp import dtsdp


#Vector de épocas
E = np.array([i for i in range(1,6)])
# Variables
Sx = np.array(["Baja","Media","Alta"])   # Demanda de alimentos en el refugio A en la época t
Sy = np.array(["Baja","Media","Alta"])   # Demanda de alimentos en el refugio B en la época t
estados = [(i, j) for i in Sx for j in Sy] # Variable conjunta
# Decisiones
A = np.array(["A","B","2"])
# Retornos Inmediatos
R = np.zeros((len(E), len(estados), len(A)))

for t in range(len(E)): 
    for s, (i,j) in enumerate(estados):
        for a in range(len(A)):
            if E[t] == 5 and (i in ["Baja","Media"]) and (j in ["Baja","Media"]):
                R[t,s,a]=1
                
# Matrices de transición
probs = {t:np.zeros((len(A), len(estados), len(estados))) for t in E}

matriz_ra = np.array([[0.31,0.1,0.09,0.124,0.04,0.036,0.186,0.06,0.054],
                        [0.21,0.26,0.03,0.084,0.104,0.012,0.126,0.156,0.018],
                        [0.17,0.25,0.08,0.068,0.1,0.032,0.102,0.15,0.048],
                        [0.186,0.06,0.054,0.3224,0.104,0.0936,0.1116,0.036,0.0324],
                        [0.126,0.156,0.018,0.2184,0.2704,0.0312,0.0756,0.0936,0.0108],
                        [0.102,0.15,0.048,0.1768,0.26,0.0832,0.0612,0.09,0.0288],
                        [0.1364,0.044,0.0396,0.31,0.1,0.09,0.1736,0.056,0.0504],
                        [0.0924,0.1144,0.0132,0.21,0.26,0.03,0.1176,0.1456,0.0168],
                        [0.0748,0.11,0.0352,0.17,0.25,0.08,0.0952,0.14,0.0448]])

matriz_rb = np.array([[0.31,0.124,0.186,0.1,0.04,0.06,0.09,0.036,0.054],
                        [0.186,0.3224,0.1116,0.06,0.104,0.036,0.054,0.0936,0.0324],
                        [0.1364,0.31,0.1736,0.044,0.1,0.056,0.0396,0.09,0.0504],
                        [0.21,0.084,0.126,0.26,0.104,0.156,0.03,0.012,0.018],
                        [0.126,0.2184,0.0756,0.156,0.2704,0.0936,0.018,0.0312,0.0108],
                        [0.0924,0.21,0.1176,0.1144,0.26,0.1456,0.0132,0.03,0.0168],
                        [0.17,0.068,0.102,0.25,0.1,0.15,0.08,0.032,0.048],
                        [0.102,0.1768,0.0612,0.15,0.26,0.09,0.048,0.0832,0.0288],
                        [0.0748,0.17,0.0952,0.11,0.25,0.14,0.0352,0.08,0.0448]])

matriz_ab = np.array([[0.3136,0.112,0.1344,0.112,0.04,0.048,0.1344,0.048,0.0576],
                        [0.2016,0.2912,0.0672,0.072,0.104,0.024,0.0864,0.1248,0.0288],
                        [0.1568,0.28,0.1232,0.056,0.1,0.044,0.0672,0.12,0.0528],
                        [0.2016,0.072,0.0864,0.2912,0.104,0.1248,0.0672,0.024,0.0288],
                        [0.1296,0.1872,0.0432,0.1872,0.2704,0.0624,0.0432,0.0624,0.0144], 
                        [0.1008,0.18,0.0792,0.1456,0.26,0.1144,0.0336,0.06,0.0264],
                        [0.1568,0.056,0.0672,0.28,0.1,0.12,0.1232,0.044,0.0528],
                        [0.1008,0.1456,0.0336,0.18,0.26,0.06,0.0792,0.1144,0.0264],
                        [0.0784,0.14,0.0616,0.14,0.25,0.11,0.0616,0.11,0.0484]])


probs = {}
for t in E:  # Iterar sobre cada época
    decisiones_dict = {}
    for posA, a in enumerate(A):
        if a == "A":
            decisiones_dict[a] = matriz_ra
        elif a == "B":
            decisiones_dict[a] = matriz_rb
        else:
            decisiones_dict[a] = matriz_ab
    probs[t] = decisiones_dict

sdpBancoAlimentos = dtsdp(E,estados,A,probs,R,0.99)
print(sdpBancoAlimentos.solve(minimize = False))