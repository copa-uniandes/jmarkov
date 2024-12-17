import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.sdp.dtsdp import dtsdp

#Vector de épocas
E = np.array([i for i in range(1,31)])

# Variables
Sx = np.array([i for i in range(90000,-9000,-9000)])   
Sy = np.array([i for i in range(0,33,3)])   
Sw = np.array([i for i in range(3000,18000,3000)]) 
estados = [(i, j, k) for i in Sx for j in Sy for k in Sw if (3000*j<=90000-i)] # Variable conjunta

# Decisiones
A = np.array([3,0,-3])

# Retornos Inmediatos
R = np.zeros((len(E),len(estados), len(A)))

#(250, 300, 350, 400, 450)
dividendos = {}
dividendos[3000] =250 
dividendos[6000] =300 
dividendos[9000] =350 
dividendos[12000] =400 
dividendos[15000] =450 

def retornos(i,j,k,a):
    retorno = -1e8
    if j>=-1*a and i>=a*k:
        retorno = j*dividendos[k] - a*k
    return retorno
for t in E:
    for posS, s in enumerate(estados):
        i = s[0]
        j = s[1]
        k = s[2]
        for posA, a in enumerate(A):
            R[t-1,posS,posA] = retornos(i,j,k,a)

# Matrices de probabilidades de transición
p1 = np.array([
    [0.333, 0.555, 0.111, 0, 0],
    [0.222, 0.444, 0.277, 0.055, 0],
    [0.181, 0.181, 0.272, 0.363, 0],
    [0, 0.222, 0.222, 0.222, 0.333],
    [0, 0, 0, 0.75, 0.25]
])
p2 = np.array([
    [0.2, 0.4, 0.4, 0, 0],
    [0.083, 0.416, 0.333, 0.166, 0],
    [0.125, 0.1875, 0.375, 0.25, 0.0625],
    [0, 0.153, 0.230, 0.384, 0.230],
    [0, 0, 0.2, 0.4, 0.4]
])
p3 = np.array([
    [0.25, 0.25, 0.5, 0, 0],
    [0.166, 0.1667, 0.333, 0.333, 0],
    [0.083, 0.25, 0.166, 0.333, 0.166],
    [0, 0.0625, 0.25, 0.25, 0.4375],
    [0, 0, 0.153, 0.461, 0.384]
])

sz_indices = {value: idx for idx, value in enumerate(Sw)}

# Probabilidades de transición
probs = {}
for t in E:  
    decisiones_dict = {}
    if t <= 10:
        matriz = p1
    elif t >10 and t <= 20:
        matriz = p2
    else: 
        matriz = p3
    for posA, a in enumerate(A):
        matriz_transicion = np.zeros((len(estados), len(estados)))
        for fila, estado_inicial in enumerate(estados):
            i = estado_inicial[0]
            j = estado_inicial[1]
            k = estado_inicial[2]
            for columna, estado_futuro in enumerate(estados):
                i_ = estado_futuro[0]
                j_ = estado_futuro[1]
                k_ = estado_futuro[2]
                
                if(j>=-1*a and i>=a*k and i_ == i - max(0,a)*k and j_ == j+a):
                    k_idx = sz_indices[k]
                    k_idx_ = sz_indices[k_]
                    matriz_transicion[fila, columna] = matriz[k_idx][k_idx_]
        decisiones_dict[a] = matriz_transicion
    probs[t] = decisiones_dict

sdpBitcoin = dtsdp(E,estados,A,probs,R,0.99)
print(sdpBitcoin.solve(minimize = False))