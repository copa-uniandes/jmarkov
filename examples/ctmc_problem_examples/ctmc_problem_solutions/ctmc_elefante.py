import sys
import os.path
import numpy as np
# context for the examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from jmarkov.ctmc import ctmc

lambdaLlegada = 0.5
mu = 1/4
theta = 1/45
omega = 1/10

# Espacio de estados
Sx = [i for i in range(0,7)] # [0, 1, 2, 3, 4, 5, 6, 7]
Sy = ["R","W"] 

estados = []
for i in Sx:
    for j in Sy:
        estado = ",".join((str(i),str(j)))
        estados.append(estado)
estados = np.array(estados)

matriz = []
for estado_actual in estados:
    i = int(estado_actual[0])
    j = estado_actual[-1]
    listaAux = []
    for estado_futuro in estados:
        i_ = int(estado_futuro[0])
        j_ = estado_futuro[-1]
        if i_==i+1 and j_==j and i<6:
            listaAux.append(lambdaLlegada)
        elif i_==i-1 and j_==j and i>0 and j=="W":
            listaAux.append(mu)
        elif i_==i and j_=="R" and j=="W":
            listaAux.append(theta)
        elif i_==i and j_=="W" and j=="R":
            listaAux.append(omega)
        else:
            listaAux.append(0)
    matriz.append(listaAux)

for estado_actual in range(0, len(estados)):
    for estado_futuro in range(0, len(estados)):
        if estado_actual == estado_futuro:
            matriz[estado_actual][estado_futuro]=-sum(matriz[estado_actual])

matriz = np.array(matriz)

elefante = ctmc(matriz)
